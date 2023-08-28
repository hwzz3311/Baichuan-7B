# Copyright 2023 Baichuan Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import argparse
import pickle
import deepspeed
import deepspeed.comm as dist
import numpy as np
import sentencepiece as spm
import torch

from models.configuration_baichuan import BaiChuanConfig
from models.modeling_baichuan import BaiChuanForCausalLM

# os.environ["CUDA_VISIBLE_DEVICES"] ="-1" # 单张卡无法加载,先用cpu debug 一下代码

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="line_data_dir",
                        help="Text files to do pre-train on")

    parser.add_argument("--tokenizer_path", type=str,
                        default="tokenizer.model",
                        help="Tokenizer model file path")

    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max tokens per sentence in corpus")

    parser.add_argument("--steps_per_epoch", type=int, default=4096,
                        help="Step intervals to save checkpoint")

    parser.add_argument("--checkpoint_saving_path", type=str,
                        default="checkpoints",
                        help="Path to store checkpoint files")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Reserved for deepspeed framework")
    return parser


arg_parser = get_argument_parser()
arg_parser = deepspeed.add_config_arguments(arg_parser) # 添加自定的参数到 deepspeed info
args = arg_parser.parse_args()
args.deepspeed_config = "config/deepspeed.json" # 由于应该使用deepspeed 启动，所以这里先手动传入
deepspeed.init_distributed()

# device_count = torch.cuda.device_count()


class DataEngine():
    def __init__(self, data_dir, tokenizer_path, micro_batch_size, max_length):
        """
        """
        self.MIN_TEXT_LEN = 20
        self.EOS_TOKEN_ID = 2 # "EOS_TOKEN_ID" 是指"End of Sequence Token ID"，即序列结束标记的标识符
        self.data_dir = data_dir
        self.sp = spm.SentencePieceProcessor() # 分词工具，可以扩展词表
        self.sp.Load(tokenizer_path) # 加载本地的分词模型
        self.micro_batch_size = micro_batch_size # 一个setp的batch size
        self.max_length = max_length
        self.data = []
        self.global_input_paths = [self.data_dir + "/" + x
                                   for x in os.listdir(self.data_dir)] # 加载所有数据文件
        # 根据当前进程的排名来选择性地从 self.global_input_paths 中筛选出一部分数据文件的本地路径，以便每个进程只加载和处理其分配的数据文件
        # 这是分布式训练中数据加载和分配的一种常见方法，可以确保每个进程都在训练过程中使用不同的数据子集。
        self.local_input_paths = [x for i, x in
                                  enumerate(self.global_input_paths)
                                  if i % dist.get_world_size() == dist.get_rank()] 
        self.local_input_paths = self.local_input_paths[:2] # 此处为了减少debug 时间,减少数据load 的等待,暂时进行了截取

    def load_data(self):
        # TODO　可以先将 文件转成 token id 的形式,再存为pkl的形式,一个文本文件对应一个pkl文件，这样会减少debug 的时间
        for file_path in self.local_input_paths:
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            file_name_without_extension  = os.path.splitext(file_name)[0]
            pkl_file_name = f"{file_name_without_extension}.pkl"
            pkl_file_path = os.path.join(file_dir,"./",pkl_file_name)
            if os.path.exists(os.path.join(file_dir,pkl_file_name)):
                all_datas = pickle.load(open(pkl_file_path,"rb"))
            else:
                all_datas = []
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    for line_id, line in enumerate(f):
                        cc = self.sp.EncodeAsIds(line.strip()) + [self.EOS_TOKEN_ID] # 文本转 token id
                        all_datas.append(cc)
                pickle.dump(all_datas,open(pkl_file_path,"wb"))
            data = []
            for line_id,cc in enumerate(all_datas):
                data.extend(cc)
                if len(data) >= self.micro_batch_size * (self.max_length + 1):
                    index = self.micro_batch_size * (self.max_length + 1)
                    self.data.append(data[:index])
                    data = []
                if line_id > 1000: # 暂时每个文件只取1万条数据
                    break
        return

    def get_data(self):
        data = self.data.pop(0)
        seq = np.asarray(data).reshape(self.micro_batch_size, self.max_length + 1)
        data = torch.LongTensor(seq)
        data = data.cuda(non_blocking=True)
        return data


def prepare_data():
    data_dir = args.data_dir
    tokenizer_path = args.tokenizer_path
    ds_config = json.load(open(args.deepspeed_config))
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"] # 一个GPU one step 处理的批量大小（无梯度累积）
    max_length = args.max_length
    data_engine = DataEngine(data_dir, tokenizer_path, micro_batch_size, max_length)
    data_engine.load_data()
    return data_engine


def prepare_model():
    with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config,
                             enabled=False,
                             mem_efficient_linear=False,
                             mpu=None):
        model = BaiChuanForCausalLM(BaiChuanConfig())

    model_parameters = filter(lambda p: p.requires_grad, model.parameters()) # 开启权重更新
    model_engine, _, _, _ = deepspeed.initialize(args=args,
                                                 model=model,
                                                 optimizer=None,
                                                 model_parameters=model_parameters)
    return model_engine


def train(data_engine, model_engine):
    model_engine.train()
    step = 0
    while step < args.steps_per_epoch:
        data = data_engine.get_data()
        loss = model_engine(data, labels=data).loss
        model_engine.backward(loss)
        model_engine.step()
        step += 1
    return


if __name__ == "__main__":
    data_engine = prepare_data()
    model_engine = prepare_model()
    epoch = 0
    while True:
        train(data_engine, model_engine)
        epoch += 1
        model_engine.save_checkpoint(f"{args.checkpoint_saving_path}",
                                     tag=f"Epoch-{epoch}")
