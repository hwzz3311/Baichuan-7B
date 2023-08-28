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

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class BaiChuanConfig(PretrainedConfig):
    model_type = "baichuan"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=64000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        """_summary_

        Args:
            vocab_size (int, optional): _description_. Defaults to 64000.  词表大小
            hidden_size (int, optional): _description_. Defaults to 4096. 隐藏层大小
            intermediate_size (int, optional): _description_. Defaults to 11008. 中间层大小
            num_hidden_layers (int, optional): _description_. Defaults to 32. 隐藏层数量
            num_attention_heads (int, optional): _description_. Defaults to 32. 注意力头数
            hidden_act (str, optional): _description_. Defaults to "silu". 隐藏层激活函数
            max_position_embeddings (int, optional): _description_. Defaults to 4096. 最大位置编码
            initializer_range (float, optional): _description_. Defaults to 0.02. 初始化范围
            rms_norm_eps (_type_, optional): _description_. Defaults to 1e-6. rms标准化的epsilon值，"RMS" 是 "Root Mean Square" 的缩写，指的是均方根（平方根平均值）
            use_cache (bool, optional): _description_. Defaults to True.是否使用缓存
            pad_token_id (int, optional): _description_. Defaults to 0. pad标记
            bos_token_id (int, optional): _description_. Defaults to 1. 开始标记
            eos_token_id (int, optional): _description_. Defaults to 2. 结束标记
            tie_word_embeddings (bool, optional): _description_. Defaults to False. 是否共享词嵌入
        """
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
