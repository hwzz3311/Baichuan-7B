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

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.utils.checkpoint
from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging, add_start_docstrings_to_model_forward, replace_return_docstrings
from xformers import ops as xops

from .configuration_baichuan import BaiChuanConfig

logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    只允许关注过去tokens而不是未来tokens的mask
    """
    bsz, tgt_len = input_ids_shape
    # torch.finfo(dtype).min 极小的负数
    # 创建一个填充了极小负值的掩码张量
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    # 创建掩码条件张量
    mask_cond = torch.arange(mask.size(-1), device=device)
    # 将对角线以下设置为0
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    # 如果 past_key_values_length 大于0，会在掩码的左侧连接一个形状为 (tgt_len, past_key_values_length) 的零张量。这考虑了增量解码时过去的关键值。
    if past_key_values_length > 0:
        # 提问，这里为什么添加零向量？
            # 答：past_key_values表示历史的key和value,是之前生成的tokens的缓存。    
            # 在当前时间步,模型需要attend这些历史tokens以实现自动回归生成。
            # 因此不能使用mask掩盖掉past_key_values,需要将其位置上的mask设置为0。
            # 0表示完全不mask,允许模型attend这些历史tokens。
            # 只有当前时间步的query位置之后需要被mask,以实现解码的顺序生成。
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    # 将掩码张量扩展到匹配注意力机制所需的形状
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    将二维的attention mask扩展为适配多头注意力计算的四维mask
    """
    # src_len就是当前生成句子的长度。
    bsz, src_len = mask.size()
    # tgt_len表示要生成句子的下一步的长度。在解码过程中,tgt_len = src_len + 1。
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    # 生成一个反转的掩码，用于屏蔽非目标区域的注意力
    inverted_mask = 1.0 - expanded_mask
    # 最后生成一个下三角矩阵,只保留一个方向的attention。用极小的负数填充反转掩码的非目标区域，以便在注意力计算时不被考虑
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        RMSNorm使用每个神经元的平方根均值（RMS）来进行归一化。
        具体来说，对于每个神经元的输出向量，RMSNorm将其除以该向量元素的RMS，从而将其归一化。
        这种方法可以看作是对每个神经元的激活函数输入进行缩放，以确保它们在一个合适的范围内。
        需要注意的是，RMSNorm并不像Batch Normalization那样使用小批量数据来计算统计信息，因此不会引入小批量的随机性。
        这使得RMSNorm在一些情况下可能更稳定，但也可能导致一些计算效率上的挑战。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) #  
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RotaryEmbedding(torch.nn.Module):
    """_summary_ 
    旋转位置编码
    TODO 这个class 并没有体现出 旋转位置编码的特性, 这里其实实现的是 绝对位置编码,而不是相对位置编码
    Args:
        torch (_type_): _description_
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """_summary_

        Args:
            dim (_type_): _description_ 编码后的维度
            max_position_embeddings (int, optional): 最大序列长度. Defaults to 2048.
            base (int, optional): 用于计算位置编码的基数. Defaults to 10000.
            device (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim)) # 频率
        self.register_buffer("inv_freq", inv_freq) # 定义一组参数 ,但是不参与参数更新

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        # 计算位置序号
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # 用einsum做矩阵乘法,得到一个max_seq_len_cached x dim的矩阵freqs。这里每个位置的向量就是不同频率余弦函数值组成的编码。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # 
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1) # 分割为正弦和余弦部分 , 可训练的 位置编码
        # emb是一个形状为(max_seq_len, dim)的位置编码矩阵。
        # emb.cos()表示对emb进行余弦运算,结果shape仍为(max_seq_len, dim)。
        # [None, None, :, :]表示在前面添加两个维度,变成一个4维张量,shape为(1, 1, max_seq_len, dim)。
        # 加入这两个维度的None的目的是:
        # 能够与后面输入tensor的shape对齐做矩阵计算。输入x的shape通常是(batch_size, num_heads, seq_len, head_dim)。
        # 加入batch维度和head维度的1, 可以方便的应用到不同的sample和head上。
        # 不需要为batch和head创建新的位置编码,直接使用共享的一个编码矩阵,减少计算和内存。
        # 这种广播机制让位置编码可以高效的应用到不同的sample和head上。
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        """_summary_
        这里实现的是绝对位置编码
        Args:
            x (_type_): _description_
            seq_len (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            # 只有出现更长的seq len 才去更新 只有当输入序列过长时,才需要扩充编码矩阵。
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        # self.cos_cached的完整shape是[1, 1, max_seq_len, dim]。
        # [:, :, :seq_len, ...]表示在第2、3个维度上进行切片, 取出前seq_len个位置的编码。
        # 切片是为了根据具体的seq_len取出相应位置的编码, 而不是每次都计算编码矩阵。
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    对 attention中的QK向量进行后半段的位置旋转
    """
    x1 = x[..., : x.shape[-1] // 2] # 取出了输入x的前半部分,也就是从第0维开始,取到一半位置。
    x2 = x[..., x.shape[-1] // 2:] # 取后半部分
    return torch.cat((-x2, x1), dim=-1) # 将后半部分乘以-1, 表示旋转变换


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """_summary_
    旋转位置编码的具体实现部分

    Args:
        q (_type_): attention 的Q 向量
        k (_type_): _description_ attention 的K 向量
        cos (_type_): _description_ attention 的V 向量 生成的绝对位置编码 的周期函数cos 的值
        sin (_type_): _description_ attention 的V 向量 生成的绝对位置编码 的周期函数sin 的值
        position_ids (_type_): _description_ 位置id

    Returns:
        _type_: _description_
    """
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim] # 去除上面在RotaryEmbedding 添加的batch和head
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    # cos[position_ids]表示根据position_ids中的索引,在cos中取出对应位置的编码向量, 增加一维是为了方便后面的广播
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    # 用q的原始表示与cos相乘,得到编码后的前半部分
    # 用旋转变换后的q/2与sin相乘,得到编码后的后半部分
    # 将两部分相加,即可得到应用了旋转位置编码的q
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Module):
    """
    Transformer中的Feed Forward Network
    Args:
        nn (_type_): _description_
    """
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        """Feed Forward Network

        Args:
            hidden_size (int): 隐藏层
            intermediate_size (int): 中间层
            hidden_act (str): 激活函数
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act] # 激活函数，根据cofig 可知这里使用的是hidden_act="silu", silu(x) = x * sigmoid(x)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    
    """

    def __init__(self, config: BaiChuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # 
        self.num_heads = config.num_attention_heads # 32 个注意力头数
        self.head_dim = self.hidden_size // self.num_heads # 计算每个头的维度
        self.max_position_embeddings = config.max_position_embeddings # 最长位置编码

        if (self.head_dim * self.num_heads) != self.hidden_size:
            # 保证 每个头的维度和 hidden_size 成倍数关系
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.W_pack = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False) # 将输入的隐状态进行升维，生成QKV向量
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False) # 将多头注意力的计算输出进行线性变换，得到最终的多头注意力结果
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings) # TODO 旋转位置编码? 但是没有体现旋转的特性,
        self.cos, self.sin = None, None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """_summary_
        将输入张量重塑为一个特定的形状，以适应多头注意力计算的输入要求。

        Args:
            tensor (torch.Tensor): _description_
            seq_len (int): _description_ 序列长度
            bsz (int): _description_ batch size 

        Returns:
            _type_: _description_
        """
        # tensor.contiguous()的作用是确保张量在内存中的存储是连续
        # 将shape 变换为(batch size, num_heads, seq_len, head_dim)
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """_summary_

        Args:
            hidden_states (torch.Tensor): _description_
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            past_key_value (Optional[Tuple[torch.Tensor]], optional): 用于缓存历史的Key和Value的,它的作用是实现更高效的自动回归生成，(key_states, value_states)。. Defaults to None.
            output_attentions (bool, optional): _description_. Defaults to False.
            use_cache (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: _description_
        """
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        # 将线性层的结果拆分出QKV
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)

        if self.training:  # for training 训练和推理的attention 不一样 , 难道是为了外推 token?
            # 提取出 QKV 向量 并将QK 的shape 变换为 (batch size,num_heads,seq_len,head_dim )
            # 这里做shape变换的主要目的是将Q/K/V按头拆分,变成适合多头注意力计算的形状
            query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim)

            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) # 对V 求出 绝对位置编码
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            query_states = query_states.transpose(1, 2) # 重新恢复为 (batch size, seq_len, num_head, head_dim)
            key_states = key_states.transpose(1, 2)
            # memory_efficient_attention doc : https://facebookresearch.github.io/xformers/components/ops.html?highlight=lowertriangularmask#xformers.ops.fmha.attn_bias.LowerTriangularMask
            # 要求memory_efficient_attention 输入的QKV向量的shape为(batch size, seq_len, num_heads, head_dim)
            # 以下代码等效的 torch实现:
                # scale = 1 / query.shape[-1] ** 0.5
                # query = query * scale
                # attn = query @ key.transpose(-2, -1) # @ 运算符表示矩阵相乘（矩阵乘法）操作
                # if attn_bias is not None:
                #     attn = attn + attn_bias
                # attn = attn.softmax(-1)
                # attn = F.dropout(attn, p)
                # return attn @ value
            # attn_bias 是用于指示哪些位置应该聚焦注意力的掩码
            # 当attn_bias被设置为LowerTriangularMask时,它实现了Transformer中的因果注意力机制(Causal Attention)。
            # 具体来说,LowerTriangularMask会创建一个下三角矩阵,其中下三角的值为0,对角线和上三角的值为一个很大的负数。
            # 这样在计算Attention时,每个位置只能关注到其左侧(过去)的位置,实现了顺序依赖。
            
            attn_output = xops.memory_efficient_attention(
                query_states, key_states, value_states,
                attn_bias=xops.LowerTriangularMask()
            )
            # 再将多头的attn 变换为 (batch size, seq_len,hidden size)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, None

        else:  # for inference 训练和推理的attention 不一样 , 难道是为了外推 token?
            query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            # 同上，旋转位置编码
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # 取出之前的kv向量，加速自回归生成
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            # 是否缓存 之前的kv向量
            past_key_value = (key_states, value_states) if use_cache else None
            # attn_weights 的计算代码其实等价与train 中的xops.memory_efficient_attention，只是实现方式不一样
            # 随着kv 的维度增加，点积结果会随着维度增加而变大，因此 需要对点积的结果进行缩放 （/ math.sqrt(self.head_dim)）
                    # 点积的计算方式是:将两个向量对应位置的元素进行一一乘积,然后将全部乘积结果相加,得到一个标量数量。
                    # 例如:
                    # 向量a = [a1, a2, a3]
                    # 向量b = [b1, b2, b3]
                    # 它们的点积为:
                    # a1 * b1 + a2 * b2 + a3 * b3
                    # 这个数量反映了a和b两个向量的相似性,值越大表示两向量越相似。
            
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                # 将attn_weights中被mask的位置替换为一个非常小的负值，如果直接将位置置0会导致对应的梯度为0,不利于参数更新。
                # 极小的负数，虽然接近0,但不会直接截断梯度。在反向传播中,这里仍会产生非0梯度。
                # 为什么是极小负数而不是极小正数？
                    # 答：负数可以更直接地表示被 Mask 的位置应该对最终结果产生负面的影响。而正数表示被 Mask 的位置可能对结果有正面的影响,这与 Mask 的目的不符。
                    # 使用负数替换可以指导模型通过梯度下降不断减小被 Mask 位置上的数值,从而减少它们的影响力,达到 Mask 的效果。而正数可能会通过梯度上升而不断增大,导致被 Mask 的位置对结果产生更大的(错误的)影响。
                    # 在 Softmax 归一化后,极小的负数接近 0,可以使被 Mask 位置的 Attention Weight 非常小。
                    # 而极小的正数无法确保经过 Softmax 后足够接近 0。
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            # attn_output的shape从[bsz, num_heads, seq_len, head_dim] 转换到 [bsz, seq_len, num_heads, head_dim]的主要目的是为了将multi-head的结果整合到一起,准备进行后续的linear projection
            attn_output = attn_output.transpose(1, 2)
            # 同上，再将多头的attn 变换为 (batch size, seq_len,hidden size)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: BaiChuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 均方层归一化RMSNorm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存推理阶段自回归时的 kV 向量
        """

        residual = hidden_states
        # 先进行一次层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 第一次残差连接 在attention 之后
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # 第二次残差连接 在 mlp（前馈网络）之后
        # 引入残差连接可以缓解深层网络的梯度消失/爆炸问题,有利于训练。
        # Self-Attention和MLP都对输入做了复杂的变换,引入两次残差可以更深层次保留输入信息。

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PreTrainedModel(PreTrainedModel):
    """继承于transformers的

    Args:
        PreTrainedModel (_type_): _description_
    """
    config_class = BaiChuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        """_summary_
        一些初始化的方法
        Args:
            module (_type_): _description_
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Model):
            module.gradient_checkpointing = value


class Model(PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DecoderLayer`]

    Args:
        config: BaiChuanConfig
    """

    def __init__(self, config: BaiChuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id # 填充的token id
        self.vocab_size = config.vocab_size # 词表大小

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) # 构建一个embedding 
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)]) # 根据隐藏层的数量和config 构建多个decode层, 32个num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # 归一化

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # 初始化权重
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """为decoder 层创建 attention mask

        创建因果注意力mask,也称为左右非对称mask。
        如果input超过1个token,则创建一个下三角形mask矩阵,只保留右上方为0,左下方为很小的负数。
        这可以防止当前token关注到未来的信息,实现自回归。
        如果外部传入了attention mask,则将其扩展到与因果mask形状一致。
        如果两者都存在,则相加作为最终的attention mask。
        传入的外部attention mask可以用来mask掉指定位置。
        而左右非对称的因果mask保证了只attend到过去。
        这样可以灵活控制attention范围:

        仅因果mask实现标准解码器。
        额外传入mask实现更复杂控制。
        两者叠加增强效果。

        Args:
            attention_mask (_type_): _description_
            input_shape (_type_): _description_
            inputs_embeds (_type_): _description_
            past_key_values_length (_type_): _description_

        Returns:
            _type_: _description_
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            # 它检查是否存在外部传入的注意力掩码 attention_mask。
            # 如果存在，就调用 _expand_mask 函数将其扩展为适用于多头注意力计算的形状，并将其赋值给 expanded_attn_mask。
            # 然后将这个扩展后的掩码与之前的 combined_attention_mask 相加，以获得综合的注意力掩码。
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            # 检查是否同时提供了 input_ids 和 inputs_embeds，如果是，则抛出一个异常，因为在同一时间只能选择一种方式来指定输入序列。
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果只提供了 input_ids，则获取其形状，即 batch_size（批次大小） 和 seq_length（序列长度）。
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            # 如果只提供了 inputs_embeds，则获取其形状，即 batch_size、seq_length 和 _（嵌入维度）。这里使用下划线 _ 表示嵌入维度，因为在这段代码中，嵌入维度并不会被具体使用。
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            # 如果既没有提供 input_ids 也没有提供 inputs_embeds，则抛出异常，因为需要至少选择一种方式来指定输入序列。
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        # 检查是否提供了过去关键值 past_key_values。
        # 如果提供了，它会获取 past_key_values 列表中的第一个元素的第一个张量（通常是第一层的过去关键值），并从中获取 shape[2]，也就是过去关键值的长度。
        # 这个长度表示模型在之前解码的步骤中已经生成了多少个标记，因此需要在当前解码步骤中考虑这些过去的信息。

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            # 创建 位置编码的id 
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # token id 获取对应的 embed
        # embed positions
        if attention_mask is None:
            # 创建  全为1 的attn mask
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        # 将attn mask 变换为一个下三角形mask矩阵,只保留右上方为0,左下方为很小的负数，用于自回归
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        # embed 就是初始的隐向量
        hidden_states = inputs_embeds
        # 训练模式下 打开缓存，用于加速
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        # 循环进每个decoder层
        for idx, decoder_layer in enumerate(self.layers):
            # 是否需要输出所有的 隐向量
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 取出每个decoder中缓存的 kv 向量
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    # 生成了当前decoder层的自定义前向计算函数,它会返回这个层的输出。
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                # 使用了 PyTorch 的 checkpoint 功能来实现 gradient checkpointing,主要目的是减少GPU显存使用和加速训练。
                # torch.utils.checkpoint.checkpoint允许我们对模型的中间结果进行持久化,从而减少反向传播时的显存开销。
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                # 直接调用decoder 层和torch.utils.checkpoint.checkpoint 有什么区别？
                # 答：节省显存
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # 每一层的第一个元素是 当前层输出的隐状态，作为下一层的输入
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        #  提问：为什么要对最后的隐状态做归一化，而不是每一层decoder做一次归一化？
        # 答：1、做局部归一化不能充分利用所有层的信息。而最后归一化可以利用每一层的输出,对全局信息进行标准化。
            # 2、每层后归一化需要引入N个Norm模块,增加参数和计算量。最后归一化只需一个Norm,计算效率更高。
            # 3、随着层数增多,中间层输入分布变动可能过大,归一化有助于维持稳定分布,防止 representations 退化。

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        # 从最后一个decoder 层中取出隐状态，追加进来。
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        # 返回格式
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BaiChuanForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Model(config)
        # 定义一个线性层lm_head,将decoder输出投影到词表维度上,用来计算语言模型的logits。（语言模型的目标是预测下一个词,这是一个分类问题,类别数就是词表大小。）
        # 提问：为什么设置bias=False？
        # 避免过拟合：语言模型学习的是语言的概率分布,具有普适性。而bias参数容易导致过拟合特定样本。
        # 模型容量：不使用bias可以减少一些参数,降低模型复杂度。
        # 数值稳定性：bias参数的更新需要依赖于批次大小,容易造成数值不稳定。去掉bias可以降低一些训练难度。
        # 标准化的效果：隐状态已经过 LayerNorm 处理,具有一定平滑化作用,类似bias。
        # 加快收敛速度：有研究表明去掉bias可以加快训练的收敛速度。
        # 简化实现：不使用bias也可以降低代码实现复杂度。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        获取输入:可以接受input_ids或者inputs_embeds作为输入,以及attention_mask,position_ids等辅助输入。

        将输入传入Transformer Decoder model完成前向计算,得到隐状态outputs。

        将隐状态传入线性层lm_head,得到语言模型的logits。

        如果传入了labels,计算自回归语言模型的损失:

        将logits和labels向右移一位(预测第i个词时label是第i+1个词)
        用交叉熵计算损失
        根据return_dict配置返回不同格式的输出,包括loss,logits,attentions, hidden_states等。
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ModelForCausalLM

        >>> model = ModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """为Transformer的自动回归生成准备相应的输入

        Args:
            input_ids (_type_): _description_
            past_key_values (_type_, optional): _description_. Defaults to None.
            attention_mask (_type_, optional): _description_. Defaults to None.
            inputs_embeds (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if past_key_values:
            # past_key_values表示使用缓存,即只需要为当前timestep准备输入。
            # 所以将input_ids切片取最后一个token,变为添加一个新token的生成。
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            # 创建位置编码position_ids:
            # 根据attention mask的长度创建位置id。
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果使用缓存,取最后一个位置,表示新增生成的一个位置。
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
