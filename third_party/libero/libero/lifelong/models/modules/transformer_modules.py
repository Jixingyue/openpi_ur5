import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


###############################################################################
#
# transformer 的构建模块
#
###############################################################################


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # 我们需要合并来自所有注意力头的输出
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                len(mask.shape) == 3
            ):  # 考虑每个批次拥有不同因果掩码的情况，通常对 MAE 实现有用
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # 记住残差连接
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """按样本丢弃路径（随机深度）（当应用于残差块的主路径时）。
    这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，不过，
    原始名称具有误导性，因为 'Drop Connect' 在另一篇论文中是另一种形式的 dropout...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我已选择
    将层和参数名称更改为 'drop path'，而不是将 DropConnect 作为层名并混用
    'survival rate' 作为参数。
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # 适用于不同维度的张量，而不仅仅是 2D 卷积网络
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """按样本丢弃路径（随机深度）（当应用于残差块的主路径时）。"""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
            self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size


###############################################################################
#
# Transformer 解码器（我们的策略仅使用 transformer 解码器）
#
###############################################################################


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
            (self.num_elements is None)
            or (self.seq_len is None)
            or (self.num_elements != input_shape[2])
            or (self.seq_len != input_shape[1])
        ):

            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                torch.triu(torch.ones(self.seq_len, self.seq_len))
                - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # 不进行掩码，仅使用完整注意力
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device
