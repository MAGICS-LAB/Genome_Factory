from typing import Union

import torch
import torch.nn as nn

from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention
from .utils import clones


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward"""

    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor, sublayer: Union[MultiHeadAttention, FeedForward]) -> torch.FloatTensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))