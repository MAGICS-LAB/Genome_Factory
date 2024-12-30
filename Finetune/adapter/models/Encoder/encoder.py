import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .multi_head_attention import MultiHeadAttention
from .utils import clones


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    """The encoder of transformer

    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """

    def __init__(self, d_model: int, d_model_in: int, d_ff: int, d_out: int = 768, n_heads: int = 1, n_layers: int = 1,
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.proj_layer = nn.Linear(d_model, d_model_in)
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model_in, dropout)
        self.feed_forward = FeedForward(d_model_in, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model_in, self.multi_headed_attention, self.feed_forward, dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.dense_layer = nn.Linear(d_model_in, d_out)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_layer(x)
        return self.dense_layer(torch.mean(self.encoder(x), dim=1))