import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b