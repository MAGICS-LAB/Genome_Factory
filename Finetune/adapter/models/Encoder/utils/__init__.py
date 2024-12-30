import copy

import torch.nn as nn

from .warmup_optimizer import WarmupOptimizer
from .positional_encoding import PositionalEncoding


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])