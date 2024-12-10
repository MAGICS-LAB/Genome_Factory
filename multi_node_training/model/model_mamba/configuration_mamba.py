import math
from typing import Optional , Union

from transformers import PretrainedConfig
class MambaConfig(PretrainedConfig):
    model_type = "mamba"
    def __init__(
        self,
        vocab_size=50277,
        d_state=16,
        d_model=2560,
        d_conv=4,
        expand=2,
        conv_bias=True,
        bias=False,
        n_layer=64,
        dt_rank: Union[int, str] = "auto",
        pad_vocab_size_multiple=8,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer= n_layer
        self.conv_bias = conv_bias
        self.expand = expand
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.d_conv = d_conv
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank
        self.initializer_range = initializer_range
        self.bias = bias
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
        super().__init__(
            **kwargs,
        )