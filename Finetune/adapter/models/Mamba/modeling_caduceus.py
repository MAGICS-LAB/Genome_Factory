"""Caduceus model for Hugging Face.
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba, Block
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, MaskedLMOutput, SequenceClassifierOutput

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .configuration_caduceus import CaduceusConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSLMHead, RCPSMambaBlock


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        device=None,
        dtype=None,
):
    """Create Caduceus block.
    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block_cls = RCPSMambaBlock if rcps else Block
    block = block_cls(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out


class CaduceusEmbeddings(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size, config.d_model, config.complement_map, **factory_kwargs
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)


class CaduceusMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)

    def forward(self, inputs_embeds, output_hidden_states=False):
        """Mixer forward."""
        all_hidden_states = []
        hidden_states = inputs_embeds

        residual = None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            # TODO: Add support for gradient checkpointing
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )

        if not self.fused_add_norm:
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states