import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        k: int = 128,
        auxk: int = 256,
        batch_size: int = 256,
        dead_steps_threshold: int = 2000,
    ):
        """
        Initialize the Sparse Autoencoder.

        Args:
            d_model: Dimension of the pLM model.
            d_hidden: Dimension of the SAE hidden layer.
            k: Number of top-k activations to keep.
            auxk: Number of auxiliary activations.
            dead_steps_threshold: How many examples of inactivation before we consider
                a hidden dim dead.

        Adapted from https://github.com/tylercosgrove/sparse-autoencoder-mistral7b/blob/main/sae.py
        based on 'Scaling and evaluating sparse autoencoders' (Gao et al. 2024) https://arxiv.org/pdf/2406.04093
        """
        super().__init__()

        self.w_enc = nn.Parameter(torch.empty(d_model, d_hidden))
        self.w_dec = nn.Parameter(torch.empty(d_hidden, d_model))

        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.k = k
        self.auxk = auxk
        self.batch_size = batch_size

        self.dead_steps_threshold = dead_steps_threshold / batch_size

        # TODO: Revisit to see if this is the best way to initialize
        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        self.w_dec.data = self.w_enc.data.T.clone()
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

        # Initialize dead neuron tracking. For each hidden dimension, save the
        # index of the example at which it was last activated.
        self.register_buffer("stats_last_nonzero", torch.zeros(d_hidden, dtype=torch.long))

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply top-k activation to the input tensor.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to apply top-k activation on.
            k: Number of top activations to keep.

        Returns:
            torch.Tensor: Tensor with only the top k activations preserved,and others
            set to zero.

        This function performs the following steps:
        1. Find the top k values and their indices in the input tensor.
        2. Apply ReLU activation to these top k values.
        3. Create a new tensor of zeros with the same shape as the input.
        4. Scatter the activated top k values back into their original positions.
        """
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def LN(
        self, x: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Layer Normalization to the input tensor.

        Args:
            x: Input tensor to be normalized.
            eps: A small value added to the denominator for numerical stability.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized tensor.
                - The mean of the input tensor.
                - The standard deviation of the input tensor.

        TODO: Is eps = 1e-5 the best value?
        """
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def auxk_mask_fn(self) -> torch.Tensor:
        """
        Create a mask for dead neurons.

        Returns:
            torch.Tensor: A boolean tensor of shape (D_HIDDEN,) where True indicates
                a dead neuron.
        """
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        return dead_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Sparse Autoencoder. If there are dead neurons, compute the
        reconstruction using the AUXK auxiliary hidden dims as well.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The reconstructed activations via top K hidden dims.
                - If there are dead neurons, the auxiliary activations via top AUXK
                    hidden dims; otherwise, None.
                - The number of dead neurons.
        """
        x, mu, std = self.LN(x)
        x = x - self.b_pre

        pre_acts = x @ self.w_enc + self.b_enc

        # latents: (BATCH_SIZE, D_EMBED, D_HIDDEN)
        latents = self.topK_activation(pre_acts, k=self.k)

        # `(latents == 0)` creates a boolean tensor element-wise from `latents`.
        # `.all(dim=(0, 1))` preserves D_HIDDEN and does the boolean `all`
        # operation across BATCH_SIZE and D_EMBED. Finally, `.long()` turns
        # it into a vector of 0s and 1s of length D_HIDDEN.
        #
        # self.stats_last_nonzero is a vector of length D_HIDDEN. Doing
        # `*=` with `M = (latents == 0).all(dim=(0, 1)).long()` has the effect
        # of: if M[i] = 0, self.stats_last_nonzero[i] is cleared to 0, and then
        # immediately incremented; if M[i] = 1, self.stats_last_nonzero[i] is
        # unchanged. self.stats_last_nonzero[i] means "for how many consecutive
        # iterations has hidden dim i been zero".
        self.stats_last_nonzero *= (latents == 0).all(dim=0).all(dim=0).long()
        self.stats_last_nonzero += 1

        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts = self.topK_activation(auxk_latents, k=k_aux)

            auxk = auxk_acts @ self.w_dec + self.b_pre
            auxk = auxk * std + mu
        else:
            auxk = None

        return recons, auxk, num_dead

    @torch.no_grad()
    def forward_val(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Sparse Autoencoder for validation.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            torch.Tensor: The reconstructed activations via top K hidden dims.
        """
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def norm_weights(self) -> None:
        """
        Normalize the weights of the Sparse Autoencoder.
        """
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

    @torch.no_grad()
    def norm_grad(self) -> None:
        """
        Normalize the gradient of the weights of the Sparse Autoencoder.
        """
        dot_products = torch.sum(self.w_dec.data * self.w_dec.grad, dim=0)
        self.w_dec.grad.sub_(self.w_dec.data * dot_products.unsqueeze(0))

    @torch.no_grad()
    def get_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the activations of the Sparse Autoencoder.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            torch.Tensor: The activations of the Sparse Autoencoder.
        """
        x, _, _ = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)
        return latents

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        acts = x @ self.w_enc + self.b_enc
        return acts, mu, std

    @torch.no_grad()
    def decode(self, acts: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        latents = self.topK_activation(acts, self.k)

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons


def loss_fn(
    x: torch.Tensor, recons: torch.Tensor, auxk: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the loss function for the Sparse Autoencoder.

    Args:
        x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.
        recons: (BATCH_SIZE, D_EMBED, D_MODEL) reconstructed activations via top K
            hidden dims.
        auxk: (BATCH_SIZE, D_EMBED, D_MODEL) auxiliary activations via top AUXK
            hidden dims. See A.2. in https://arxiv.org/pdf/2406.04093.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The MSE loss.
            - The auxiliary loss.
    """
    mse_scale = 1
    auxk_coeff = 1.0 / 32.0  # TODO: Is this the best coefficient?

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0)
    return mse_loss, auxk_loss
