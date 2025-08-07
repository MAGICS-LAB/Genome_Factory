from functools import cache

import pytorch_lightning as pl
import torch
from GenomeBridge.genomeBridge.Interpretation.SAE.model_wrapper import Model
from GenomeBridge.genomeBridge.Interpretation.SAE.sae_model import SparseAutoencoder, loss_fn


@cache
def get_model(model_name="/projects/p32572/interprot/interprot/nonover_4_mer_BERT"):
    """
    Get cached DNABERT model instance
    """
    model = Model(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    return model


class DNASAELightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model_name = args.model_name
        self.sae_model = SparseAutoencoder(
            d_model=args.d_model,
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            batch_size=args.batch_size,
            dead_steps_threshold=args.dead_steps_threshold,
        )
        self.validation_step_outputs = []

    def forward(self, x):
        return self.sae_model(x)

    def training_step(self, batch, batch_idx):
        seqs = batch["Sequence"]
        batch_size = len(seqs)
        
        with torch.no_grad():
            model = get_model(self.model_name)
            tokens, dna_activations, attention_mask = model.get_layer_activations(seqs)
            
        # Forward pass through SAE - keep exactly the same as original ESM
        recons, auxk, num_dead = self(dna_activations)
        mse_loss, auxk_loss = loss_fn(dna_activations, recons, auxk)
        loss = mse_loss + auxk_loss
        
        # Logging - keep exactly the same as original
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_mse_loss",
            mse_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "train_auxk_loss",
            auxk_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        self.log(
            "num_dead_neurons",
            num_dead,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        val_seqs = batch["Sequence"]
        batch_size = len(val_seqs)
        
        with torch.no_grad():
            model = get_model(self.model_name)
            
        mse_loss_all = torch.zeros(batch_size, device=self.device)
        
        # Running inference one sequence at a time - keep the same as original
        for i, seq in enumerate(val_seqs):
            with torch.no_grad():
                tokens, dna_activations, attention_mask = model.get_layer_activations(seq)
                
                # Calculate MSE - keep exactly the same as original
                recons = self.sae_model.forward_val(dna_activations)
                mse_loss, auxk_loss = loss_fn(dna_activations, recons, None)
                mse_loss_all[i] = mse_loss

        val_metrics = {
            "mse_loss": mse_loss_all.mean(),
        }
        # Return batch-level metrics for aggregation
        self.validation_step_outputs.append(val_metrics)
        return val_metrics

    def on_validation_epoch_end(self):
        # Aggregate metrics across batches - keep the same as original
        avg_mse_loss = torch.stack([x["mse_loss"] for x in self.validation_step_outputs]).mean()

        # Log aggregated metrics
        self.log(
            "avg_mse_loss", avg_mse_loss, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def on_after_backward(self):
        # SAE weight and gradient normalization - keep exactly the same as original
        self.sae_model.norm_weights()
        self.sae_model.norm_grad() 