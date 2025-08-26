import argparse
import glob
import os

import pytorch_lightning as pl
import wandb
from genomeFactory.Interpretation.SAE.dna_data_module import DNASequenceDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from genomeFactory.Interpretation.SAE.dna_sae_module import DNASAELightningModule

parser = argparse.ArgumentParser()

# Based on original training.py parameters, but adapted for DNA data
parser.add_argument("--data-file", type=str, default="sequences_sae.txt", help="Path to DNA sequences file")
parser.add_argument("--d-model", type=int, default=768, help="DNABERT-2 embedding dimension")
parser.add_argument("--d-hidden", type=int, default=4096, help="SAE hidden dimension")
parser.add_argument("-b", "--batch-size", type=int, default=48, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--k", type=int, default=64, help="Top-k activations")
parser.add_argument("--auxk", type=int, default=256, help="Auxiliary k")
parser.add_argument("--dead-steps-threshold", type=int, default=2000, help="Dead neuron threshold")
parser.add_argument("-e", "--max-epochs", type=int, default=3, help="Max epochs")
parser.add_argument("-d", "--num-devices", type=int, default=1, help="Number of devices")
parser.add_argument("--model-suffix", type=str, default="", help="Model suffix")
parser.add_argument("--wandb-project", type=str, default="dna-interprot", help="Wandb project")
parser.add_argument("--num-workers", type=int, default=None, help="Number of workers")

args = parser.parse_args()

# Based on original output directory structure
args.output_dir = (
    f"dna_results_dim{args.d_hidden}_k{args.k}_auxk{args.auxk}_{args.model_suffix}"
)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Based on original model naming
sae_name = (
    f"dnabert2_768_sae{args.d_hidden}_"
    f"k{args.k}_auxk{args.auxk}_{args.model_suffix}"
)

# Based on original wandb setup
wandb_logger = WandbLogger(
    project=args.wandb_project,
    name=sae_name,
    save_dir=os.path.join(args.output_dir, "wandb"),
)

# Use DNA SAE module
model = DNASAELightningModule(args)
wandb_logger.watch(model, log="all")

# Use DNA data module
data_module = DNASequenceDataModule(args.data_file, args.batch_size, args.num_workers)

# Based on original checkpoint setup
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.output_dir, "checkpoints"),
    filename=sae_name + "-{step}-{avg_mse_loss:.2f}",
    save_top_k=3,
    monitor="train_loss",
    mode="min",
    save_last=True,
)

# Based on original trainer setup
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    accelerator="gpu",
    devices=list(range(args.num_devices)),
    strategy="auto",
    logger=wandb_logger,
    log_every_n_steps=10,
    val_check_interval=100,
    limit_val_batches=10,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
)

# Based on original training workflow
trainer.fit(model, data_module)
trainer.test(model, data_module)

# Based on original artifact logging
for checkpoint in glob.glob(os.path.join(args.output_dir, "checkpoints", "*.ckpt")):
    # Process artifact names, replace disallowed characters with allowed ones
    checkpoint_name = os.path.basename(checkpoint)
    # wandb artifact names don't allow "=" character, replace with "-"
    artifact_name = checkpoint_name.replace("=", "-")
    
    # Create artifact and add file
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(checkpoint)
    wandb.log_artifact(artifact)

wandb.finish() 