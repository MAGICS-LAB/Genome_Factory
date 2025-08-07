import multiprocessing
import polars as pl
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import numpy as np


class DNADataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"Sequence": self.sequences[idx], "Entry": f"seq_{idx}"}


def train_val_test_split(sequences, train_frac=0.9):
    """
    Split the sequences into training, validation, and test sets.
    
    Args:
        sequences: List of DNA sequences
        train_frac: The fraction of examples to use for training
        
    Returns:
        A tuple containing the training, validation, and test sets
    """
    n_total = len(sequences)
    indices = np.random.permutation(n_total)
    
    n_train = int(train_frac * n_total)
    n_val = int((n_total - n_train) * 0.5)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_seqs = [sequences[i] for i in train_indices]
    val_seqs = [sequences[i] for i in val_indices]
    test_seqs = [sequences[i] for i in test_indices]
    
    return train_seqs, val_seqs, test_seqs


class DNASequenceDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, num_workers=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else multiprocessing.cpu_count() - 1

    def setup(self, stage=None):
        # Read DNA sequences from text file
        with open(self.data_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        
        # Filter out empty sequences and sequences with invalid characters
        valid_sequences = []
        valid_chars = set('ACGT')
        for seq in sequences:
            if seq and all(c in valid_chars for c in seq.upper()):
                valid_sequences.append(seq.upper())
        
        print(f"Loaded {len(valid_sequences)} valid DNA sequences")
        
        self.train_data, self.val_data, self.test_data = train_val_test_split(valid_sequences)
        
        print(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            DNADataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            DNADataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            DNADataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        ) 