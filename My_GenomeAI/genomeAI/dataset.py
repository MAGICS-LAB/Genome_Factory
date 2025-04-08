import csv
import json
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Sequence

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

def get_alter_of_dna_sequence(sequence: str) -> str:
    """Get the complement of a DNA sequence (A<->T, C<->G)."""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # Note: Reversed complement can be obtained by reversing the sequence if needed.
    return "".join([MAP.get(c, c) for c in sequence])

def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from a DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """
    Load or generate k-mer strings for each DNA sequence in texts.
    The generated k-mer strings will be saved to a file with suffix '_{k}mer.json'.
    """
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer_list = json.load(f)
    else:
        logging.warning("Generating k-mer strings...")
        kmer_list = [generate_kmer_str(text, k) for text in texts]
        # Save the generated k-mer list for future use
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer_list, f)
    return kmer_list

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (supports single or pair sequence classification)."""
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, kmer: int = -1):
        super().__init__()
        # Load data from CSV (skip header)
        with open(data_path, "r") as f:
            data = list(csv.reader(f))
            # If the first row is header, skip it
            if len(data) > 0 and not data[0]:
                data = data[1:]
            else:
                data = data[1:]
        # Determine format by number of columns
        if len(data[0]) == 2:
            # Format: [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [row[0] for row in data]
            labels = [int(row[1]) for row in data]
        elif len(data[0]) == 3:
            # Format: [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[row[0], row[1]] for row in data]
            labels = [int(row[2]) for row in data]
        else:
            raise ValueError("Data format not supported. Each row should have 2 or 3 columns.")
        # If k-mer is requested, generate or load k-mer transformed sequences
        if kmer != -1:
            # Only generate on one process (for distributed training)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() not in [0, -1]:
                    torch.distributed.barrier()
            logging.warning(f"Using {kmer}-mer as input... converting sequences.")
            texts = load_or_generate_kmer(data_path, texts, kmer)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    torch.distributed.barrier()
        # Tokenize the texts
        encodings = tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]
        # Modify pad tokens (id 0) in the first sample to a different token (id 2)
        self.input_ids[0][self.input_ids[0] == 0] = 2  # Change which tokens are attended vs not
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

@dataclass
class DataArguments:
    """
    Data-related arguments.
    """
    data_path: str  # Path to directory containing train.csv, dev.csv, test.csv
    kmer: int = -1  # k-mer size for input sequences (-1 for no k-mer transformation)

class DataCollatorForSupervisedDataset:
    """Collate function for supervised fine-tuning dataset."""
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Separate input_ids and labels from instances
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # Pad input_ids to the longest sequence in the batch
        input_ids_tensor = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": input_ids_tensor.ne(self.tokenizer.pad_token_id),
            "labels": labels_tensor
        }
