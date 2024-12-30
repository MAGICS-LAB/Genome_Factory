import torch
import pysam
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass


class GTDBDataset(Dataset):
    def __init__(
                self, 
                data_dir,
                ids=None
            ):
        self.embeddings = np.load(data_dir+"/embedding.npy")
        if ids != None:
            self.embeddings = [self.embeddings[i] for i in ids]
        
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        inputs = self.embeddings[idx]
        
        return {
            'ids': idx,
            'embeddings': torch.from_numpy(inputs)
        }