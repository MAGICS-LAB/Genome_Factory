import torch
import pysam
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass


# @dataclass
# class DataCollatorForCFDNA(object):
#     def __init__(self):
#         pass
            
#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         instances = [instance for instance in instances if instance is not None]

#         means = torch.stack([instance["mean"] for instance in instances])
#         stds = torch.stack([instance["std"] for instance in instances])
        
#         labels = torch.tensor([instance["label"] for instance in instances], dtype=torch.float)
        
#         return dict(
#             inputs=(means, stds),
#             labels=labels,
#             )


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