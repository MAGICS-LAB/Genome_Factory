import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    """
    adapter module.
    Performs mean pooling over the sequence of embeddings and passes through a two-layer MLP,
    mapping the result to num_labels.
    """
    def __init__(self, input_dim: int, num_labels: int):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x = torch.mean(x, dim=1)  # Mean pooling: [batch, input_dim]
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)  # [batch, num_labels]
        return logits