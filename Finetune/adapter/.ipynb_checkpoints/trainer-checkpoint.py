import transformers
import torch
from torch import Tensor
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class GTDBTrainer(transformers.Trainer):
    def __init__(self, train_label_path=None, val_label_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.train_label_matrix = pd.read_csv(train_label_path, index_col=None, header=None)
        self.val_label_matrix = pd.read_csv(val_label_path, index_col=None, header=None)
        self.loss = MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs['ids']
        embeddings = inputs['embeddings']
        
        if self.args.fp16 or self.args.bf16:
            with torch.cuda.amp.autocast():
                features = model(embeddings)
        else:
            features = model(embeddings)
            
        used_label_matrix = self.train_label_matrix.iloc[np.array(labels.cpu()), np.array(labels.cpu())]
        used_label_matrix = torch.from_numpy(np.array(used_label_matrix)).float().to(features.device)
                
        if labels is not None:
            if self.args.loss_function == 'mse':
                loss, outputs = self.loss(features, used_label_matrix)
            else:
                raise ValueError(f"Invalid loss function: {self.args.loss_function}")
            
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):

        labels = inputs['ids']
        embeddings = inputs['embeddings']
        
        model.eval()
        
        with torch.no_grad():
            features = model(embeddings)
            used_label_matrix = self.val_label_matrix.iloc[np.array(labels.cpu()), np.array(labels.cpu())]
            used_label_matrix = torch.from_numpy(np.array(used_label_matrix)).float().to(features.device)
            loss, outputs = self.loss(features, used_matrix)
                
        model.train()

        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, outputs, used_matrix)
    
class MSELoss():
    def __init__(
        self
    ):
        self.mse_loss = nn.MSELoss()

    def __call__(
        self,
        q_reps,
        labels
    ):
        
        if torch.distributed.is_initialized():
            full_q_reps = mismatched_sizes_all_gather(q_reps)
            full_q_reps = torch.cat(full_q_reps)
        else:
            full_q_reps = q_reps
        
        # compute loss
        batch_size = full_q_reps.shape[0]
        features = full_q_reps
        mask = torch.eye(batch_size, dtype=torch.bool).to(full_q_reps.device)
        mask = ~mask
        
        # all_sim = torch.mm(features, features.t().contiguous())*mask.float()
        # Use the difference between each sample to predict the labels
        differences = features[:, None, :] - features[None, :, :]
        distances_squared = torch.sum(differences**2, dim=2)*mask.float()
        
        loss = self.mse_loss(distances_squared, labels)
        return loss, distances_squared
    
# The following code is provided for the multu-node distributed training, to conbine the results from all nodes.
# The following is ofter not used
    
class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor, group, async_op):
        torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True) for i in range(torch.distributed.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank], None, None

all_gather_with_grad = AllGather.apply

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def mismatched_sizes_all_gather(tensor: Tensor, group=None, async_op=False, mismatched_axis=0):
    # all_gather doesn't support tensor lists where the first dimension is mismatched. This does.
    assert torch.distributed.is_initialized(), "torch.distributed not initialized"
    world_size = torch.distributed.get_world_size()
    # let's get the sizes for everyone
    mismatched_sizes = torch.tensor([tensor.shape[mismatched_axis]], dtype=torch.int64, device="cuda")
    sizes = [torch.zeros_like(mismatched_sizes) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, mismatched_sizes, group=group, async_op=async_op)
    sizes = torch.cat(sizes).cpu().tolist()
    # now pad to the max dim-0 size
    max_size = max(sizes)
    padded = torch.zeros((*tensor.shape[:mismatched_axis], max_size, *tensor.shape[mismatched_axis+1:]),
                         device=tensor.device, dtype=tensor.dtype)
    # selects the place where we're adding information
    padded_to_fill = padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])
    padded_to_fill[...] = tensor
    # gather the padded tensors
    tensor_list = [torch.zeros(padded.shape, device=padded.device, dtype=padded.dtype) for _ in range(world_size)]
    all_gather_with_grad(tensor_list, padded, group, async_op)
    # trim off the padding
    for rank in range(world_size):
        # checks that the rest is 0
        assert not tensor_list[rank].narrow(mismatched_axis, sizes[rank], padded.shape[mismatched_axis]-sizes[rank]).count_nonzero().is_nonzero(), \
            "This would remove non-padding information"
        tensor_list[rank] = tensor_list[rank].narrow(mismatched_axis, 0, sizes[rank])
    return tensor_list
