import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder.encoder import TransformerEncoder

class Tran_S(nn.Module):
    def __init__(self, embedding_dim=768*2, output_dim=768, input_len=125, feat_dim=128):
        super(Tran_S, self).__init__()
        self.model = TransformerEncoder(d_model=embedding_dim, d_model_in=1024, d_ff=128, d_out=output_dim, n_heads=4, n_layers=1)
        self.emb_size = output_dim
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
        
    def forward(self, inputs, labels=None):        
        outputs = self.model(inputs)    
        outputs = self.contrast_head(outputs)
     
        return outputs
        
    def save_pretrained(self, dir):
        pass