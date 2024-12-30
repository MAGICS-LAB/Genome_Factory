import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel

class MLP(nn.Module):
    def __init__(self, embedding_dim = 768, output_dim = 768, input_len = 125):
        super(MLP, self).__init__()
        
        self.input_len = input_len
        
        if self.input_len == 1:
            self.layer1 = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU()
            )
            
            self.layer2 = nn.Linear(1024, output_dim)
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(input_len, input_len*4),
                nn.ReLU(),
                nn.Linear(input_len*4, input_len),
                nn.ReLU(),
                nn.Linear(input_len, 4),
                nn.ReLU(),
                nn.Flatten()
            )
            
            self.layer2 = nn.Linear(embedding_dim*4, output_dim)
    
    def forward(self, x):
        if self.input_len > 1:
            x = x.transpose(1, 2)
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class MLP_S(nn.Module):
    def __init__(self, embedding_dim=3072, output_dim=768, input_len=10240, feat_dim=128):
        super(MLP_S, self).__init__()
        self.model = MLP(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
        self.emb_size = output_dim
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
        
    def forward(self, inputs, labels=None):        
        outputs = self.model.forward(inputs)    
        outputs = self.contrast_head(outputs)
     
        return outputs

    def save_pretrained(self, dir):
        pass