import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, embedding_dim = 768*2, output_dim = 768, input_len = 125):
        super(CNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.input_len = input_len
        
        if self.input_len == 1:
            self.cnn_model = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=256, kernel_size=8, bias=True),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, bias=True),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(2),

                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=8, bias=True),
                nn.BatchNorm1d(32),
                nn.MaxPool1d(2),

                nn.Flatten()
            )
        else:
            self.cnn_model = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=4, bias=True),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=4, bias=True),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(2),

                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4, bias=True),
                nn.BatchNorm1d(32),
                nn.MaxPool1d(2),

                nn.Flatten()
            )
        self.dense_model = nn.Linear(self.count_flatten_size(), output_dim)

    def count_flatten_size(self):
        if self.input_len == 1:
            x = torch.zeros([1, 1, self.embedding_dim], dtype=torch.float)
        else:
            x = torch.zeros([1, self.input_len, self.embedding_dim], dtype=torch.float)
            x = x.transpose(1, 2)
        x = self.cnn_model(x)
        return x.shape[1]

    def forward(self, x):
        if self.input_len == 1:
            x = x.unsqueeze(1)
        else:
            x = x.transpose(1, 2)
        x = self.cnn_model(x)
        x = self.dense_model(x)
        return x

class CNN_S(nn.Module):
    def __init__(self, embedding_dim=768, output_dim=768, input_len=125, feat_dim=128):
        super(CNN_S, self).__init__()
        self.model = CNN(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
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