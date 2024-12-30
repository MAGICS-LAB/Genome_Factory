import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel


class LSTMNet(nn.Module):
    def __init__(self, embedding_dim = 768*2, output_dim = 768, input_len = 125):
        super(LSTMNet, self).__init__()
        self.hidden_size = int(embedding_dim/4)
        self.num_layers = 1
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_dim)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(torch.mean(out, dim=1))
        return out

class LSTM_S(nn.Module):
    def __init__(self, embedding_dim=768*2, output_dim=768, input_len=125, feat_dim=128):
        super(LSTM_S, self).__init__()
        self.model = LSTMNet(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
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