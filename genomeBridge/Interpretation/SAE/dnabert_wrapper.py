import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Set Hugging Face cache directory
os.environ['TRANSFORMERS_CACHE'] = '/projects/p32572'
os.environ['HF_HOME'] = '/projects/p32572'


class DNABERTModel(pl.LightningModule):
    def __init__(self, model_name="/projects/p32572/interprot/interprot/nonover_4_mer_BERT"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embed_dim = self.model.config.hidden_size  # 768 for DNABERT-2-117M
        
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

    def get_layer_activations(self, input, layer_idx=None):
        """
        Get the last hidden state activations for DNA sequences.
        
        Args:
            input: Either a string (single sequence) or list of strings (batch of sequences)
            layer_idx: Not used for DNABERT (we use last hidden state), kept for compatibility
            
        Returns:
            tokens: Input token IDs
            hidden_states: Last hidden state activations [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask for proper pooling [batch_size, seq_len]
        """
        if isinstance(input, str):
            sequences = [input]
        elif isinstance(input, list):
            sequences = input
        else:
            raise ValueError("Input must be a string or list of strings")
        
        # Follow correct DNABERT-2 usage
        with torch.no_grad():
            # Use correct DNABERT-2 approach: get input_ids and attention_mask
            tokenized = self.tokenizer(
                sequences, 
                return_tensors="pt", 
                padding="max_length",  # Ensure all sequences are padded to max_length
                truncation=True,
                max_length=50  # Adjust based on your sequence lengths
            )
            
            tokens = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Move to device
            tokens = tokens.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Get model outputs - DNABERT-2 returns tuple, take first element
            hidden_states = self.model(tokens, attention_mask=attention_mask)[0]  # [batch_size, seq_len, 768]
            
            return tokens, hidden_states, attention_mask

    def get_sequence(self, x, layer_idx=None):
        """
        For compatibility with ESM interface. 
        Since we're using DNABERT for representation learning, not generation,
        we can return the hidden states directly or implement a simple linear head.
        """
        # For now, just return the input as this method is used for validation
        # In a real implementation, you might want to add a language modeling head
        return x

    def forward(self, input_ids, attention_mask=None):
        """Standard forward pass"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask) 