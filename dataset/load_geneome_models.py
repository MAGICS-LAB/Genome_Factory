import torch
from transformers import AutoTokenizer, AutoModel

class LoadGenomeModels:
    def __init__(self, model_name="DNABERT", cache_dir=None):
        """
        Initialize the model loader with the specified model name.
        :param model_name: Name of the model to load, e.g., "DNABERT"
        :param cache_dir: Directory to cache the model files
        :param download: Whether to download the model if not present
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load the model and tokenizer based on the specified model name.
        """
        if self.model_name == "DNABERT":
            model_path = "zhihan1996/DNABERT-2-117M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
        else:
            raise ValueError(f"Model name '{self.model_name}' is not recognized.")
        
        return self.tokenizer, self.model

    def encode(self, dna_sequence):
        """
        Encode the DNA sequence into input IDs using the tokenizer.
        :param dna_sequence: A string of DNA sequence
        :return: Tensor of input IDs for the model
        """
        return self.tokenizer(dna_sequence, return_tensors="pt")["input_ids"]

    def predict_mean(self, dna_sequence):
        """
        Pass the DNA sequence through the model and compute mean pooled embedding.
        :param dna_sequence: A string of DNA sequence
        :return: Mean pooled embedding
        """
        inputs = self.encode(dna_sequence)
        with torch.no_grad():
            hidden_states = self.model(inputs)[0]  # Shape: [1, sequence_length, hidden_size]
        embedding_mean = torch.mean(hidden_states[0], dim=0)  # Shape: [hidden_size]
        return embedding_mean

    def predict_max(self, dna_sequence):
        """
        Pass the DNA sequence through the model and compute max pooled embedding.
        :param dna_sequence: A string of DNA sequence
        :return: Max pooled embedding
        """
        inputs = self.encode(dna_sequence)
        with torch.no_grad():
            hidden_states = self.model(inputs)[0]  # Shape: [1, sequence_length, hidden_size]
        embedding_max = torch.max(hidden_states[0], dim=0)[0]  # Shape: [hidden_size]
        return embedding_max

# Usage example
if __name__ == "__main__":
    # Instantiate the model loader
    genome_model = LoadGenomeModels(model_name="DNABERT")
    
    # DNA sequence to test
    dna_sequence = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
    
    # Get mean and max pooled embeddings separately
    embedding_mean = genome_model.predict_mean(dna_sequence)
    embedding_max = genome_model.predict_max(dna_sequence)
    
    # Print shapes to verify
    print("Mean Pooling Embedding Shape:", embedding_mean.shape)  # Expected: torch.Size([768])
    print("Max Pooling Embedding Shape:", embedding_max.shape)    # Expected: torch.Size([768])
