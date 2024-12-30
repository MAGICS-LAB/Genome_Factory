import torch
import os
import gdown
import zipfile
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoConfig, AutoModelForCausalLM,PreTrainedTokenizer,PreTrainedModel
from genslm import GenSLM, SequenceDataset
import numpy as np
from torch.utils.data import DataLoader
class LoadGenomeModels:
    def __init__(self, model_name: str, cache_dir: str = None):
        """
        Initialize the model loader with the specified model name.
        :param model_name: Name of the model to load, e.g., "DNABERT-2", "hyenadna", "nucleotide-transformer", "evo-1", or "caduceus"
        :param cache_dir: Directory to cache the model files
        """
        self.model_name: str = model_name
        self.cache_dir: str = cache_dir
        self.tokenizer: PreTrainedTokenizer = None
        self.model: PreTrainedModel = None
        self.load_model()


    def load_model(self) -> None:
        """
        Load the model and tokenizer based on the specified model name.
        """
        
        if self.model_name == "DNABERT-2":
            model_path = "zhihan1996/DNABERT-2-117M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
        
        elif self.model_name == "hyenadna":
            model_path = "LongSafari/hyenadna-medium-160k-seqlen-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path,cache_dir=self.cache_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        
        elif self.model_name == "nucleotide-transformer":
            model_path = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
        
        elif self.model_name == "evo-1":
            model_path = "togethercomputer/evo-1-131k-base"
            config = AutoConfig.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True, revision="1.1_fix")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, config=config, cache_dir=self.cache_dir, trust_remote_code=True, revision="1.1_fix")
        
        elif self.model_name == "caduceus":
            model_path = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path, cache_dir=self.cache_dir, trust_remote_code=True)
        
        elif self.model_name=="genslm":
            download_url = "https://drive.google.com/uc?id=1pgD5hqlV62JqmVPTEsL1pkmeXEnzpTKZ&export=download"

            zip_filename = "model.zip"
            gdown.download(download_url, zip_filename, quiet=False)
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall('.')

            os.remove(zip_filename)
            model_temp = GenSLM("genslm_25M_patric", model_cache_dir="./")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_temp.to(device)
            self.model=model_temp.model
            self.tokenizer=model_temp.tokenizer
        
        else:
            raise ValueError(f"Model name '{self.model_name}' is not recognized.")

    def get_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Return the loaded model and tokenizer.
        :return: Tuple of (model, tokenizer)
        """
        return self.model, self.tokenizer

# Usage example
if __name__ == "__main__":
    
    # Instantiate the model loader for DNABERT-2
    genome_model_dnabert = LoadGenomeModels(model_name="DNABERT-2")
    model, tokenizer = genome_model_dnabert.get_model_and_tokenizer()
    print("Model and Tokenizer for DNABERT-2 loaded:", model, tokenizer)
    
    # Instantiate the model loader for hyenadna
    genome_model_hyenadna = LoadGenomeModels(model_name="hyenadna")
    model, tokenizer = genome_model_hyenadna.get_model_and_tokenizer()
    print("Model and Tokenizer for hyenadna loaded:", model, tokenizer)
    
    # Instantiate the model loader for nucleotide-transformer
    genome_model_nt_transformer = LoadGenomeModels(model_name="nucleotide-transformer")
    model, tokenizer = genome_model_nt_transformer.get_model_and_tokenizer()
    print("Model and Tokenizer for nucleotide-transformer loaded:", model, tokenizer)
    
    # Instantiate the model loader for evo-1
    genome_model_evo = LoadGenomeModels(model_name="evo-1")
    model, tokenizer = genome_model_evo.get_model_and_tokenizer()
    print("Model and Tokenizer for evo-1 loaded:", model, tokenizer)
    
    # Instantiate the model loader for caduceus
    genome_model_caduceus = LoadGenomeModels(model_name="caduceus")
    model, tokenizer = genome_model_caduceus.get_model_and_tokenizer()
    print("Model and Tokenizer for caduceus loaded:", model, tokenizer)

    genome_model_genslm = LoadGenomeModels(model_name="genslm")
    model, tokenizer = genome_model_genslm.get_model_and_tokenizer()
    print("Model and Tokenizer for genslm loaded:", model, tokenizer)
    
    