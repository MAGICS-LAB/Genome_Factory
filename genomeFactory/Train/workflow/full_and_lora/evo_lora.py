from evo import Evo
import torch
import os
import types
from torch import nn
import transformers
from transformers import TrainerCallback

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

class CustomEmbedding(nn.Module):
    def unembed(self, u):
        return u

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv,dense,gated_layers,wo,classifier", metadata={"help": "where to perform LoRA"})
    

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})

# ---------------------------------------------------------------------------
# Extra training arguments that are NOT part of HuggingFace TrainingArguments.
# Keep anything new here to avoid subclassing TrainingArguments (which breaks
# JSON serialization inside the Trainer callbacks).
# ---------------------------------------------------------------------------

@dataclass
class ExtraArguments:
    # Extra flags not present in transformers.TrainingArguments
    cache_dir: Optional[str] = None
    model_max_length: int = 512
    find_unused_parameters: bool = False
    eval_and_save_results: bool = True
    save_model: bool = False

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 ):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        
        # First pass: tokenize all texts and find max length
        tokenized_texts = []
        for text in texts:
            input_ids = tokenizer.tokenize(text)
            tokenized_texts.append(input_ids)
        
        max_length = max(len(ids) for ids in tokenized_texts)
        
        # Second pass: pad to max length with LEFT padding
        output = []
        for input_ids in tokenized_texts:
            # Left pad if too short (pad_id = 1)
            if len(input_ids) < max_length:
                input_ids = [1] * (max_length - len(input_ids)) + input_ids
            
            input_ids = torch.tensor(input_ids, dtype=torch.int)
            output.append(input_ids)

        self.input_ids = output
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.stack(input_ids)
        labels = torch.Tensor(labels).long()

        return dict(
            input_ids=input_ids,
            labels=labels,
        )

class AdapterModel(nn.Module):
    """
    Adapter model wrapper.
    Freeze the pretrained model and add last-token pooling + classification layer on top.
    """
    def __init__(self, pretrained_model, num_labels: int, hidden_size: int = 4096):
        super(AdapterModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.hidden_size = hidden_size
        self.pad_token_id = 1
        
        # Direct classification layer (4096 -> num_labels)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Add name_or_path attribute that PEFT might expect
        self.name_or_path = getattr(pretrained_model, 'name_or_path', 'custom_adapter_model')
        
        # Create config that mimics transformers.PretrainedConfig
        # Check if pretrained_model has a valid transformers-compatible config
        has_valid_config = (
            hasattr(pretrained_model, 'config') and 
            pretrained_model.config is not None and
            hasattr(pretrained_model.config, 'to_dict') and
            callable(pretrained_model.config.to_dict)
        )
        
        if has_valid_config:
            self.config = pretrained_model.config
        else:
            # Import and create a real transformers config
            from transformers import GPT2Config
            
            # Create a real GPT2Config object which PEFT definitely supports
            self.config = GPT2Config(
                vocab_size=1000,
                n_positions=512, 
                n_embd=hidden_size,
                n_layer=12,
                n_head=8,
                num_labels=num_labels,
                # Add any other attributes that might be needed
            )
            # Override model_type to ensure it's recognized
            self.config.model_type = "gpt2"

    def get_input_embeddings(self):
        """Return input embeddings for PEFT compatibility"""
        # Try to get embeddings from pretrained model first
        if hasattr(self.pretrained_model, 'get_input_embeddings'):
            return self.pretrained_model.get_input_embeddings()
        elif hasattr(self.pretrained_model, 'embeddings'):
            return self.pretrained_model.embeddings
        elif hasattr(self.pretrained_model, 'embed_tokens'):
            return self.pretrained_model.embed_tokens
        else:
            # Create a dummy embedding layer if none found
            dummy_embedding = nn.Embedding(self.config.vocab_size, self.hidden_size)
            return dummy_embedding

    def get_output_embeddings(self):
        """Return output embeddings for PEFT compatibility"""
        # Try to get output embeddings from pretrained model first
        if hasattr(self.pretrained_model, 'get_output_embeddings'):
            return self.pretrained_model.get_output_embeddings()
        elif hasattr(self.pretrained_model, 'lm_head'):
            return self.pretrained_model.lm_head
        else:
            # Return None if no output embeddings (this is acceptable for some models)
            return None

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Required method for PEFT compatibility"""
        if hasattr(self.pretrained_model, 'prepare_inputs_for_generation'):
            return self.pretrained_model.prepare_inputs_for_generation(*args, **kwargs)
        else:
            return {}

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        # Get hidden states from the pretrained model
        embed, _ = self.pretrained_model(input_ids)
        embed = embed.float()  # [batch, seq_len, hidden_size]
        
        # Last token pooling. With left-padding, the last token of the sequence
        # is assumed to be the last non-padding token.
        pooled_embed = embed[:, -1, :]  # [batch, hidden_size]
        
        # Pass through classification layer
        logits = self.classifier(pooled_embed)  # [batch, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Return in standard format like original
        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

def find_all_linear_names(model: torch.nn.Module):
    """
    Find the names of all torch.nn.Linear modules in the model.
    This is useful if we want to apply LoRA to all linear layers automatically.
    """
    linear_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_module_names.append(name)
    return linear_module_names

def find_all_in_and_out_proj_names(model: torch.nn.Module):
    proj_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("in_proj" in name or "out_proj" in name or "score" in name):
            proj_names.append(name)
    return proj_names

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)

"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

def train():
    # Parse standard HF arguments plus our extra ones
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments, ExtraArguments))
    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()

    # Load Evo model and tokenizer
    evo_model = Evo(model_args.model_name_or_path)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.unembed = CustomEmbedding()

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"))
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"))
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"))
    data_collator = DataCollatorForSupervisedDataset()

    # Create adapter model
    model = AdapterModel(pretrained_model=model, num_labels=train_dataset.num_labels)
    print(model)
    
    # configure LoRA
    if model_args.use_lora:
        # Add debugging information
        print(f"Model type: {type(model)}")
        print(f"Model config: {model.config}")
        print(f"Model config type: {type(model.config)}")
        print(f"Model has config: {hasattr(model, 'config')}")
        
        # Ensure config has all required methods
        if hasattr(model.config, 'to_dict') and callable(model.config.to_dict):
            print("Config to_dict method exists and is callable")
        else:
            print("Config to_dict method missing or not callable")
            
        if model_args.lora_target_modules.strip().lower() == "all":
            target_modules = find_all_linear_names(model)
            print(f"LoRA target_modules = all => found {len(target_modules)} linear layers: {target_modules}")
        elif model_args.lora_target_modules.strip().lower() == "all_in_and_out_proj":
            target_modules = find_all_in_and_out_proj_names(model)
            print(f"LoRA target_modules = all_in_and_out_proj => found {len(target_modules)} in_proj and out_proj layers: {target_modules}")
        else:
            target_modules = list(model_args.lora_target_modules.split(","))
            print(f"LoRA target_modules = {target_modules}")
            
        if "hyenadna" in model_args.model_name_or_path.lower():
            orig_forward = model.forward  

            def forward_silent(self,
                            input_ids=None,
                            labels=None,
                            **_):         
                return orig_forward(input_ids=input_ids, labels=labels)

            model.forward = types.MethodType(forward_silent, model)

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        
        print("About to call get_peft_model...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    output = trainer.train()
    print(output.metrics)

    if extra_args.save_model:
        trainer.save_state()
        trainer.save_model(training_args.output_dir)

    # get the evaluation results from trainer
    if extra_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    train()