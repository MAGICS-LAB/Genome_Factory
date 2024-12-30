import os 
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import csv
import json
import random
import argparse
import pysam
import transformers
import sklearn
import logging
import numpy as np
import torch
import torch.nn as nn
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from dataloader.dataset import GTDBDataset
from trainer import GTDBTrainer
from models import CNN_S, MLP_S, LSTM_S, Tran_S 

transformers.logging.get_logger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=10240, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="no"),
    warmup_steps: int = field(default=0)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    save_safetensors: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=False)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    loss_function: str = field(default="mse")
    max_grad_norm: float = field(default=1.0)
    dataloader_num_workers: int = field(default=1)
    pj_name: str = field(default="Genome_Ocean")
    log_name: str = field(default="CNN")
    
    # data args
    data_dir: str = field(default=None, metadata={"help": "Path to the input dataset."}) 
    label_dir: str = field(default=None, metadata={"help": "Path to the label."}) 
    
    # model args
    model_type: Optional[str] = field(default="CNN", metadata={"help": "Which adapter to use."}) 
    feat_dim: int = field(default=3072, metadata={"help": "Input feature dimension of adapter."}) 
    output_dim: int = field(default=768, metadata={"help": "Output feature dimension of adapter."}) 
    resolution: int = field(default=10240, metadata={"help": "Resolution used in extracting the embedding of foundation model."}) 


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def run():
    logger = transformers.logging.get_logger(__name__)
    parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]
    set_global_random_seed(training_args.seed)
    
    wandb.init(project=training_args.pj_name, name=training_args.log_name)

    # define datasets and data collator
    logger.info("Loading datasets and data collator")
    train_dataset = GTDBDataset(training_args.data_dir+'/train') 
    val_dataset = GTDBDataset(training_args.data_dir+'/val')
    
    # load model
    embedding_dim=training_args.feat_dim
    output_dim=training_args.output_dim
    input_len=int(training_args.model_max_length/training_args.resolution)
    if training_args.model_type == "CNN":
      model = CNN_S(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
    elif training_args.model_type == "MLP":
      model = MLP_S(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
    elif training_args.model_type == "LSTM":
      if input_len == 1:
        raise ValueError("Sequence length cannot be 1 for LSTM. Please set a different sequence length.")
      model = LSTM_S(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
    elif training_args.model_type == "Tran":
      if input_len == 1:
        raise ValueError("Sequence length cannot be 1 for Transformer. Please set a different sequence length.")
      model = Tran_S(embedding_dim=embedding_dim, output_dim=output_dim, input_len=input_len)
    
    
    optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=training_args.learning_rate, 
                    weight_decay=training_args.weight_decay
                    )
    trainer = GTDBTrainer(
                    model=model,
                    tokenizer=None,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    optimizers=(optimizer, None),
                    train_label_path=training_args.label_dir+'/train/label.csv', 
                    val_label_path=training_args.label_dir+'/val/label.csv',
                    )
    trainer.evaluate()
    trainer.train()

if __name__ == '__main__':
    run()