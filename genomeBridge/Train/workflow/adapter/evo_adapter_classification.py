from evo import Evo
import torch
import os
from torch import nn
import transformers
from genomeBridge.Train.metric.metric_classification import (
    calculate_metric_with_sklearn,
    preprocess_logits_for_metrics,
    compute_metrics,
)

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
import pandas as pd
import scipy.stats as stats
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from genomeBridge.Train.workflow.adapter.adapter_model.Adapter import Adapter
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# Data cleaning parameters
MIN_LEN = 50
MAX_LEN = 1000
GC_LOW = 0.3
GC_HIGH = 0.7
AMBIG_N_THRESHOLD = 0.05
RARE_PROFILE_THRESHOLD = 0.01

def gc_content(seq):
    """Calculate GC content of a sequence"""
    if len(seq) == 0:
        return 0.0
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)

def basic_quality_filter(df):
    """Apply basic quality filters to genomic sequences"""
    def is_valid(seq):
        if not MIN_LEN <= len(seq) <= MAX_LEN:
            return False
        gc = gc_content(seq)
        if not GC_LOW <= gc <= GC_HIGH:
            return False
        if seq.count('N') / len(seq) > AMBIG_N_THRESHOLD:
            return False
        return True
    
    filtered = df[df['sequence'].apply(is_valid)].copy()
    filtered['gc_content'] = filtered['sequence'].apply(gc_content)
    filtered['length'] = filtered['sequence'].apply(len)
    return filtered

def advanced_statistical_qc(df):
    """Apply advanced statistical quality control"""
    # (a) Call Rate analog: exclude sequences with >AMBIG_N_THRESHOLD ambiguous bases
    df = df[df['sequence'].apply(lambda s: s.count('N') / len(s) <= AMBIG_N_THRESHOLD)]

    # (b) HWE analog: chi-square test for A/T/C/G distribution vs. expected uniform
    def hwe_test(seq):
        counts = {b: seq.count(b) for b in 'ATCG'}
        total = sum(counts.values())
        if total == 0:
            return False
        expected = [total / 4] * 4
        observed = [counts[b] for b in 'ATCG']
        try:
            _, p = stats.chisquare(f_obs=observed, f_exp=expected)
            return p > 0.001  # reject if too extreme
        except:
            return False
    
    df = df[df['sequence'].apply(hwe_test)]

    # (c) Rare compositional profiles
    df['profile'] = df['sequence'].apply(lambda s: (s.count('A'), s.count('T'), s.count('C'), s.count('G')))
    profile_counts = df['profile'].value_counts(normalize=True)
    common_profiles = profile_counts[profile_counts > RARE_PROFILE_THRESHOLD].index
    df = df[df['profile'].isin(common_profiles)]
    return df.drop(columns=['profile'])

def final_validation(df):
    """Perform final validation with pseudo case-control analysis"""
    if len(df) == 0:
        return df, {}
        
    # Pseudo case-control
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['group'] = ['case' if i % 2 == 0 else 'control' for i in range(len(df))]

    results = {}
    for metric in ['gc_content', 'length']:
        case_vals = df[df['group'] == 'case'][metric]
        ctrl_vals = df[df['group'] == 'control'][metric]
        if len(case_vals) > 0 and len(ctrl_vals) > 0:
            try:
                t_stat, p_val = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
                results[metric] = {
                    'case_mean': case_vals.mean(),
                    'control_mean': ctrl_vals.mean(),
                    'p_value': p_val
                }
            except:
                results[metric] = {
                    'case_mean': case_vals.mean() if len(case_vals) > 0 else 0,
                    'control_mean': ctrl_vals.mean() if len(ctrl_vals) > 0 else 0,
                    'p_value': 1.0
                }

    return df.drop(columns=['group']), results

def run_data_cleaning_pipeline(csv_path):
    """Run the complete data cleaning pipeline"""
    df = pd.read_csv(csv_path)
    print(f"Original dataset size for {csv_path}: {len(df)}")

    # Skip cleaning if no 'sequence' column (for non-genomic data)
    if 'sequence' not in df.columns:
        print("No 'sequence' column found, skipping genomic data cleaning")
        return df

    df = basic_quality_filter(df)
    print(f"After basic filtering: {len(df)}")

    df = advanced_statistical_qc(df)
    print(f"After advanced statistical QC: {len(df)}")

    df_validated, validation_results = final_validation(df)
    print("Final validation results:")
    for k, v in validation_results.items():
        print(f"{k}: case={v['case_mean']:.4f}, control={v['control_mean']:.4f}, p={v['p_value']:.4g}")
    
    return df_validated

class CustomEmbedding(nn.Module):
  def unembed(self, u):
    return u


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv,dense,gated_layers,wo,classifier", metadata={"help": "where to perform LoRA"})
    use_flash_attention: bool = field(default=True, metadata={"help": "whether to use flash attention"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    remove_unused_columns: bool = field(default=False)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    saved_model_dir: str = field(
        default="",
        metadata={"help": "If non-empty, final model will be saved at this path."}
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 ):

        super(SupervisedDataset, self).__init__()

        # Apply data cleaning pipeline first
        try:
            # Try to read as pandas DataFrame first for cleaning
            df_cleaned = run_data_cleaning_pipeline(data_path)
            
            # Convert cleaned DataFrame back to expected format
            if 'sequence' in df_cleaned.columns and 'label' in df_cleaned.columns:
                # Genomic data format: sequence, label
                data = [[row['sequence'], int(row['label'])] for _, row in df_cleaned.iterrows()]
            elif len(df_cleaned.columns) == 2:
                # Generic 2-column format
                data = df_cleaned.values.tolist()
            elif len(df_cleaned.columns) >= 3:
                # 3+ column format, take first 3 columns
                data = df_cleaned.iloc[:, :3].values.tolist()
            else:
                raise ValueError("Cleaned data format not supported.")
                
        except Exception as e:
            # Fallback to original CSV reading method if cleaning fails
            logging.warning(f"Data cleaning failed: {e}. Falling back to original method.")
            with open(data_path, "r") as f:
                data = list(csv.reader(f))[1:]  # Skip header
        
        # Process the data based on format
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
        
        output = []
        for text in texts:
            input_ids = torch.tensor(tokenizer.tokenize(text),dtype=torch.int)
            output.append(input_ids)

        self.input_ids = output
        #self.attention_mask = output["attention_mask"]
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
    Freeze the pretrained model and add an adapter module on top.
    The adapter's input is the pretrained model's last hidden states,
    and its output is num_labels.
    """
    def __init__(self, pretrained_model, num_labels: int):
        super(AdapterModel, self).__init__()
        self.pretrained_model = pretrained_model
        # Freeze pretrained model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.adapter = Adapter(input_dim=4096, num_labels=num_labels)

    def forward(self, input_ids,labels=None):
        # Get hidden states from the pretrained model
        embed, _ = self.pretrained_model(input_ids)
        embed = embed.float()

        logits = self.adapter(embed)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits)





def train_sft():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    evo_model = Evo(model_args.model_name_or_path)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.unembed = CustomEmbedding()
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      )
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     )
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     )
    data_collator = DataCollatorForSupervisedDataset()
    model = AdapterModel(pretrained_model=model, num_labels=train_dataset.num_labels)

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   #tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    
    output = trainer.train()
    print(output.metrics)
    
    custom_dir = training_args.saved_model_dir.strip()
    if model_args.use_lora==False:
        if custom_dir:
            trainer.save_model(custom_dir)
            
        else:
            trainer.save_model("./Trained_model")



    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)