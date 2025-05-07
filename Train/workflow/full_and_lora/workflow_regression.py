from genomeBridge.Train.metric.metric_regression import (
    calculate_metric_with_sklearn,
    preprocess_logits_for_metrics,
    compute_metrics,
)

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, List
import os
import csv
import json
import logging
import types
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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
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
    """Dataset for supervised fine-tuning (regression)."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 ):

        super(SupervisedDataset, self).__init__()

        # Load data from the CSV file
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # Data format: [text, continuous label]
            logging.warning("Perform single sequence regression...")
            texts = [d[0] for d in data]
            labels = [float(d[1]) for d in data]
        elif len(data[0]) == 3:
            # Data format: [text1, text2, continuous label]
            logging.warning("Perform sequence-pair regression...")
            texts = [[d[0], d[1]] for d in data]
            labels = [float(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        #self.attention_mask = output["attention_mask"]
        self.labels = labels
        # For regression, the number of labels is fixed to 1
        self.num_labels = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning (regression)."""

    tokenizer: transformers.PreTrainedTokenizer
    model_name_or_path: str
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels, dtype=torch.float)
        if "hyenadna" in self.model_name_or_path:
            return dict(
                input_ids=input_ids,
                labels=labels,
                #attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
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

def train_sft():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # Define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      )
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     )
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "test.csv"), 
                                      )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model_name_or_path=model_args.model_name_or_path)

    if model_args.use_flash_attention:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )
    # Set problem type to regression for MSE loss
    model.config.problem_type = "regression"

    # Configure LoRA if enabled
    if model_args.use_lora:
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
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    
    output = trainer.train()
    print(output.metrics)
    custom_dir = training_args.saved_model_dir.strip()
    if not model_args.use_lora:
        if custom_dir:
            trainer.save_model(custom_dir)
        else:
            trainer.save_model("./Trained_model")
    else:
        model = model.merge_and_unload()
        if custom_dir:
            model.save_pretrained(custom_dir)
            tokenizer.save_pretrained(custom_dir)
        else:
            model.save_pretrained("./lora_Trained_model")
            tokenizer.save_pretrained("./lora_Trained_model")

    # Get evaluation results and save
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)
