# genomeAI/train.py
from genomeAI.Train.metric.metric_classification import (
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
from torch.utils.data import Dataset
from genomeAI.Train.workflow.adapter.adapter_model.Adapter import Adapter
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
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
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
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model_name_or_path: str
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        if "hyenadna" in self.model_name_or_path:
            return dict(
                input_ids=input_ids,
                labels=labels,
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
        # Ensure pretrained model returns hidden states
        #if "hyenadna" or "caduceus" in self.model_name_or_path:
        try:
            hidden_size = self.pretrained_model.config.d_model
        except:
            self.pretrained_model.config.output_hidden_states = True
            hidden_size = self.pretrained_model.config.hidden_size
        #hidden_size = self.pretrained_model.get_input_embeddings().embedding_dim
        # Select adapter module based on adapter_type
      
        self.adapter = Adapter(input_dim=hidden_size, num_labels=num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None,token_type_ids=None):
        # Get hidden states from the pretrained model
        '''
        if "hyenadna" in self.model_name_or_path:
            outputs = self.pretrained_model(input_ids=input_ids)
        else:

            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
            '''
        try:
            outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        except:
            outputs = self.pretrained_model(input_ids=input_ids)
        # Use last hidden state (all tokens)
        hidden_states = outputs[0]     #.hidden_states[-1] # [batch, seq_len, hidden_size]
        # Pass through the selected adapter to get logits
        logits = self.adapter(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits)





def train_sft():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
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
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model_name_or_path=model_args.model_name_or_path)

    
    # load model
    if model_args.use_flash_attention:
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    else:
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
    model = AdapterModel(pretrained_model=model, num_labels=train_dataset.num_labels)

    # define trainer
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
    if model_args.use_lora==False:
        if custom_dir:
            trainer.save_model(custom_dir)
            model.pretrained_model.config.save_pretrained(custom_dir)
            tokenizer.save_pretrained(custom_dir)
        else:
            trainer.save_model("./Trained_model")
            model.pretrained_model.config.save_pretrained("./Trained_model")
            tokenizer.save_pretrained("./Trained_model")
        #trainer.save_model("./Trained_model")
    if model_args.use_lora==True:
        model = model.merge_and_unload()
        if custom_dir:
            model.save_pretrained(custom_dir)
            tokenizer.save_pretrained(custom_dir)
        else:
            model.save_pretrained("./lora_Trained_model")
            tokenizer.save_pretrained("./lora_Trained_model")
        #model.save_pretrained("./lora_Trained_model")
        #tokenizer.save_pretrained("./lora_Trained_model")



    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)
