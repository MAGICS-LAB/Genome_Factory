# genomeAI/train.py
import os
import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, List
import torch
import torch.nn as nn
import transformers
import numpy as np
import sklearn.metrics
from torch.utils.data import Dataset
from adapter_model.Adapter import Adapter
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# =========================
# Metric Functions for Regression
# =========================
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    """
    Calculate the MSE and MAE with sklearn.
    Exclude any labels set to -100 (used for padding).
    """
    valid_mask = labels != -100  # Exclude padding tokens if any
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    mse = sklearn.metrics.mean_squared_error(valid_labels, valid_predictions)
    mae = sklearn.metrics.mean_absolute_error(valid_labels, valid_predictions)
    return {"mse": mse, "mae": mae}

def preprocess_logits_for_metrics(logits, _):
    """
    For huggingface trainer: preprocess logits for metrics calculation.
    For regression, simply squeeze the last dimension.
    """
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.squeeze(-1)

def compute_metrics(eval_pred):
    """
    Compute regression metrics using sklearn.
    """
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

# =========================
# Argument Classes
# =========================
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="Wqkv,dense,gated_layers,wo,classifier", metadata={"help": "where to perform LoRA"})
    use_fp16: bool = field(default=False, metadata={"help": "Use FP16 precision."})
    use_8bit: bool = field(default=False, metadata={"help": "Use 8-bit quantization (int8)."})
    use_4bit: bool = field(default=False, metadata={"help": "Use 4-bit quantization."})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})

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

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collect the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# =========================
# Utility Functions for DNA Sequences
# =========================
def get_alter_of_dna_sequence(sequence: str):
    """
    Get the altered DNA sequence by mapping each nucleotide.
    """
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])

def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from a DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """
    Load or generate k-mer strings for each DNA sequence.
    Save the generated k-mer strings to a JSON file.
    """
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning("Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer

# =========================
# Dataset and Data Collator
# =========================
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning for regression tasks."""
    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        # Load data from CSV
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # Data format: [text, label] for regression
            logging.warning("Perform single sequence regression...")
            texts = [d[0] for d in data]
            labels = [float(d[1]) for d in data]
        elif len(data[0]) == 3:
            # Data format: [text1, text2, label] for regression
            logging.warning("Perform sequence-pair regression...")
            texts = [[d[0], d[1]] for d in data]
            labels = [float(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            # Only generate/load k-mer on the main process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()
            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)
            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        # For regression, number of labels is 1
        self.num_labels = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning for regression."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Pad the input_ids sequences
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # Convert labels to float tensor
        labels = torch.tensor(labels, dtype=torch.float)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

# =========================
# Model Definition
# =========================
def find_all_linear_names(model: torch.nn.Module):
    """
    Find all torch.nn.Linear modules in the model.
    Useful for applying LoRA to these layers automatically.
    """
    linear_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_module_names.append(name)
    return linear_module_names

class AdapterModel(nn.Module):
    """
    Adapter model wrapper for regression.
    Freeze the pretrained model and add an adapter module.
    The adapter takes the pretrained model's last hidden states and outputs a continuous value.
    """
    def __init__(self, pretrained_model, num_labels: int):
        super(AdapterModel, self).__init__()
        self.pretrained_model = pretrained_model
        # Freeze pretrained model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        # Ensure the pretrained model returns hidden states
        self.pretrained_model.config.output_hidden_states = True
        hidden_size = self.pretrained_model.config.hidden_size
        # Initialize adapter module with output dimension equal to num_labels (1 for regression)
        self.adapter = Adapter(input_dim=hidden_size, num_labels=num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None,token_type_ids=None):
        # Get hidden states from the pretrained model
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state (shape: [batch, seq_len, hidden_size])
        hidden_states = outputs[0]
        # Get regression output from the adapter (expected shape: [batch, 1])
        logits = self.adapter(hidden_states)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # Squeeze logits to match the shape of labels
            loss = loss_fct(logits.squeeze(-1), labels.float())
        return (loss, logits)

# =========================
# Training Function
# =========================
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
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(data_args.data_path, "train.csv"), 
        kmer=data_args.kmer
    )
    val_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(data_args.data_path, "dev.csv"), 
        kmer=data_args.kmer
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer, 
        data_path=os.path.join(data_args.data_path, "test.csv"), 
        kmer=data_args.kmer
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    extra_args = {}
    if model_args.use_fp16:
        extra_args["torch_dtype"] = torch.float16
    elif model_args.use_8bit:
        extra_args["load_in_8bit"] = True
        extra_args["torch_dtype"] = torch.float16  
    elif model_args.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        extra_args["quantization_config"] = bnb_config

    # Load pretrained model
    model = transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        **extra_args
    )
    # Wrap the pretrained model with the adapter (for regression)
    model = AdapterModel(pretrained_model=model, num_labels=train_dataset.num_labels)

    # Define trainer with the regression metric functions
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    custom_dir = training_args.saved_model_dir.strip()
    if not model_args.use_lora:
        if custom_dir:
            trainer.save_model(custom_dir)
        else:
            trainer.save_model("./Trained_model")
    if model_args.use_lora:
        model = model.merge_and_unload()
        if custom_dir:
            model.save_pretrained(custom_dir)
            tokenizer.save_pretrained(custom_dir)
        else:
            model.save_pretrained("./lora_Trained_model")
            tokenizer.save_pretrained("./lora_Trained_model")

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Evaluate the model on the test dataset and save results
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    train_sft()
