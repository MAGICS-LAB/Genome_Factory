
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    DataCollatorForLanguageModeling,
)

from torch.cuda import reset_peak_memory_stats
from torch.cuda import max_memory_allocated
import torch
from torch import nn
from torch.optim import Adam

from torch import autocast
from torch.nn.parallel import DistributedDataParallel

num_gpus = torch.cuda.device_count()

# Load tokenizer and model


rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "2"))

dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
reset_peak_memory_stats()
torch.cuda.set_device(rank)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
config = AutoConfig.from_pretrained("./model/model_gpt2")

#     config._attn_implementation = "flash_attention_2"

#     config.is_causal = True
MODEL_TYPE = AutoModelForCausalLM


model = MODEL_TYPE.from_config(config, trust_remote_code=True)

# Wrap the model with FullyShardedDataParallel
fsdp_model = FullyShardedDataParallel(
    module=model,
    device_id=rank,
    auto_wrap_policy=partial(
        size_based_auto_wrap_policy,
        min_num_params=1e4,
    ),
)
optimizer = Adam(fsdp_model.parameters(), lr=3e-5)

# Example input text
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt").to(rank)
input_ids = inputs["input_ids"]

for _ in range(10):
    optimizer.zero_grad()
    print("start")
    output = fsdp_model(input_ids, labels=input_ids)
    print("end")
    loss = output.loss
    loss.backward()
    optimizer.step()
    memory = max_memory_allocated()

if rank == 0:
    print(
        f"GPT-2 large peak memory allocated using {num_gpus} GPUs: {memory / 1e9:.3f}G"
    )
