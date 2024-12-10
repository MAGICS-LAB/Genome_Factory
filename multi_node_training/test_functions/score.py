import os
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import argparse

# set it to prevent the warning message when use the model
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Change this to the sequences you want to score
dna_sequences = []
with open("/root/data/pre-train/metagenomics/gre_15k.txt", "r") as f:
    dna_sequences = f.readlines()[:200]
    

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B"

# This function computes the perplexity of a list of DNA sequences using a model in model_dir
# Returns a numpy array of perplexities
def compute_perplexity(dna_sequences, model_dir, use_ppl=False):
    print(f"Getting perplexity for {len(dna_sequences)} sequences")
    print(f"Model directory: {model_dir}")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )
    
    model.to("cuda")
    

    encodings = tokenizer(dna_sequences, 
                        padding=False,)

    max_length = 1024 # current max length of the model
    stride = 512
    device = "cuda"
    
    perplexities = np.zeros(len(dna_sequences))

    for i, sample in tqdm.tqdm(enumerate(encodings["input_ids"])):
        nlls = []
        prev_end_loc = 0
        sample = torch.tensor(sample).unsqueeze(0)
        seq_len = sample.size(1)
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = sample[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        losses = torch.stack(nlls).mean().item()
        score = torch.exp(losses).item() if use_ppl else losses
        perplexities[i] = score
    
    return perplexities

scores = compute_perplexity(dna_sequences, model_dir)
