import os
import numpy as np
import collections
import transformers
import torch
import torch.nn as nn
import tqdm
import time
from vllm import LLM, SamplingParams

# limit the maximum GPU memory usage

# set it to prevent the warning message when use the model
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Change this to the sequences you want to score
all_sequences = []
with open("/root/data/pre-train/len5k/dev_len5k.txt", "r") as f:
    all_sequences = f.readlines()

# dna_sequences = all_sequences[:10]

dna_sequences = []
for seq in all_sequences:
    if len(seq) > 2048:
        dna_sequences.append(seq[:1024])
dna_sequences = dna_sequences[:2]
    

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B_emb"


def count_vocab(tokenizer):
    vocab = tokenizer.get_vocab()
    stats = collections.defaultdict(list)
    for token, idx in vocab.items():
        for char in ["A", "T", "C", "G"]:
            if token.startswith(char):
                stats[char].append(idx)
    
    for char in stats:
        print(f"Number of tokens starting with {char}: {len(stats[char])}")
    
    for char in ["A", "T", "C", "G"]:
        char_mask = np.zeros(len(vocab))
        for idx in stats[char]:
            char_mask[idx] = 1
        
        stats[char] = char_mask
        
    return stats
    
# This function computes the perplexity of a list of DNA sequences using a model in model_dir
# Returns a numpy array of perplexities
def compute_perplexity(
        dna_sequences, 
        model_dir,
        max_length=10240,
        ):
    print(f"Model directory: {model_dir}")
    
    llm = LLM(
            model=model_dir,
            tokenizer=model_dir,
            tokenizer_mode="slow",
            trust_remote_code=True,
            seed=0,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.5,
            max_logprobs=4096,
            max_num_seqs=1024,
            max_model_len=10240,
        )
        
        
    sampling_params = SamplingParams(
            n=1,
            temperature=1, 
            top_k=-1,
            stop_token_ids=[2],
            max_tokens=1,
            min_tokens=1,
            detokenize=False,
            logprobs=512,
        )
    
    MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    vocab_stats = count_vocab(tokenizer)
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    dna_sequence = [seq[:max_length] for seq in dna_sequences]
    all_losses = []
    for seq in dna_sequences:
        cur_losses = []
        
        prompts = []
        for i in range(1, len(seq)):
            prompts.append("[CLS]"+seq[:i])
        
        
        prompt_token_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

        all_outputs = llm.generate(
            prompts=None,
            prompt_token_ids=prompt_token_ids, 
            sampling_params=sampling_params,
        )
        
        cur_labels = [MAPPING.get(char) for char in seq[1:]] 

        for j, llm_output in enumerate(all_outputs):
            logprobs = llm_output.outputs[0].logprobs[0]
            raw_logits = torch.zeros(4096)
            for idx, logprob in logprobs.items():
                raw_logits[idx] = np.exp(logprob.logprob)
            
            logits = torch.zeros(4)
            for char, idx in MAPPING.items():
                logits[idx] = torch.sum(raw_logits[vocab_stats[char] == 1]) 
            
            if abs(raw_logits.sum().item() - logits.sum().item()) < 0.1:
                logits /= logits.sum()
            
            label = torch.tensor(cur_labels[j])
            loss = loss_fct(logits, label)
            cur_losses.append(loss.item())

        all_losses.append(cur_losses)
    
    return all_losses




all_losses = compute_perplexity(dna_sequences, model_dir)
# print([len(loss) for loss in all_losses])
print(all_losses[0])