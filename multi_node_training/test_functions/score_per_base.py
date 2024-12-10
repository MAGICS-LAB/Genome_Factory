import os
import numpy as np
import collections
import transformers
import torch
import torch.nn as nn
import tqdm

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
        dna_sequences.append(seq[:12])
dna_sequences = dna_sequences[:2]
    

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B"


def count_vocab(tokenizer):
    vocab = tokenizer.get_vocab()
    stats = collections.defaultdict(list)
    for token, idx in vocab.items():
        for char in ["A", "T", "C", "G"]:
            if token.startswith(char):
                stats[char].append(idx)
    
    #for char in stats:
    #    print(f"Number of tokens starting with {char}: {len(stats[char])}")
    
    for char in ["A", "T", "C", "G"]:
        char_mask = np.zeros(len(vocab))
        for idx in stats[char]:
            char_mask[idx] = 1
        
        stats[char] = char_mask
        
    return stats
    


# This function computes the perplexity of a list of DNA sequences using a model in model_dir
# Returns a numpy array of perplexities
def compute_perplexity(dna_sequences, model_dir):
    print(f"Computing per-base perplexity for {len(dna_sequences)} sequences")
    print(f"Model directory: {model_dir}")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )
    
    model.to("cuda")

    max_length = 10240 # current max length of the model
    device = "cuda"
    
    all_losses = []
    counter = collections.defaultdict(int)
    
    # In per-base prediction, we only consider 4 possible tokens: A, T, C, and G. 
    # E.g., the probability of having A as the next token equals the sum of the probabilities of all tokens that start with A.
    vocab_stats = count_vocab(tokenizer)
    MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
    with torch.no_grad():
        for i, seq in tqdm.tqdm(enumerate(dna_sequences)):
            cur_labels = [MAPPING.get(char) for char in seq[1:]] 
            cur_losses = []
            
            for j in range(len(seq) - 1):
                sample = seq[:j + 1]
                label = torch.tensor(cur_labels[j])
                sample = tokenizer(sample, padding=False, return_tensors="pt")
                input_ids = sample["input_ids"][:, :-1].to(device)
                attention_mask = sample["attention_mask"][:, :-1].to(device)
            
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                past_key_values = outputs.past_key_values
                print(past_key_values)
                raw_logits = outputs.logits
                raw_logits = raw_logits[0, -1, :].cpu().detach() # Get the logits of the last token
                
                logits = torch.zeros(4)
                for char, idx in MAPPING.items():
                    logits[idx] = np.sum(raw_logits[vocab_stats[char] == 1]) / 1024
                
                logits_tensor = torch.tensor(logits)
                label_tensor = torch.tensor(label)
                
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(logits_tensor, label_tensor)
                cur_losses.append(loss.item())
                
            
            all_losses.append(cur_losses)
        
    
    return all_losses



all_losses = compute_perplexity(dna_sequences, model_dir)
# print([len(loss) for loss in all_losses])
print(all_losses[0])