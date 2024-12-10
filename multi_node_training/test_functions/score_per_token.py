import os
import numpy as np
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
        dna_sequences.append(seq[:2048])
dna_sequences = dna_sequences[:10]
    

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B"

# This function computes the perplexity of a list of DNA sequences using a model in model_dir
# Returns a numpy array of perplexities
def compute_perplexity(dna_sequences, model_dir):
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

    max_length = 10240 # current max length of the model
    device = "cuda"
    
    all_losses = []
    
    for i, sample in tqdm.tqdm(enumerate(encodings["input_ids"])):
        losses = []
        sample = torch.tensor(sample).unsqueeze(0)
        seq_len = sample.size(1)
        if seq_len > max_length:
            raise ValueError(f"Sequence length {seq_len} is greater than max_length {max_length}")
        
        input_ids = sample.to(device)
        target_ids = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels).cpu().detach().numpy().tolist()
            losses.extend(loss)
        
        print(np.mean(losses))

        
        all_losses.append(losses)
    
    # transfer the per token loss to
    per_token_losses = []
    for loss, sample in zip(all_losses, encodings["input_ids"]):
        sample = sample[1:-1]
        loss = loss[:-1]
        assert len(sample) == len(loss), f"Sample length {len(sample)} does not match loss length {len(loss)}"
        
        per_token_loss = []
        for token, token_loss in zip(sample, loss):
            token_length = len(tokenizer.decode([token]))
            per_token_loss.extend([token_loss] * token_length)
            
        per_token_losses.append(per_token_loss)
        
    
    return per_token_losses

all_losses = compute_perplexity(dna_sequences, model_dir)
print([len(loss) for loss in all_losses])
# print(all_losses[0])