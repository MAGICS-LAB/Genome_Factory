import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import csv
import argparse
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def calculate_llm_score(dna_sequences, model_name_or_path, model_max_length=400, batch_size=20):
    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path
    
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")

    all_scores = 0
    train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda() if 'attention_mask' in token_feat else torch.ones_like(input_ids).cuda()
            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
            else:
                model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
            model_output = torch.clip(model_output, 0, 1)
            all_scores += model_output.sum().item()
            
    return all_scores/len(dna_sequences)




def calculate_clm_loss(dna_sequences, model_name_or_path, model_max_length=400, batch_size=25):
    # Reorder the sequences by length to speed up computation
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")

    # Prepare the data loader
    train_loader = util_data.DataLoader(
        dna_sequences, batch_size=batch_size * max(n_gpu, 1), shuffle=False, num_workers=2 * max(n_gpu, 1)
    )

    all_seq_loss = []
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                batch, 
                max_length=model_max_length, 
                return_tensors='pt', 
                padding='longest', 
                truncation=True,
                return_special_tokens_mask=True
            )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            special_tokens_mask = token_feat['special_tokens_mask'].cuda()
            
            # Prepare labels, setting special tokens and padding tokens to -100
            labels = input_ids.clone()
            labels[special_tokens_mask == 1] = -100
            labels[attention_mask == 0] = -100
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
            
            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute per-token loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_ids.size(0), -1)  # Shape: [batch_size, seq_length - 1]
            
            # Compute loss per sequence
            loss_sum = loss.sum(dim=1)
            valid_tokens = (shift_labels != -100).sum(dim=1)
            # Avoid division by zero
            seq_loss = loss_sum / valid_tokens.float().clamp(min=1e-9)
            seq_loss = seq_loss.detach().cpu()
            
            all_seq_loss.append(seq_loss)

    # Concatenate all sequence losses and reorder them to match the original input order
    all_seq_loss = torch.cat(all_seq_loss, dim=0)
    all_seq_loss = all_seq_loss[np.argsort(idx)]
    
    return all_seq_loss.numpy()



if __name__ == "__main__":
    
    # data_dir = "/root/data/cami2/marine_plant_20_unknown.tsv"
    # data_dir = "/root/MOE_DNA/ICLR/augmentation/ref_1/no_aug/train.csv"
    data_dir = "/root/MOE_DNA/ICLR/generated/go_0.3_50_700_1000_0.0_0.0_1.0_1.csv"
    # data_dir = "/root/MOE_DNA/ICLR/generated/from_marine_plant_20_unknown_evo.txt"
    with open(data_dir, "r") as f:
        delimiter = "," if data_dir.endswith("csv") else "\t"
        lines = list(csv.reader(f, delimiter=delimiter))[1:1000]
        dna_sequences = [line[0] for line in lines]
    
    
    model_name_or_path = "/root/MOE_DNA/ICLR/output/dnabert2_regression/checkpoint-2500"
    scores = calculate_llm_score(dna_sequences, model_name_or_path, model_max_length=512, batch_size=64)
    print(scores)
    
    # model_name_or_path = "/root/weiminwu/dnabert-3/ICLR/model/meta_hmp_seq10240"
    # scores = calculate_clm_loss(dna_sequences, model_name_or_path, model_max_length=500, batch_size=25)
    # print(scores)
    # print(scores.mean())