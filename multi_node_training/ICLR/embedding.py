import os
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import argparse
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




def calculate_go_embedding(dna_sequences, model_name_or_path, model_max_length=400, batch_size=25):
    # reorder the sequences by length
    # process sequences with similar lengths in the same batch can greatly speed up the computation
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


    model = transformers.AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )

            
    

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")


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
            attention_mask = token_feat['attention_mask'].cuda()
            model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
                
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            
            if j==0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().float().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings



def calculate_llm_embedding(dna_sequences, model_name_or_path, model_max_length=400, batch_size=20):
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
    
    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        ) 
    else:
        model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")


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
                
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            
            if j==0:
                embeddings = embedding
            else:
                
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings


with open("/root/data/cami2/marine_plant_20_unknown.tsv", "r") as f:
    lines = f.readlines()
    real_data_raw = [line.strip().split("\t") for line in lines]

with open("/root/MOE_DNA/ICLR/generated/unknown/go_1.0_50_600_1000_0.5_0.5_1.0_1.txt", "r") as f:
    lines = f.readlines()
    go_data_raw = [line.strip().split("\t") for line in lines]

with open("/root/MOE_DNA/ICLR/generated/unknown/from_marine_plant_20_unknown_evo.txt", "r") as f:
    lines = f.readlines()
    evo_data_raw = [line.strip().split("\t") for line in lines]  

with open("/root/MOE_DNA/ICLR/generated/unknown/0_2000_from_marine_plant_20_unknown_genslm_2.5B.txt", "r") as f:
    lines = f.readlines()
    genslm_data_raw = [line.strip().split("\t") for line in lines]


num_seq = 2000
real_data_raw, go_data_raw, evo_data_raw, genslm_data_raw = real_data_raw[:num_seq], go_data_raw[:num_seq], evo_data_raw[:num_seq], genslm_data_raw[:num_seq]


sequences = [d[0][:2000] for d in real_data_raw + go_data_raw + evo_data_raw + genslm_data_raw]
labels = [0] * len(real_data_raw) + [1] * len(go_data_raw) + [2] * len(evo_data_raw) + [3] * len(genslm_data_raw)

# embeddings = calculate_go_embedding(sequences, "/root/weiminwu/dnabert-3/ICLR/model/meta_hmp_seq10240", model_max_length=500, batch_size=25)
# embeddings = calculate_go_embedding(sequences, "togethercomputer/evo-1-131k-base", model_max_length=2000, batch_size=4)
# embeddings = calculate_llm_embedding(sequences, "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", model_max_length=500, batch_size=25)
embeddings = calculate_llm_embedding(sequences, "zhihan1996/DNABERT-2-117M", model_max_length=500, batch_size=25)
# embeddings = calculate_llm_embedding(sequences, "LongSafari/hyenadna-small-32k-seqlen-hf", model_max_length=2000, batch_size=64)



print(embeddings.shape)


tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'x': tsne_results[:, 0],
    'y': tsne_results[:, 1],
    'label': labels
})

# Set up the plot
plt.figure(figsize=(10, 8))

# Use a color palette with distinct colors
color_palette = ['blue', 'orange', 'green', 'red']
sns.scatterplot(data=df, x='x', y='y', hue='label', palette=color_palette)

# Customize the plot
plt.title('t-SNE visualization of embeddings')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')

# Correct the legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Real', 'GO', 'EVO', 'GenSLM'], title='Label')

plt.savefig("/root/MOE_DNA/ICLR/embedding.png")