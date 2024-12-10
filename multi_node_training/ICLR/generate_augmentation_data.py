

# import os 
# import numpy as np

# with open("/root/MOE_DNA/ICLR/generated/from_marine_plant_20_unknown_evo.txt", "r") as f:
#     lines = f.readlines()
#     evo_data_raw = [line.strip().split("\t") for line in lines]
#     evo_data_raw = [(d[0][:2000], d[1]) for d in evo_data_raw]

# with open("/root/MOE_DNA/ICLR/generated/go_1.0_50_600_1000_0.5_0.5_1.0_1.txt", "r") as f:
#     lines = f.readlines()
#     go_data_raw = [line.strip().split("\t") for line in lines]
#     go_data_raw = [(d[0][:2000], d[1]) for d in go_data_raw]
    
# with open("/root/data/cami2/marine_plant_20_unknown.tsv", "r") as f:
#     lines = f.readlines()
#     real_data_raw = [line.strip().split("\t") for line in lines]
#     real_data_raw = [(d[0][:2000], d[1]) for d in real_data_raw]

# print(f"Average length: {np.mean([len(d[0]) for d in real_data_raw + go_data_raw + evo_data_raw])}")    

# with open("/root/MOE_DNA/ICLR/generated/test/unknown.tsv", "w") as f:
#     for d in real_data_raw + go_data_raw + evo_data_raw:
#         f.write(f"{d[0]}\t{d[1]}\n")


import os
import csv
import json
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
import tqdm
import torch.utils.data as util_data



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

with open("/root/data/cami2/reference/clustering_0.tsv", "r") as f:
    lines = f.readlines()
    real_data_raw = [line.strip().split("\t") for line in lines[1:]]
    real_data_raw = [(d[0][:2000], d[1]) for d in real_data_raw]
    print(f"Real data: {len(real_data_raw)}")
    
labels = [d[1] for d in real_data_raw]
label2id = {label: i for i, label in enumerate(set(labels))}
print(len(label2id))
real_data = [(d[0], label2id[d[1]]) for d in real_data_raw]

# only keep 60 samples for each label, 10 for training, 10 for validation, 40 for testing
num_samples_per_label = 60
num_samples_per_label_train = 10
num_samples_per_label_val = 10
num_samples_per_label_test = 40

data_train = []
data_val = []
data_test = []

for label in label2id:
    sequences = [d[0] for d in real_data if d[1] == label2id[label]]
    sequences_train = sequences[:num_samples_per_label_train]
    sequences_val = sequences[50:50+num_samples_per_label_val]
    sequences_test = sequences[50+num_samples_per_label_val:50+num_samples_per_label_val+num_samples_per_label_test]
    
    data_train.extend([(seq, label2id[label]) for seq in sequences_train])
    data_val.extend([(seq, label2id[label]) for seq in sequences_val])
    data_test.extend([(seq, label2id[label]) for seq in sequences_test])
    
print(len(data_train), len(data_val), len(data_test))

print(f"Average sequence length: {np.mean([len(d[0]) for d in real_data])}")

output_dir = "/root/MOE_DNA/ICLR/augmentation/ref_1/no_aug"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "train.csv"), "w") as f:
    f.write("sequence,label\n")
    for d in data_train:
        f.write(f"{d[0]},{d[1]}\n")
with open(os.path.join(output_dir, "dev.csv"), "w") as f:
    f.write("sequence,label\n")
    for d in data_val:
        f.write(f"{d[0]},{d[1]}\n")
with open(os.path.join(output_dir, "test.csv"), "w") as f:
    f.write("sequence,label\n")
    for d in data_test:
        f.write(f"{d[0]},{d[1]}\n")
        

with open("/root/MOE_DNA/ICLR/generated/go_1.0_50_600_1000_0.5_0.5_1.0_1.csv", "r") as f:
    lines = list(csv.reader(f))
    go_data_raw = [(line[0][:2000], line[1]) for line in lines if line[1] != "label"]
    print(f"GO data: {len(go_data_raw)}")


print(f"Average augment sequence length: {np.mean([len(d[0]) for d in go_data_raw])}")    

### compute embedding for training data and augmentation data
# real_data = [d[0] for d in data_train]
# go_data = [d[0] for d in go_data_raw]
# real_embedding = calculate_go_embedding(real_data, "/root/weiminwu/dnabert-3/ICLR/model/meta_hmp_seq10240", model_max_length=512, batch_size=25)
# go_embedding = calculate_go_embedding(go_data, "/root/weiminwu/dnabert-3/ICLR/model/meta_hmp_seq10240", model_max_length=512, batch_size=25)
# last_embedding = np.zeros(3072)
# selected_data = []
# for label in label2id:
#     mask_real = np.array([d[1] == label2id[label] for d in data_train])
#     mask_go = np.array([int(d[1]) == label2id[label] for d in go_data_raw])
#     real_embedding_label = real_embedding[mask_real]
#     go_embedding_label = go_embedding[mask_go]
    
#     mean_real_embedding = np.mean(real_embedding_label, axis=0)
#     average_distance_real = np.mean(np.linalg.norm(real_embedding_label - mean_real_embedding, axis=1))
#     max_distance_real = np.max(np.linalg.norm(real_embedding_label - mean_real_embedding, axis=1))
#     average_distance_go = np.mean(np.linalg.norm(go_embedding_label - mean_real_embedding, axis=1))
#     top_10_go = np.argsort(np.linalg.norm(go_embedding_label - mean_real_embedding, axis=1))[:10]
#     average_distance_go_top_10 = np.mean(np.linalg.norm(go_embedding_label[top_10_go] - mean_real_embedding, axis=1))
    
    
#     average_distance_real_last = np.mean(np.linalg.norm(real_embedding_label - last_embedding, axis=1))
#     last_embedding = mean_real_embedding
#     print(f"Label: {label}, Average distance real: {average_distance_real}, Average distance go: {average_distance_go}, Average distance go top 10: {average_distance_go_top_10}, distance to last embedding: {average_distance_real_last}")
    
#     diff_go = np.linalg.norm(go_embedding_label - mean_real_embedding, axis=1)
#     # select the ones that has a distance smaller than max_distance_real
#     selected_idx = np.where(diff_go <= max_distance_real)[0]
#     go_data_label = [go_data_raw[idx] for idx in np.where(mask_go)[0]]
#     for idx in selected_idx:
#         selected_data.append(go_data_label[idx])
    
# print(f"Selected go: {len(selected_data)}")
    
# data_train += selected_data
print(f"Average length of training data: {np.mean([len(d[0]) for d in data_train])}")
data_train = [(d[0] + go[0], d[1]) for d, go in zip(data_train, go_data_raw)]
print(f"Averge length of training data after augmentation: {np.mean([len(d[0]) for d in data_train])}")
print(f"Total training data: {len(data_train)}")
output_dir = "/root/MOE_DNA/ICLR/augmentation/ref_1/aug"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "train.csv"), "w") as f:
    f.write("sequence,label\n")
    for d in data_train:
        f.write(f"{d[0]},{d[1]}\n")
# copy dev and test data
os.system(f"cp /root/MOE_DNA/ICLR/augmentation/ref_1/no_aug/dev.csv {output_dir}/dev.csv")
os.system(f"cp /root/MOE_DNA/ICLR/augmentation/ref_1/no_aug/test.csv {output_dir}/test.csv")
