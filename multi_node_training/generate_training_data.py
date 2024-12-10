import os
import random

file2seq = {}
file2bp = {}
min_len = 500
max_len = 10000

data_root = "/root/data/pre-train/metagenomics/"


all_sequences = []

all_seq_count = 0
all_bp_count = 0
for file_name in ["antarctic.fa", "gre.fa", "harvardforest.fa", "mendota.fa", "neon.fa", "oilcane.fa", "tara.fa"]:
    current_seq_count = 0
    current_bp_count = 0
    print(f"Processing {file_name}")
    with open(os.path.join(data_root, file_name), 'r') as file:
        sequence = ""
            
        # Process the .fa file line by line
        for line in file:
            if line.startswith('>'):
                # This is a header line
                if len(sequence) >= min_len:
                    for i in range(0, len(sequence), max_len):
                        all_sequences.append(sequence[i:i+max_len])
                    total_bp = len(sequence)
                    current_seq_count += total_bp // max_len
                    current_bp_count += total_bp
                                        
                sequence = ""
            else:
                # This is a sequence line
                sequence += line.strip()  # Append the sequence line to the current sequence
        
        if len(sequence) >= min_len:
            for i in range(0, len(sequence), max_len):
                all_sequences.append(sequence[i:i+max_len])
            total_bp = len(sequence)
            current_seq_count += total_bp // max_len
            current_bp_count += total_bp
    
    print(f"Processed {current_seq_count} sequences and {current_bp_count} base pairs")
    all_seq_count += current_seq_count
    all_bp_count += current_bp_count
    file2seq[file_name] = current_seq_count
    file2bp[file_name] = current_bp_count

print(f"In total: Processed {all_seq_count} sequences and {all_bp_count} base pairs")

num_dev = 100000

random.shuffle(all_sequences)

with open(os.path.join(data_root, "train.txt"), 'w') as train_file:
    for sequence in all_sequences[num_dev:]:
        train_file.write(sequence + "\n")
with open(os.path.join(data_root, "dev.txt"), 'w') as dev_file:
    for sequence in all_sequences[:num_dev]:
        dev_file.write(sequence + "\n")
        
        
        
# in global_step16000
import torch   
model = torch.load("mp_rank_00_model_states.pt", map_location='cpu')
model["universal_checkpoint_info"] = {
    "universal_checkpoint_version": 0.2,
    "original_vocab_size": 4096,
    "padded_vocab_size": 4096
}
model["iteration"] = 16000
torch.save(model, "mp_rank_00_model_states.pt")


# in global_step16000_universal
import torch   
model = torch.load("mp_rank_00_model_states.pt", map_location='cpu')
model["iteration"] = torch.Tensor([16000])
torch.save(model, "mp_rank_00_model_states.pt")


# python ds_to_universal.py --input_folder /pscratch/sd/z/zhihanz/models/mistral_4B_1024_new/checkpoint-16000/global_step16000 --output_folder /pscratch/sd/z/zhihanz/models/mistral_4B_1024_new/checkpoint-16000/global_step16000_universal --num_extract_workers 8 --num_merge_workers 8 --keep_temp_folder


# read .fa file
import os
import random
from Bio import SeqIO

# Specify the path to your FASTA file
fasta_file = "/root/data/pre-train/BGC/allbgcs_dedeup.fa"

# Open the FASTA file and read each record
sequences = []
with open(fasta_file, "r") as file:
    for record in SeqIO.parse(file, "fasta"):
        sequences.append(str(record.seq))

print(f"Read {len(sequences)} sequences from {fasta_file}")

import numpy as np

lengths = [len(seq) for seq in sequences]
print(f"Min length: {np.min(lengths)} Max length: {np.max(lengths)} Mean length: {np.mean(lengths)}")

random.shuffle(sequences)
num_validation = 1000
train = sequences[num_validation:]
validation = sequences[:num_validation]

with open("/root/data/BGC/20240423/allbgcs_train.txt", "w") as f:
    for seq in train:
        f.write(seq + "\n")
with open("/root/data/BGC/20240423/allbgcs_validation.txt", "w") as f:
    for seq in validation:
        f.write(seq + "\n")
        
        
        
        
"""
further train meta on mixture of human-related microbe and metagenomics data on 20000-bp sequences
"""        


import os
import random

min_len = 500
max_len = 50000

data_root = "/root/data/pre-train/metagenomics/processed/meta_min500_max10000"
all_sequences = []
with open(os.path.join(data_root, "train_ae.txt"), 'r') as file:
    for line in file:
        seq = line.strip().strip("\n")
        if len(seq) >= min_len:
            all_sequences.append(seq)

print(f"Read {len(all_sequences)} sequences from {data_root}")
print(f"Total number of base pairs in Billion: {sum([len(seq) for seq in all_sequences]) / 1e9}")

min_len = 500
max_len = 50000
hmp_sequences = []
hmp_seq_count = 0
hmp_bp_count = 0
with open(os.path.join("/root/data/pre-train/metagenomics/", "hmp2.fa"), 'r') as file:
    sequence = ""
        
    # Process the .fa file line by line
    for line in file:
        if line.startswith('>'):
            # This is a header line
            if len(sequence) >= min_len:
                for i in range(0, len(sequence), max_len):
                    hmp_sequences.append(sequence[i:i+max_len])
                total_bp = len(sequence)
                hmp_seq_count += total_bp // max_len
                hmp_bp_count += total_bp
                                    
            sequence = ""
        else:
            # This is a sequence line
            sequence += line.strip()  # Append the sequence line to the current sequence

print(f"Processed {hmp_seq_count} sequences and {hmp_bp_count} base pairs")
print(f"Total number of base pairs in Billion: {hmp_bp_count / 1e9}")


random.shuffle(all_sequences)
all_sequences = all_sequences[:50000000]
print("Meta: Total number of base pairs in Billion: ", sum([len(seq) for seq in all_sequences]) / 1e9)
print("HMP: Total number of base pairs in Billion: ", hmp_bp_count / 1e9)
all_sequences.extend(hmp_sequences)
random.shuffle(all_sequences)
print(f"Total number of sequences: {len(all_sequences)}")
print(f"Total number of base pairs in Billion: {sum([len(seq) for seq in all_sequences]) / 1e9}")

num_dev = 10000
# data_root_hmp = "/root/data/pre-train/metagenomics/processed/hmp_meta_min1000_max20000"
data_root_hmp = "/root/data/pre-train/metagenomics/processed/hmp_meta_min500_max50000"
os.makedirs(data_root_hmp, exist_ok=True)
with open(os.path.join(data_root_hmp, "train.txt"), 'w') as train_file:
    for sequence in all_sequences[num_dev:]:
        train_file.write(sequence + "\n")
with open(os.path.join(data_root_hmp, "dev.txt"), 'w') as dev_file:
    for sequence in all_sequences[:num_dev]:
        dev_file.write(sequence + "\n")









"""
Training long BGC model
"""
# read .fa file
import os
import random
from Bio import SeqIO

# Specify the path to your FASTA file
fasta_file = "/root/data/pre-train/BGC/allbgcs_dedeup.fa"

# Open the FASTA file and read each record
sequences = []
with open(fasta_file, "r") as file:
    for record in SeqIO.parse(file, "fasta"):
        sequences.append(str(record.seq))

print(f"Read {len(sequences)} sequences from {fasta_file}")

import numpy as np

lengths = [len(seq) for seq in sequences]
print(f"Min length: {np.min(lengths)} Max length: {np.max(lengths)} Mean length: {np.mean(lengths)}")

# remove sequences shorter than 10k and split sequences longer than 100k
min_len = 10000
max_len = 100000
new_sequences = []
for seq in sequences:
    if len(seq) < min_len:
        continue
    if len(seq) <= max_len:
        new_sequences.append(seq)
    else:
        for i in range(0, len(seq), max_len):
            piece = seq[i:i+max_len]
            if len(piece) >= min_len:
                new_sequences.append(piece)

random.shuffle(new_sequences)
num_validation = 1000
train = new_sequences[num_validation:]
validation = new_sequences[:num_validation]

with open("/root/data/pre-train/BGC/allbgcs_len10k_train.txt", "w") as f:
    for seq in train:
        f.write(seq + "\n")
with open("/root/data/pre-train/BGC/allbgcs_len10k_validation.txt", "w") as f:
    for seq in validation:
        f.write(seq + "\n")
        
        
     