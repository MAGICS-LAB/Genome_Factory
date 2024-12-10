

import os
import random

min_len = 500
max_len = 4000

data_root = "/root/data/pre-train/metagenomics/processed/meta_min500_max10000"
all_sequences = []
with open(os.path.join(data_root, "train_ad.txt"), 'r') as file:
    for line in file:
        seq = line.strip().strip("\n")
        if len(seq) >= min_len:
            all_sequences.append(seq)

print(f"Read {len(all_sequences)} sequences from {data_root}")
print(f"Total number of base pairs in Billion: {sum([len(seq) for seq in all_sequences]) / 1e9}")

min_len = 500
max_len = 4000
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

num_dev = 20000
# data_root_hmp = "/root/data/pre-train/metagenomics/processed/hmp_meta_min1000_max20000"
data_root_hmp = "/root/data/pre-train/metagenomics/processed/hmp_meta_min500_max4000"
os.makedirs(data_root_hmp, exist_ok=True)
with open(os.path.join(data_root_hmp, "train.txt"), 'w') as train_file:
    for sequence in all_sequences[num_dev:]:
        train_file.write(sequence + "\n")
with open(os.path.join(data_root_hmp, "dev.txt"), 'w') as dev_file:
    for sequence in all_sequences[:num_dev]:
        dev_file.write(sequence + "\n")





from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

model_dir = "jaandoui/DNABERT2-AttentionExtracted"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

seq = "ACGTAGCATGCATGCATGCATCGATCGATGCATGCAT"
inputs = tokenizer(seq, 
                   return_tensors="pt"
                   )
outputs = model(**inputs, output_hidden_states=True, return_dict=True)
all_hidden_states = outputs.hidden_states