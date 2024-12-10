import os
import csv
import json
import random
import numpy as np
import collections
from itertools import product


random.seed(0)

def kmer_similarity_normalized(seq1, seq2, k=4):
    """
    Calculate the similarity between two DNA sequences based on normalized k-mer frequency.

    Parameters:
        seq1 (str): First DNA sequence.
        seq2 (str): Second DNA sequence.
        k (int): Length of k-mers.

    Returns:
        float: Cosine similarity between the normalized k-mer frequency vectors.
    """

    def get_kmer_frequencies(sequence, k, kmer_list):
        """Compute normalized k-mer frequencies."""
        counts = collections.Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))
        total = sum(counts.values())
        if total == 0:
            return np.zeros(len(kmer_list))
        return np.array([counts.get(kmer, 0) / total for kmer in kmer_list])

    # Generate all possible k-mers
    nucleotides = ['A', 'C', 'G', 'T']
    kmer_list = [''.join(p) for p in product(nucleotides, repeat=k)]

    # Get normalized k-mer frequencies for both sequences
    seq1_freqs = get_kmer_frequencies(seq1.upper(), k, kmer_list)
    seq2_freqs = get_kmer_frequencies(seq2.upper(), k, kmer_list)

    # Compute cosine similarity
    dot_product = np.dot(seq1_freqs, seq2_freqs)
    norm_seq1 = np.linalg.norm(seq1_freqs)
    norm_seq2 = np.linalg.norm(seq2_freqs)

    if norm_seq1 == 0 or norm_seq2 == 0:
        return 0.0

    similarity = dot_product / (norm_seq1 * norm_seq2)
    return similarity


    
def reorder_dna_sequence(original_sequence):
    # Count the occurrences of each nucleotide
    nucleotide_counts = collections.Counter(original_sequence.upper())
    
    # Create a list of nucleotides based on their counts
    nucleotides = []
    for nucleotide, count in nucleotide_counts.items():
        nucleotides.extend([nucleotide] * count)
    
    # Shuffle the list to randomize the order
    random.shuffle(nucleotides)
    
    # Join the shuffled nucleotides into a string
    new_sequence = ''.join(nucleotides)
    
    return new_sequence

def generate_random_dna_sequence(dna_sequence):
    length = len(dna_sequence)
    nucleotides = ['A', 'C', 'G', 'T']
    sequence = ''.join(random.choices(nucleotides, k=length))
    return sequence

generate_unknown = False

if generate_unknown:
    """
    unknown
    """
    with open("/root/data/cami2/marine_plant_20_unknown.tsv", "r") as f:
        lines = f.readlines()
        real_data_raw = [line.strip().split("\t") for line in lines]

    with open("/root/MOE_DNA/ICLR/generated/unknown/2000_4B/go_1.0_-1_0.9_600_1000_0.5_0.5_1.0_1.tsv", "r") as f:
        lines = f.readlines()
        go_data_raw = [line.strip().split("\t") for line in lines]

    with open("/root/MOE_DNA/ICLR/generated/unknown/from_marine_plant_20_unknown_evo_2k.txt", "r") as f:
        lines = f.readlines()
        evo_data_raw = [line.strip().split("\t") for line in lines]  
        
    with open("/root/MOE_DNA/ICLR/generated/unknown/0_2000_from_marine_plant_20_unknown_genslm_2.5B.txt", "r") as f:
        lines = f.readlines()
        genslm_data_raw = [line.strip().split("\t") for line in lines]


    # real_data_raw, go_data_raw, evo_data_raw, genslm_data_raw = real_data_raw[:1000], go_data_raw[:1000], evo_data_raw[:1000], genslm_data_raw[:1000]
    # postfix = "1"
    real_data_raw, go_data_raw, evo_data_raw, genslm_data_raw = real_data_raw[1000:2000], go_data_raw[1000:2000], evo_data_raw[1000:2000], genslm_data_raw[1000:2000]
    postfix = "2"

else:
    """
    known
    """
    with open("/root/data/cami2/marine_plant_30_known_2k.tsv", "r") as f:
        lines = f.readlines()
        real_data_raw = [line.strip().split("\t") for line in lines]

    with open("/root/MOE_DNA/ICLR/generated/known/2000/go_1.0_-1_0.9_600_1000_0.5_0.5_1.0_1.tsv", "r") as f:
        lines = f.readlines()
        go_data_raw = [line.strip().split("\t") for line in lines]

    with open("/root/MOE_DNA/ICLR/generated/known/evo_marine_plant_30_known_1.0_0_3000.txt", "r") as f:
        lines = f.readlines()
        evo_data_raw = [line.strip().split("\t") for line in lines]  
        
    with open("/root/weiminwu/dnabert-3/ICLR/genslm_gene/generated/0_3000_marine_plant_30_known_2k_genslm_2.5B.txt", "r") as f:
        lines = f.readlines()
        genslm_data_raw = [line.strip().split("\t") for line in lines]


    # real_data_raw, go_data_raw, evo_data_raw, genslm_data_raw = real_data_raw[:1000], go_data_raw[:1000], evo_data_raw[:1000], genslm_data_raw[:1000]
    # postfix = "3"
    # real_data_raw, go_data_raw, evo_data_raw, genslm_data_raw = real_data_raw[1000:2000], go_data_raw[1000:2000], evo_data_raw[1000:2000], genslm_data_raw[1000:2000]
    # postfix = "4"
    real_data_raw, go_data_raw, evo_data_raw, genslm_data_raw = real_data_raw[2000:3000], go_data_raw[2000:3000], evo_data_raw[2000:3000], genslm_data_raw[2000:3000]
    postfix = "5"


print(f"Real{len(real_data_raw)} {len(go_data_raw)} {len(evo_data_raw)} {len(genslm_data_raw)}")

unique_labels = set([d[1] for d in real_data_raw])
label2id = {label: i for i, label in enumerate(unique_labels)}


max_sequence_length = 2000
ratio_train = 0.5
ratio_val = 0.2
ratio_test = 0.3

train_data = collections.defaultdict(list)
val_data = collections.defaultdict(list)
test_data = collections.defaultdict(list)

real_go_similarities = []
real_evo_similarities = []
real_real_similarities = []
real_reorder_similarities = []
real_random_similarities = []
real_genslm_similarities = []

for label in label2id:
    real_data = [d for d in real_data_raw if d[1] == label]
    real_data = [(d[0][:max_sequence_length], label2id[label]) for d in real_data]
    reorder_data = [(reorder_dna_sequence(d[0]), d[1]) for d in real_data]
    random_data = [(generate_random_dna_sequence(d[0])[:max_sequence_length], d[1]) for d in real_data]
    go_data = [d for d in go_data_raw if d[1] == label]
    go_data = [(d[0][:max_sequence_length], label2id[label]) for d in go_data]
    evo_data = [d for d in evo_data_raw if d[1] == label]
    evo_data = [(d[0][:max_sequence_length], label2id[label]) for d in evo_data]
    genslm_data = [d for d in genslm_data_raw if d[1] == label]
    genslm_data = [(d[0][:max_sequence_length], label2id[label]) for d in genslm_data]
    
    train_data["real"].extend(real_data[:int(ratio_train*len(real_data))])
    train_data["reorder"].extend(reorder_data[:int(ratio_train*len(reorder_data))])
    train_data["random"].extend(random_data[:int(ratio_train*len(random_data))])
    train_data["go"].extend(go_data[:int(ratio_train*len(go_data))])
    train_data["evo"].extend(evo_data[:int(ratio_train*len(evo_data))])
    train_data["genslm"].extend(genslm_data[:int(ratio_train*len(genslm_data))])
    
    val_data["real"].extend(real_data[int(ratio_train*len(real_data)):int((ratio_train+ratio_val)*len(real_data))])
    val_data["reorder"].extend(reorder_data[int(ratio_train*len(reorder_data)):int((ratio_train+ratio_val)*len(reorder_data))])
    val_data["random"].extend(random_data[int(ratio_train*len(random_data)):int((ratio_train+ratio_val)*len(random_data))])
    val_data["go"].extend(go_data[int(ratio_train*len(go_data)):int((ratio_train+ratio_val)*len(go_data))])
    val_data["evo"].extend(evo_data[int(ratio_train*len(evo_data)):int((ratio_train+ratio_val)*len(evo_data))])
    val_data["genslm"].extend(genslm_data[int(ratio_train*len(genslm_data)):int((ratio_train+ratio_val)*len(genslm_data))])
    
    test_data["real"].extend(real_data[int((ratio_train+ratio_val)*len(real_data)):])
    test_data["reorder"].extend(reorder_data[int((ratio_train+ratio_val)*len(reorder_data)):])
    test_data["random"].extend(random_data[int((ratio_train+ratio_val)*len(random_data)):])
    test_data["go"].extend(go_data[int((ratio_train+ratio_val)*len(go_data)):])
    test_data["evo"].extend(evo_data[int((ratio_train+ratio_val)*len(evo_data)):])
    test_data["genslm"].extend(genslm_data[int((ratio_train+ratio_val)*len(genslm_data)):])
    
    
    # similarity between real samples and go samples
    real_go_similarity = [kmer_similarity_normalized(real[0], go[0]) for real, go in zip(real_data, go_data)]
    real_evo_similarity = [kmer_similarity_normalized(real[0], evo[0]) for real, evo in zip(real_data, evo_data)]
    real_genslm_similarity = [kmer_similarity_normalized(real[0], genslm[0]) for real, genslm in zip(real_data, genslm_data)]
    real_reorder_similarity = [kmer_similarity_normalized(real[0], reorder[0]) for real, reorder in zip(real_data, reorder_data)]
    real_random_similarity = [kmer_similarity_normalized(real[0], random[0]) for real, random in zip(real_data, random_data)]
    shuffuled_real = real_data.copy()
    random.shuffle(shuffuled_real)
    real_real_similarity = [kmer_similarity_normalized(real1[0], real2[0]) for real1, real2 in zip(real_data, shuffuled_real) if real1[0] != real2[0]]
    real_go_similarities.extend(real_go_similarity)
    real_evo_similarities.extend(real_evo_similarity)
    real_genslm_similarities.extend(real_genslm_similarity)
    real_real_similarities.extend(real_real_similarity)
    real_reorder_similarities.extend(real_reorder_similarity)
    real_random_similarities.extend(real_random_similarity)

print(f"Real vs GO similarity: {np.mean(real_go_similarities)} {np.min(real_go_similarities)} {np.max(real_go_similarities)}")
print(f"Real vs EVO similarity: {np.mean(real_evo_similarities)} {np.min(real_evo_similarities)} {np.max(real_evo_similarities)}")
print(f"Real vs Genslm similarity: {np.mean(real_genslm_similarities)} {np.min(real_genslm_similarities)} {np.max(real_genslm_similarities)}")
print(f"Real vs Real similarity: {np.mean(real_real_similarities)} {np.min(real_real_similarities)} {np.max(real_real_similarities)}")
print(f"Real vs Reorder similarity: {np.mean(real_reorder_similarities)} {np.min(real_reorder_similarities)} {np.max(real_reorder_similarities)}")
print(f"Real vs Random similarity: {np.mean(real_random_similarities)} {np.min(real_random_similarities)} {np.max(real_random_similarities)}")

# print(len(real_go_similarities), len(real_evo_similarities), len(real_genslm_similarities), len(real_real_similarities), len(real_reorder_similarities), len(real_random_similarities))
# with open("/root/MOE_DNA/ICLR/data/similarity/real_go_similarity.json", "w") as f:
#     json.dump(real_go_similarities, f)
# with open("/root/MOE_DNA/ICLR/data/similarity/real_evo_similarity.json", "w") as f:
#     json.dump(real_evo_similarities, f)
# with open("/root/MOE_DNA/ICLR/data/similarity/real_genslm_similarity.json", "w") as f:
#     json.dump(real_genslm_similarities, f)
# with open("/root/MOE_DNA/ICLR/data/similarity/real_reorder.json", "w") as f:
#     json.dump(real_reorder_similarities, f)


for key in train_data:
    print(key, len(train_data[key]), len(val_data[key]), len(test_data[key]))

output_dir_species = "/root/MOE_DNA/ICLR/classification/species"
if postfix is not None:
    output_dir_species = f"{output_dir_species}_{postfix}"
def write_data(data, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["sequence", "label"])
        writer.writerows(data)

def save_data(train_data, val_data, test_data, train, test, output_dir):
    save_dir = os.path.join(output_dir, f"{train}_{test}")
    os.makedirs(save_dir, exist_ok=True)
    write_data(train_data, os.path.join(save_dir, "train.csv"))
    write_data(val_data, os.path.join(save_dir, "dev.csv"))
    write_data(test_data, os.path.join(save_dir, "test.csv"))


save_data(train_data["real"], val_data["real"], test_data["real"], "real", "real", output_dir_species)
save_data(train_data["real"], val_data["go"], test_data["go"], "real", "go", output_dir_species)
save_data(train_data["go"], val_data["real"], test_data["real"], "go", "real", output_dir_species)
save_data(train_data["real"], val_data["evo"], test_data["evo"], "real", "evo", output_dir_species)
save_data(train_data["evo"], val_data["real"], test_data["real"], "evo", "real", output_dir_species)
save_data(train_data["real"], val_data["genslm"], test_data["genslm"], "real", "genslm", output_dir_species)
save_data(train_data["genslm"], val_data["real"], test_data["real"], "genslm", "real", output_dir_species)
save_data(train_data["real"], val_data["random"], test_data["random"], "real", "random", output_dir_species)
save_data(train_data["random"], val_data["real"], test_data["real"], "random", "real", output_dir_species)
save_data(train_data["real"], val_data["reorder"], test_data["reorder"], "real", "reorder", output_dir_species)
save_data(train_data["reorder"], val_data["real"], test_data["real"], "reorder", "real", output_dir_species)



output_dir_realfake = "/root/MOE_DNA/ICLR/classification/real_fake"
if postfix is not None:
    output_dir_realfake = f"{output_dir_realfake}_{postfix}"
for fake in ["go", "evo", "genslm", "random", "reorder"]:
    train_real = [(d[0], 1) for d in train_data['real']]
    val_read = [(d[0], 1) for d in val_data['real']]
    test_real = [(d[0], 1) for d in test_data['real']]
    train_fake = [(d[0], 0) for d in train_data[fake]]
    val_fake = [(d[0], 0) for d in val_data[fake]]
    test_fake = [(d[0], 0) for d in test_data[fake]]
    
    train_real_fake = train_real + train_fake
    val_real_fake = val_read + val_fake
    test_real_fake = test_real + test_fake
    
    print(f"real_fake: {len(train_real_fake)}, {len(val_real_fake)}, {len(test_real_fake)}")
    
    save_data(train_real_fake, val_real_fake, test_real_fake, "real", fake, output_dir_realfake)
    

output_dir_pairwise = "/root/MOE_DNA/ICLR/classification/pairwise"
if postfix is not None:
    output_dir_pairwise = f"{output_dir_pairwise}_{postfix}"
def generate_pairwise_data(real_data, fake_data):
    positive_pairs = [(real[0] + fake[0], 1) for real, fake in zip(real_data, fake_data)]
    fake_data = fake_data[::-1]
    negative_pairs = [(real[0] + fake[0], 0) for real, fake in zip(real_data, fake_data)]
    
    data = positive_pairs + negative_pairs
    random.shuffle(data)
    return data


for fake in ["go", "evo", "genslm", "random", "reorder"]:
    train_real = [(d[0], 1) for d in train_data['real']]
    val_read = [(d[0], 1) for d in val_data['real']]
    test_real = [(d[0], 1) for d in test_data['real']]
    train_fake = [(d[0], 0) for d in train_data[fake]]
    val_fake = [(d[0], 0) for d in val_data[fake]]
    test_fake = [(d[0], 0) for d in test_data[fake]]
    
    train_pairwise = generate_pairwise_data(train_real, train_fake)
    val_pairwise = generate_pairwise_data(val_read, val_fake)
    test_pairwise = generate_pairwise_data(test_real, test_fake)
    
    print(f"pairwise: {len(train_pairwise)}, {len(val_pairwise)}, {len(test_pairwise)}")
    
    save_data(train_pairwise, val_pairwise, test_pairwise, "real", fake, output_dir_pairwise)