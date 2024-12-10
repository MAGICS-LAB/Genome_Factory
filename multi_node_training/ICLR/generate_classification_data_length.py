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


    

for prompt in ["500", "1000", "2000", "4000", "8000", "16000"]:
    
    with open("/root/data/cami2/marine_plant_30_known_2k.tsv", "r") as f:
        lines = f.readlines()
        real_data_raw = [line.strip().split("\t") for line in lines]
    
    with open(f"/root/MOE_DNA/ICLR/generated/known/{prompt}_4B/go_1.0_-1_0.9_600_600_0.5_0.5_1.0_1.tsv", "r") as f:
        lines = f.readlines()
        go_data_raw = [line.strip().split("\t") for line in lines]


    real_data_raw, go_data_raw = real_data_raw[:1000], go_data_raw[:1000]
    postfix = f"3_{prompt}"
    # real_data_raw, go_data_raw = real_data_raw[1000:2000], go_data_raw[1000:2000]
    # postfix = f"4_{prompt}"
    # real_data_raw, go_data_raw= real_data_raw[2000:3000], go_data_raw[2000:3000]
    # postfix = f"5_{prompt}"



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
        go_data = [d for d in go_data_raw if d[1] == label]
        go_data = [(d[0][:max_sequence_length], label2id[label]) for d in go_data]
        
        train_data["real"].extend(real_data[:int(ratio_train*len(real_data))])
        train_data["go"].extend(go_data[:int(ratio_train*len(go_data))])
        
        val_data["real"].extend(real_data[int(ratio_train*len(real_data)):int((ratio_train+ratio_val)*len(real_data))])
        val_data["go"].extend(go_data[int(ratio_train*len(go_data)):int((ratio_train+ratio_val)*len(go_data))])
        
        test_data["real"].extend(real_data[int((ratio_train+ratio_val)*len(real_data)):])
        test_data["go"].extend(go_data[int((ratio_train+ratio_val)*len(go_data)):])
        
        
        # similarity between real samples and go samples
        real_go_similarity = [kmer_similarity_normalized(real[0], go[0]) for real, go in zip(real_data, go_data)]
        shuffuled_real = real_data.copy()
        random.shuffle(shuffuled_real)
        real_real_similarity = [kmer_similarity_normalized(real1[0], real2[0]) for real1, real2 in zip(real_data, shuffuled_real) if real1[0] != real2[0]]
        real_go_similarities.extend(real_go_similarity)
        real_real_similarities.extend(real_real_similarity)

    print(f"Real vs GO similarity: {np.mean(real_go_similarities)} {np.min(real_go_similarities)} {np.max(real_go_similarities)}")
    print(f"Real vs Real similarity: {np.mean(real_real_similarities)} {np.min(real_real_similarities)} {np.max(real_real_similarities)}")
    
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


    save_data(train_data["real"], val_data["go"], test_data["go"], "real", "go", output_dir_species)
    save_data(train_data["go"], val_data["real"], test_data["real"], "go", "real", output_dir_species)

