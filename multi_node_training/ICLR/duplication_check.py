
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




# sequences = []
# length = 4000
# for env in ["marine", "plant"]:
#     for data in [5]:
#         print(f"Processing {env} {data}")
#         with open(f"/root/data/cami2/{env}/binning_{data}.tsv", "r") as f:
#             lines = list(csv.reader(f, delimiter="\t"))[1:]
#             # lines_unknown = [line for line in lines if (not line[1].startswith("Otu") and line[1].startswith("RNODE")) and len(line[0]) >= length]
#             for line in lines:
#                 seq = line[0]
#                 all_seqs = [seq[:length]]
#                 if len(seq) // length > 1:
#                     for i in range(1, len(seq) // length):
#                         all_seqs.append(seq[i*length:(i+1)*length])
#                 sequences.extend(all_seqs)

# print(len(sequences))
# random.shuffle(sequences)
# sequences = sequences[:20000]
# sequences = [seq for seq in sequences if len(seq) >= 4000]
# sequences = sequences[:1000]

# similarities = []
# for seq in sequences:
#     similarity = kmer_similarity_normalized(seq[:2000], seq[2000:4000])
#     similarities.append(similarity)
    
# with open("/root/MOE_DNA/ICLR/data/similarity/real_real.json", "w") as f:
#     json.dump(similarities, f)




with open("/root/MOE_DNA/ICLR/generated/cai/genslm.csv", "r") as f:
    seq_genslm = list(csv.reader(f))[1:]
    
with open("/root/MOE_DNA/ICLR/generated/cai/go_1.0_-1_0.9_600_1000_0.5_0.5_1.0_1.csv", "r") as f:
    seq_go = list(csv.reader(f))[1:]

with open("/root/MOE_DNA/ICLR/generated/cai/evo_microbes_cds_filtered_1.0_0_700.txt", "r") as f:
    lines = f.readlines()
    seq_evo = [line.split("\t") for line in lines]
    
similarity_genslm = []
similarity_go = []
similarity_evo = []

for seq in seq_genslm:
    similarity = kmer_similarity_normalized(seq[0], seq[1])
    similarity_genslm.append(similarity)
    
for seq in seq_go:
    similarity = kmer_similarity_normalized(seq[0], seq[1])
    similarity_go.append(similarity)
    
for seq in seq_evo:
    similarity = kmer_similarity_normalized(seq[0], seq[1])
    similarity_evo.append(similarity)
    
print(f"GenSLM: {np.mean(similarity_genslm)}")
print(f"GO: {np.mean(similarity_go)}")
print(f"Evo: {np.mean(similarity_evo)}")




with open("/root/MOE_DNA/ICLR/generated/coding_non_coding/go_1.0_-1_0.9_600_1000_0.5_0.5_1.0_1.csv", "r") as f:
    lines = list(csv.reader(f))
    for line in lines[:10]:
        print(line[0][:2000])
        print(line[1])
        print("\n")