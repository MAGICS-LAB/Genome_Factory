import os
import csv
import json
import random
import numpy as np
import collections

random.seed(0)

with open("/root/data/cami2/reference/clustering_0.tsv", "r") as f:
    lines = f.readlines()
    real_data_raw = [line.strip().split("\t")[0][:2000] for line in lines]
    print(f"Real data: {len(real_data_raw)}")
    
with open("/root/data/cami2/plant/clustering_0.tsv", "r") as f:
    lines = f.readlines()
    real_data_raw += [line.strip().split("\t")[0][:2000] for line in lines]
    print(f"Real data: {len(real_data_raw)}")


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

def segment_shuffule(sequence, segment_length=100):
    segments = [sequence[i:i+segment_length] for i in range(0, len(sequence), segment_length)]
    random.shuffle(segments)
    return ''.join(segments)

def mute_by_base(sequence, mutation_rate=0.5):
    """
    Mute a DNA sequence by replacing a specified percentage of bases with random different bases.
    
    :param sequence: str, the input DNA sequence
    :param mutation_rate: float, the fraction of bases to mutate (default 0.5 for 50%)
    :return: str, the mutated DNA sequence
    """
    bases = list(sequence)
    num_mutations = int(len(sequence) * mutation_rate)
    
    # Randomly select positions to mutate
    positions_to_mutate = random.sample(range(len(sequence)), num_mutations)
    
    for pos in positions_to_mutate:
        original_base = bases[pos]
        # List of bases excluding the original base
        possible_mutations = [b for b in 'ACGT' if b != original_base]
        # Randomly select a new base
        new_base = random.choice(possible_mutations)
        bases[pos] = new_base
    
    return ''.join(bases)

random.shuffle(real_data_raw)
real_data_to_keep = real_data_raw[:10000]
real_data_to_mutate = real_data_raw[10000:20000]
real_data = [(d, 1.0) for d in real_data_to_keep]
reorder_data = [(reorder_dna_sequence(d), 0.0) for d in real_data_to_mutate]
random_data = [(generate_random_dna_sequence(d), 0.0) for d in real_data_to_mutate]
segment_shuffle_data = [(segment_shuffule(d, 10), 0.0) for d in real_data_to_mutate]
mute_data = [(mute_by_base(d, 0.5), 0.0) for d in real_data_to_mutate]

print(f"Real data: {len(real_data)} Reorder data: {len(reorder_data)} Random data: {len(random_data)} Segment shuffle data: {len(segment_shuffle_data)} Mute data: {len(mute_data)}")

data = real_data + reorder_data + random_data + segment_shuffle_data + mute_data
random.shuffle(data)
train_data = data[:int(0.8*len(data))]
val_data = data[int(0.8*len(data)):]

def write_data(data, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["sequence", "label"])
        writer.writerows(data)

def save_data(train_data, val_data, test_data, train, test):
    save_dir = os.path.join(output_dir, f"{train}_{test}")
    os.makedirs(save_dir, exist_ok=True)
    write_data(train_data, os.path.join(save_dir, "train.csv"))
    write_data(val_data, os.path.join(save_dir, "dev.csv"))
    write_data(test_data, os.path.join(save_dir, "test.csv"))

output_dir = "/root/MOE_DNA/ICLR/regression"
os.makedirs(output_dir, exist_ok=True)
save_data(train_data, val_data, val_data, "real", "rand_reorder")

