import csv
import numpy as np
import collections

with open("/root/data/cami2/marine/binning_5.tsv", "r") as f:
    lines = list(csv.reader(f, delimiter="\t"))[1:]
    lines_known = [line for line in lines if line[1].startswith("Otu")]
    lines_unknown = [line for line in lines if not line[1].startswith("Otu")]

min_len = 2048
lines_known = [line for line in lines_known if len(line[0]) > min_len]
lines_unknown = [line for line in lines_unknown if len(line[0]) > min_len]
print(len(lines_known), len(lines_unknown))

lines_known = [line[:min_len] for line in lines_known[:500]]
lines_unknown = [line[:min_len] for line in lines_unknown[:500]]

with open("/root/data/cami2/marine/binning_5_known.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(lines_known)

with open("/root/data/cami2/marine/binning_5_unknown.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(lines_unknown)



"""
Unknown species
"""

import csv
import random
import numpy as np
import collections

random.seed(0)

sequences = collections.defaultdict(list)
length = 2000
for env in ["marine", "plant"]:
    for data in [5,6,7,8,9]:
        print(f"Processing {env} {data}")
        with open(f"/root/data/cami2/{env}/binning_{data}.tsv", "r") as f:
            lines = list(csv.reader(f, delimiter="\t"))[1:]
            lines_unknown = [line for line in lines if (line[1].startswith("RNODE")) and len(line[0]) >= length]
            # lines_unknown = [line for line in lines if (not line[1].startswith("Otu") and line[1].startswith("RNODE")) and len(line[0]) >= length]
            for line in lines_unknown:
                seq = line[0]
                all_seqs = [seq[:length]]
                if len(seq) // length > 1:
                    for i in range(1, len(seq) // length):
                        all_seqs.append(seq[i*length:(i+1)*length])
                sequences[line[1]].extend(all_seqs)

selected_species = [species for species in sequences if len(sequences[species]) >= 100]
selected_species = selected_species[:20]
new_sequences = {}
for species in selected_species:
    random.shuffle(sequences[species])
    new_sequences[species] = sequences[species][:100]
    
with open("/root/data/cami2/marine_plant_20_unknown.tsv", "w") as f:
    for species in new_sequences:
        for seq in new_sequences[species]:
            f.write(f"{seq}\t{species}\n")
        
        



"""
Known species
"""

import csv
import random
import numpy as np
import collections

random.seed(0)

sequences = collections.defaultdict(list)
length = 16000
for env in ["marine", "plant"]:
    for data in [5,6,7,8,9]:
        print(f"Processing {env} {data}")
        with open(f"/root/data/cami2/{env}/binning_{data}.tsv", "r") as f:
            lines = list(csv.reader(f, delimiter="\t"))[1:]
            lines_unknown = [line for line in lines if (not line[1].startswith("RNODE")) and len(line[0]) >= length]
            # lines_unknown = [line for line in lines if (not line[1].startswith("Otu") and line[1].startswith("RNODE")) and len(line[0]) >= length]
            for line in lines_unknown:
                seq = line[0]
                all_seqs = [seq[:length]]
                if len(seq) // length > 1:
                    for i in range(1, len(seq) // length):
                        all_seqs.append(seq[i*length:(i+1)*length])
                sequences[line[1]].extend(all_seqs)

selected_species = [species for species in sequences if len(sequences[species]) >= 100]
print(f"Selected species: {len(selected_species)}")
random.shuffle(selected_species)
selected_species = selected_species[:30]
new_sequences = {}
for species in selected_species:
    random.shuffle(sequences[species])
    new_sequences[species] = sequences[species][:100]
    
with open("/root/data/cami2/marine_plant_30_known.tsv", "w") as f:
    for species in new_sequences:
        for seq in new_sequences[species]:
            f.write(f"{seq}\t{species}\n")
            
with open("/root/data/cami2/marine_plant_30_known_2k.tsv", "w") as f:
    for species in new_sequences:
        for seq in new_sequences[species]:
            f.write(f"{seq[-2000:]}\t{species}\n")
        