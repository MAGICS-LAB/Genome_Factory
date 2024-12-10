import os
import csv
import random
import numpy as np


root_dir = "/root/MOE_DNA/ICLR/data/microbes_cds_csv_filtered"
data = []

for file_name in os.listdir(root_dir):
    species = file_name.split(".")[0]
    with open(os.path.join(root_dir, file_name), "r") as f:
        lines = list(csv.reader(f))[1:]
    
    for line in lines:
        data.append((line[0], line[1], species))
        
print(f"Total data: {len(data)}")

with open("/root/MOE_DNA/ICLR/data/microbes_cds_filtered.csv", "w") as f:
    for d in data:
        f.write(f"{d[0]},{d[1]},{d[2]}\n")
        
        
        

"""
Merge evo
"""

# dir_1 = "/root/MOE_DNA/ICLR/generated/cai/evo_microbes_cds_filtered_1.0_0_233.txt"
# dir_2 = "/root/MOE_DNA/ICLR/generated/cai/evo_microbes_cds_filtered_1.0_233_466.txt"
# dir_3 = "/root/MOE_DNA/ICLR/generated/cai/evo_microbes_cds_filtered_1.0_466_700.txt"

# output_dir = "/root/MOE_DNA/ICLR/generated/cai/evo_microbes_cds_filtered_1.0_0_700.txt"

# with open(dir_1, "r") as f:
#     lines_1 = f.readlines()
# with open(dir_2, "r") as f:
#     lines_2 = f.readlines()
# with open(dir_3, "r") as f:
#     lines_3 = f.readlines()

# lines = lines_1 + lines_2 + lines_3
# with open(output_dir, "w") as f:
#     for line in lines:
#         f.write(line)

