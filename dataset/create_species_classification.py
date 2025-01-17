
import os
import csv
import random
import math
from Bio import SeqIO

# If both NCBIDownloader and GenomeDataset classes are in NcbiDatasetCli.py, you might do:
# from NcbiDatasetCli import GenomeDataset
# Or if they are separated, import them accordingly:
# from some_module import GenomeDataset

from GenomeDataset import GenomeDataset  # Assuming we can import this directly

#############################################
# 1. Define the 10 species and label mapping
#############################################
species_list = [
    "Homo sapiens",
    "Mus musculus",
    "Drosophila melanogaster",
    "Pan troglodytes",
    "Saccharomyces cerevisiae",
    "Caenorhabditis elegans",
    "Arabidopsis thaliana",
    "Danio rerio",
    "Bos taurus",
    "Gallus gallus"
]
species_label_map = {species: idx for idx, species in enumerate(species_list)}


#############################################
# 2. Extract 10,000-bp segments from .fna files
#############################################
def extract_ordered_segments(fna_file: str, segment_length: int = 10000) -> list[str]:
    """
    Reads all contigs from the given .fna file in 10,000-bp increments.
    Any leftover smaller than segment_length is discarded.
    Segments containing 'N' are skipped.
    Segments are converted to uppercase.

    Returns a list of valid (uppercase) 10,000-bp segments.
    """
    segments = []
    # Parse each contig in the .fna file
    for record in SeqIO.parse(fna_file, "fasta"):
        seq_str = str(record.seq)
        seq_len = len(seq_str)

        # Iterate in steps of 'segment_length' (default=10,000)
        for start_idx in range(0, seq_len, segment_length):
            end_idx = start_idx + segment_length
            # Discard if it goes beyond the sequence length
            if end_idx > seq_len:
                break

            segment = seq_str[start_idx:end_idx].upper()  # Convert to uppercase
            if 'N' in segment:
                # Skip this segment if it contains 'N'
                continue

            segments.append(segment)
    return segments


#############################################
# 3. Retrieve up to 100 valid segments (10kb each) per species
#############################################
def get_species_segments(species: str, max_segments=100, seg_length=10000) -> list[tuple[str, int]]:
    """
    Uses GenomeDataset to download and locate all .fna files for the species.
    Extracts 10,000-bp segments in order, discarding any containing 'N'.
    Keeps only the first 100 valid segments (or fewer if not enough).
    Returns a list of (segment_string, species_label).
    """
    # 1) Initialize and download
   
    dataset = GenomeDataset(species=species, download_folder=None, download=False)

    # 2) Get .fna file paths from the GenomeDataset object
    fna_files = dataset.fna_files

    # 3) Extract ordered 10kb segments, discarding those with 'N'
    label = species_label_map[species]
    all_segments = []

    for fna_path in fna_files:
        valid_segments = extract_ordered_segments(fna_path, seg_length)
        all_segments.extend(valid_segments)

    # 4) Keep only the first 100 valid segments
    all_segments = all_segments[:max_segments]

    return [(seg, label) for seg in all_segments]


#############################################
# 4. Aggregate data, shuffle, and split into train/dev/test for finetune in the future
#############################################
def main():
    # Collect all (sequence, label) pairs
    all_data = []

    for sp in species_list:
        print(f"Processing species: {sp}")
        species_data = get_species_segments(
            species=sp,
            max_segments=100,      # Up to 100 segments
            seg_length=10000       # Each segment is 10,000 bp
        )
        all_data.extend(species_data)

    # Shuffle the data
    random.shuffle(all_data)

    # Split 8:1:1
    total_count = len(all_data)
    train_end = math.floor(total_count * 0.8)
    dev_end = math.floor(total_count * 0.9)

    train_data = all_data[:train_end]
    dev_data = all_data[train_end:dev_end]
    test_data = all_data[dev_end:]

    # Ensure the output directory exists
    output_dir = "species_classification"
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to write CSV with a header
    def write_csv(filename, data):
        with open(os.path.join(output_dir, filename), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write the header first
            writer.writerow(["sequence", "label"])
            # Then each row: [sequence, label]
            for seq_str, label_int in data:
                writer.writerow([seq_str, label_int])

    # Write train.csv, dev.csv, test.csv
    write_csv("train.csv", train_data)
    write_csv("dev.csv", dev_data)
    write_csv("test.csv", test_data)

    print("Data is generated and saved under 'species_classification' folder:")
    print(" - train.csv")
    print(" - dev.csv")
    print(" - test.csv")


if __name__ == "__main__":
    main()