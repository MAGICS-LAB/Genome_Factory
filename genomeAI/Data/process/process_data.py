import os
import random
import pandas as pd
import glob
import numpy as np
import argparse

def find_fna_files(root_dir):
    """Find all .fna files in subdirectories."""
    all_fna_files = []
    for dirpath, _, _ in os.walk(root_dir):
        fna_files = glob.glob(os.path.join(dirpath, "*.fna"))
        all_fna_files.extend(fna_files)
    return all_fna_files

def extract_dna_segments(fna_file, num_segments=100, segment_length=10000):
    """Extract DNA segments from .fna file."""
    segments = []
    with open(fna_file, 'r') as f:
        content = f.read()
    
    # Remove headers (lines starting with '>') and join all sequences
    lines = content.split('\n')
    sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
    sequence = sequence.upper()  # Convert to uppercase
    
    # Ensure sequence is long enough
    if len(sequence) < segment_length:
        print(f"Warning: Sequence in {fna_file} is too short ({len(sequence)} bp). Skipping.")
        return []
    
    # Extract random segments
    for _ in range(num_segments):
        if len(sequence) <= segment_length:
            # If sequence is exactly segment_length, use the whole sequence
            segments.append(sequence)
        else:
            # Otherwise, pick a random starting position
            start = random.randint(0, len(sequence) - segment_length)
            segment = sequence[start:start + segment_length]
            segments.append(segment)
    
    return segments

def process_data(root_dir, output_dir, segments_per_species=100, segment_length=10000, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    """Process data and create train/dev/test CSV files."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subdirectories (each represents a species)
    species_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    all_data = []
    
    # Process each species directory
    for species_idx, species_dir in enumerate(species_dirs):
        species_path = os.path.join(root_dir, species_dir)
        
        # Find all .fna files in this species directory
        fna_files = []
        for dirpath, _, _ in os.walk(species_path):
            fna_files.extend(glob.glob(os.path.join(dirpath, "*.fna")))
        
        if not fna_files:
            print(f"Warning: No .fna files found for species {species_dir}")
            continue
        
        print(f"Processing species {species_idx}: {species_dir} ({len(fna_files)} .fna files)")
        
        # Calculate how many segments to extract from each .fna file
        segments_per_file = segments_per_species // len(fna_files)
        extra_segments = segments_per_species % len(fna_files)
        
        species_segments = []
        
        # Process each .fna file
        for i, fna_file in enumerate(fna_files):
            # Determine number of segments for this file
            num_segments = segments_per_file
            if i < extra_segments:
                num_segments += 1
                
            # Extract segments from this file
            segments = extract_dna_segments(fna_file, num_segments=num_segments, segment_length=segment_length)
            species_segments.extend(segments)
            
        # Ensure we have exactly the required number of segments
        if len(species_segments) > segments_per_species:
            species_segments = species_segments[:segments_per_species]
        elif len(species_segments) < segments_per_species:
            print(f"Warning: Only found {len(species_segments)} segments for species {species_dir}, " 
                  f"needed {segments_per_species}")
            
        # Add segments with labels to the dataset
        for segment in species_segments:
            all_data.append({"sequence": segment, "label": species_idx})
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Calculate split sizes
    total_size = len(df)
    train_size = int(total_size * train_ratio)
    dev_size = int(total_size * dev_ratio)
    
    # Split the data
    train_df = df[:train_size]
    dev_df = df[train_size:train_size + dev_size]
    test_df = df[train_size + dev_size:]
    
    # Save to CSV files
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Data processing complete. Files saved to {output_dir}")
    print(f"Train: {len(train_df)} samples")
    print(f"Dev: {len(dev_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Total species: {len(species_dirs)}")
    print(f"Total samples: {len(df)} (should be {len(species_dirs) * segments_per_species})")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process DNA data from .fna files.')
    parser.add_argument('--root_dir', type=str, default="/mnt/c/Users/11817/GenomeAI/My_GenomeAI/All_data",
                        help='Directory containing species folders')
    parser.add_argument('--output_dir', type=str, default="./species_classification1",
                        help='Output directory for CSV files')
    parser.add_argument('--segments_per_species', type=str, default="100",
                        help='Number of segments to extract per species')
    parser.add_argument('--segment_length', type=str, default="10000",
                        help='Length of each DNA segment in nucleotides')
    parser.add_argument('--train_ratio', type=str, default="0.7",
                        help='Ratio of data to use for training')
    parser.add_argument('--dev_ratio', type=str, default="0.15",
                        help='Ratio of data to use for development')
    parser.add_argument('--test_ratio', type=str, default="0.15",
                        help='Ratio of data to use for testing')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Process data with specified parameters
    process_data(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        segments_per_species=int(args.segments_per_species),
        segment_length=int(args.segment_length),
        train_ratio=float(args.train_ratio),
        dev_ratio=float(args.dev_ratio),
        test_ratio=float(args.test_ratio)
    )
