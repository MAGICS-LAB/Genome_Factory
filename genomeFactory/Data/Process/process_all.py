#!/usr/bin/env python3

import argparse
import sys


# --- Original scripts embedded verbatim ---
PROCESS_EMP_CODE = r'''#!/usr/bin/env python3
# build_epimark_dataset.py  (H3K36me3 gene-body real signal)
import argparse, gzip, random, shutil, re
from pathlib import Path
import requests, pandas as pd, pyBigWig, numpy as np
from tqdm import tqdm
from intervaltree import Interval, IntervalTree
from pyfaidx import Fasta

ENS_REL = "110"
SRC = {
    "hg38": {
        "gtf": f"https://ftp.ensembl.org/pub/release-{ENS_REL}/gtf/homo_sapiens/"
               f"Homo_sapiens.GRCh38.{ENS_REL}.gtf.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
        # Using Roadmap H3K36me3 data - confirmed available
        "bw" : "https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/foldChange/"
               "E003-H3K36me3.fc.signal.bigwig"  # H1 Embryonic Stem Cell
    },
    "mm10": {
        "gtf": f"https://ftp.ensembl.org/pub/release-{ENS_REL}/gtf/mus_musculus/"
               f"Mus_musculus.GRCm39.{ENS_REL}.gtf.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
        # Using confirmed ENCODE mm10 H3K36me3 file from search results
        "bw" : "https://www.encodeproject.org/files/ENCFF179NTO/@@download/ENCFF179NTO.bigWig"  # mm10 H3K36me3
    },
}
gene_id_re = re.compile(r'gene_id "([^"]+)"')

def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length",0))
        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=f"↓ {dst.name}") as bar, open(dst,"wb") as f:
            for blk in r.iter_content(1<<20):
                f.write(blk); bar.update(len(blk))

def gunzip(src, dst):
    if dst.exists(): return
    with gzip.open(src,"rb") as fi, open(dst,"wb") as fo: shutil.copyfileobj(fi,fo)

def load_gene_bodies(gtf_gz):
    with gzip.open(gtf_gz,"rt") as fh:
        for ln in fh:
            if ln[0]=="#": continue
            chrom,_,feat,start,end,_,strand,_,attrs = ln.rstrip("\n").split("\t")
            if feat!="gene": continue
            gid = gene_id_re.search(attrs).group(1)
            s,e=int(start)-1,int(end)  # 0-based half-open
            yield chrom,s,e,strand,gid

def to_tree(regs):
    idx={}
    for c,s,e,*_ in regs:
        idx.setdefault(c,IntervalTree()).add(Interval(s,e))
    return idx

def rc(seq): return seq.translate(str.maketrans("ACGTN","TGCAN"))[::-1]

def get_seq_safe(fa, c, s, e, strand):
    """Safely extract sequence with error handling"""
    try:
        seq = fa[c][s:e].upper()
        return rc(seq) if strand == "-" else seq
    except (KeyError, IndexError):
        return ""  # Return empty string if chromosome not found or coordinates invalid

def is_valid_sequence(seq: str, min_len: int = 100, max_n_ratio: float = 0.1) -> bool:
    """Check if sequence is valid: not empty, reasonable length, not too many Ns"""
    if not seq or len(seq) < min_len:
        return False
    n_count = seq.count('N')
    n_ratio = n_count / len(seq)
    return n_ratio <= max_n_ratio

def convert_chrom_name(chrom, target_chroms):
    """Convert chromosome names between different formats (e.g., '1' <-> 'chr1')"""
    # Direct match first
    if chrom in target_chroms:
        return chrom
    
    # Try adding 'chr' prefix
    chr_version = f"chr{chrom}"
    if chr_version in target_chroms:
        return chr_version
    
    # Try removing 'chr' prefix
    if chrom.startswith('chr'):
        no_chr_version = chrom[3:]
        if no_chr_version in target_chroms:
            return no_chr_version
    
    # No conversion found
    return None

def signal_to_class(signal_values):
    """Convert continuous signal to binary classification labels based on median"""
    # Use median as threshold for binary classification
    threshold = np.median(signal_values)
    
    labels = []
    for sig in signal_values:
        if sig <= threshold:
            labels.append(0)  # Low H3K36me3
        else:
            labels.append(1)  # High H3K36me3
    return labels

def main(out, seed=42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Dataset split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
    rng=random.Random(seed)
    splits=(out/"splits"); splits.mkdir(parents=True, exist_ok=True)
    all_df=[]
    for sp,urls in SRC.items():
        raw=out/"raw"/sp; raw.mkdir(parents=True,exist_ok=True)
        gtf_gz,fa_gz,fa_plain,bw_path=[ raw/p for p in ("gene.gtf.gz","genome.fa.gz","genome.fa","mark.bw") ]

        for url,path in zip(urls.values(),[gtf_gz,fa_gz,bw_path]):
            fetch(url,path)
        gunzip(fa_gz,fa_plain)

        genes=list(load_gene_bodies(gtf_gz))
        print(f"\n=== Processing {sp.upper()} ===")
        print(f"Loaded {len(genes)} genes from GTF")
        
        bw=pyBigWig.open(str(bw_path))
        bw_chroms = set(bw.chroms().keys())  # Convert to set for faster lookup
        
        fa=Fasta(fa_plain, as_raw=True, sequence_always_upper=True)
        fa_chroms = set(fa.keys())  # Convert to set for faster lookup
        
        idx=to_tree(genes)

        records=[]
        skipped_regions = 0
        
        for chrom,s,e,strand,gid in genes:
            try:
                # Convert chromosome names if needed
                bw_chrom = convert_chrom_name(chrom, bw_chroms)
                fa_chrom = convert_chrom_name(chrom, fa_chroms)
                
                # Check if chromosome exists in BigWig
                if bw_chrom is None:
                    skipped_regions += 1
                    continue
                
                # Check if chromosome exists in FASTA
                if fa_chrom is None:
                    skipped_regions += 1
                    continue
                
                # Get signal, handle None values
                sig = bw.stats(bw_chrom, s, e, type="mean", nBins=1)[0]
                if sig is None:
                    sig = 0.0
                
                # Extract sequence with error handling
                seq = get_seq_safe(fa, fa_chrom, s, e, strand)
                
                # Basic quality filters
                if not is_valid_sequence(seq):
                    skipped_regions += 1
                    continue
                    
                records.append((f"{sp}:{gid}", sp, sig, seq))
                
            except Exception as e:
                skipped_regions += 1
                continue
        
        print(f"Generated {len(records)} valid gene records for {sp} (skipped {skipped_regions})")
        bw.close()

        # Final validation before creating DataFrame
        valid_records = []
        for record_id, species, signal, sequence in records:
            if is_valid_sequence(sequence) and isinstance(signal, (int, float)):
                valid_records.append((record_id, species, signal, sequence))
        
        print(f"Final valid records for {sp}: {len(valid_records)}")
        df=pd.DataFrame(valid_records, columns=["id","species","signal","sequence"])
        all_df.append(df)

    if not all_df or all(len(df) == 0 for df in all_df):
        print("\n❌ ERROR: No valid records found!")
        print("This is likely due to chromosome name mismatches between GTF and BigWig files")
        return

    data=pd.concat(all_df, ignore_index=True).sample(frac=1, random_state=seed)
    
    # Comprehensive final cleanup: remove any remaining low-quality sequences
    initial_count = len(data)
    data = data[data['sequence'].str.len() >= 100]  # Minimum gene length
    data = data[data['sequence'].str.len() > 0]  # Remove empty sequences
    data = data[data['sequence'].str.contains('^[ACGTN]+$', na=False)]  # Valid DNA only
    data = data[data['sequence'].notna()]  # Remove NaN sequences
    
    # Additional quality check: remove sequences with too many Ns
    def has_valid_n_ratio(seq):
        if not seq or len(seq) == 0:
            return False
        return seq.count('N') / len(seq) <= 0.1
    
    data = data[data['sequence'].apply(has_valid_n_ratio)]
    
    print(f"Total samples after filtering: {len(data)} (removed {initial_count - len(data)})")
    
    if len(data) == 0:
        print("❌ ERROR: No records remaining after cleanup!")
        return
    
    # Convert continuous signal to binary classification labels
    labels = signal_to_class(data['signal'].values)
    data['label'] = labels
    
    # Show class distribution
    label_counts = pd.Series(labels).value_counts()
    print(f"Class distribution: Low={label_counts.get(0, 0)}, High={label_counts.get(1, 0)}")
    
    n=len(data); tr,va=int(n*train_ratio),int(n*(train_ratio+val_ratio))
    
    # Save with sequence and label columns for binary classification
    train_data = data[['sequence', 'label']].iloc[:tr]
    val_data = data[['sequence', 'label']].iloc[tr:va]
    test_data = data[['sequence', 'label']].iloc[va:]
    
    train_data.to_csv(splits/"train.csv",index=False)
    val_data.to_csv(splits/"dev.csv",index=False)
    test_data.to_csv(splits/"test.csv",index=False)
    
    print(f"Final dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print(f"✅ Dataset saved: {splits}")
    
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Build H3K36me3 binary classification dataset from gene bodies")
    ap.add_argument("--out_dir", type=Path, default=Path("epimark_dataset"),
                    help="Output directory for dataset (default: epimark_dataset)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="Fraction of data for training (default: 0.8)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of data for validation (default: 0.1)")
    
    args = ap.parse_args()
    
    # Validate ratios
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        ap.error("train_ratio must be between 0 and 1")
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        ap.error("val_ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        ap.error("train_ratio + val_ratio must be < 1.0 to leave room for test set")
    
    main(args.out_dir, args.seed, args.train_ratio, args.val_ratio)
'''


PROCESS_DATA_CODE = r'''import os
import random
import pandas as pd
import glob
import numpy as np
import argparse


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
    parser.add_argument('--root_dir', type=str, default="./All_data",
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
'''


PROCESS_ENHANCER_CODE = r'''#!/usr/bin/env python3
# build_enhancer_dataset.py
# -------------------------------------------------------------
# 1. Download FANTOM5 enhancer BED files (hg38_enhancers.bed / mm10_enhancers.bed)
# 2. Download UCSC reference genome FASTA (hg38 / mm10)
# 3. Extract [-500, +500] flanking regions for each enhancer (total length+1000 bp) as positives
# 4. Generate random negatives of same length, no overlap with positives
# 5. Shuffle and split into train/val/test CSV files
#    CSV format: id,species,label,sequence   (label=1 enhancer, 0 negative)
# -------------------------------------------------------------

import argparse, gzip, random, shutil
from pathlib import Path
from typing import List
import requests, pandas as pd
from tqdm import tqdm
from intervaltree import Interval, IntervalTree
from pyfaidx import Fasta

# ------------ Data sources -----------------
SOURCES = {
    "hg38": {
        "bed": "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/enhancer/F5.hg38.enhancers.bed.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    },
    "mm10": {
        "bed": "https://fantom.gsc.riken.jp/5/datafiles/latest/extra/Enhancers/mouse_permissive_enhancers_phase_1_and_2.bed.gz", 
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    },
}

FLANK = 500                     # Add 500 bp on each side

# ------------ Download / Unzip ---------------
def fetch(url: str, dst: Path, chunk: int = 1 << 20):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        tot = int(r.headers.get("content-length", 0))
        with tqdm(total=tot, unit="B", unit_scale=True,
                  desc=f"↓ {dst.name}") as bar, open(dst, "wb") as fh:
            for blk in r.iter_content(chunk):
                fh.write(blk); bar.update(len(blk))

def gunzip(src: Path, dst: Path):
    if dst.exists(): return
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)

# ------------ BED Parsing -----------------
def load_enhancers(bed: Path):
    # Check if the file is a .gz compressed file
    if bed.suffix == '.gz':
        fh = gzip.open(bed, 'rt')
    else:
        fh = open(bed)
    
    try:
        for ln in fh:
            if ln.startswith(("#","track","browser")) or not ln.strip():
                continue
            chrom, start, end, *rest = ln.rstrip("\n").split("\t")
            start, end = int(start), int(end)         # FANTOM5 is 0-based half-open
            yield chrom, start, end
    finally:
        fh.close()

def build_tree(regs):
    idx = {}
    for c,s,e in regs:
        idx.setdefault(c, IntervalTree()).add(Interval(s, e))
    return idx

def rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]

def convert_chrom_name(chrom, target_chroms):
    """Convert chromosome names between different formats (e.g., '1' <-> 'chr1')"""
    # Direct match first
    if chrom in target_chroms:
        return chrom
    
    # Try adding 'chr' prefix
    chr_version = f"chr{chrom}"
    if chr_version in target_chroms:
        return chr_version
    
    # Try removing 'chr' prefix
    if chrom.startswith('chr'):
        no_chr_version = chrom[3:]
        if no_chr_version in target_chroms:
            return no_chr_version
    
    # No conversion found
    return None

def get_seq_safe(fa, c, s, e):
    """Safely extract sequence with error handling"""
    try:
        seq = fa[c][s:e].upper()
        return seq
    except (KeyError, IndexError):
        return ""  # Return empty string if chromosome not found or coordinates invalid

def is_valid_sequence(seq: str, min_len: int = 100, max_n_ratio: float = 0.1) -> bool:
    """Check if sequence is valid: not empty, reasonable length, not too many Ns"""
    if not seq or len(seq) < min_len:
        return False
    n_count = seq.count('N')
    n_ratio = n_count / len(seq)
    return n_ratio <= max_n_ratio

# ------------ Random Negative Samples ---------------
def sample_negatives(fa: Fasta, idx, length: int, n: int, rng, fa_chroms):
    chroms = list(fa_chroms); lens = {c: len(fa[c]) for c in chroms}
    out=[]
    attempts = 0
    max_attempts = n * 10  # Avoid infinite loops
    
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        c = rng.choice(chroms)
        if lens[c] < length: continue
        s = rng.randint(0, lens[c] - length); e = s + length
        if c in idx and idx[c].overlap(s, e): continue
        
        # Check if sequence is valid
        seq = get_seq_safe(fa, c, s, e)
        if is_valid_sequence(seq, min_len=length//2):  # Allow shorter sequences for negatives
            out.append((c,s,e))
    
    return out

# ------------ Main Flow -------------------
def build(out_dir: Path, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Dataset split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
    rng=random.Random(seed)
    raw = out_dir/"raw"; splits = out_dir/"splits"
    splits.mkdir(parents=True, exist_ok=True)
    dfs=[]
    
    for sp, urls in SOURCES.items():
        print(f"\n=== Processing {sp.upper()} ===")
        bed = raw/f"{sp}_enhancers.bed.gz"  # Use .gz filename
        fa_gz, fa_plain = raw/f"{sp}.fa.gz", raw/f"{sp}.fa"
        fetch(urls["bed"], bed)
        fetch(urls["fa"], fa_gz); gunzip(fa_gz, fa_plain)
        fa = Fasta(fa_plain, as_raw=True, sequence_always_upper=True)
        
        # Get available chromosomes
        fa_chroms = set(fa.keys())
        
        # Load and process BED regions
        bed_regions = list(load_enhancers(bed))
    
        enhancers = []
        skipped_regions = 0
        
        for c,s,e in bed_regions:
            # Convert chromosome name if needed
            fa_chrom = convert_chrom_name(c, fa_chroms)
            if fa_chrom is None:
                skipped_regions += 1
                continue
                
            s_flank = max(0, s-FLANK)
            e_flank = e+FLANK
            
            # Extract sequence to check quality
            seq = get_seq_safe(fa, fa_chrom, s_flank, e_flank)
            if is_valid_sequence(seq):
                enhancers.append((fa_chrom, s_flank, e_flank))
        
        print(f"Loaded {len(enhancers)} valid enhancer regions for {sp} (skipped {skipped_regions})")
        idx = build_tree(enhancers)
    
        # Extract positive sequences with validation
        pos_sequences = []
        pos_ids = []
        for i, (c, s, e) in enumerate(enhancers):
            seq = get_seq_safe(fa, c, s, e)
            if is_valid_sequence(seq):
                pos_sequences.append(seq)
                pos_ids.append(f"{sp}:ENH:{i}")
    
        pos_df = pd.DataFrame({
            "id": pos_ids,
            "species": sp, 
            "label": 1,
            "sequence": pos_sequences,
        })
    
        neg_coords=[]
        for c,s,e in enhancers:
            neg_coords.extend(
                sample_negatives(fa, idx, e-s, 1, rng, fa_chroms))   # Each positive sample is paired with 1 negative sample
        
        # Extract negative sequences with validation
        neg_sequences = []
        neg_ids = []
        for i, (c, s, e) in enumerate(neg_coords):
            seq = get_seq_safe(fa, c, s, e)
            if is_valid_sequence(seq):
                neg_sequences.append(seq)
                neg_ids.append(f"{sp}:NEG:{i}")
        
        neg_df = pd.DataFrame({
            "id": neg_ids,
            "species": sp, 
            "label": 0,
            "sequence": neg_sequences,
        })
    
        print(f"Generated {len(pos_sequences)} positives and {len(neg_sequences)} negatives for {sp}")
        dfs.extend([pos_df, neg_df])
    
    data = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed)
    
    # Final cleanup: remove any remaining low-quality sequences
    data = data[data['sequence'].str.len() >= 100]  # Minimum sequence length
    data = data[data['sequence'].str.contains('^[ACGTN]+$', na=False)]  # Valid DNA only
    data = data[data['sequence'].str.len() > 0]  # Remove empty sequences
    
    print(f"Total samples after filtering: {len(data)}")
    
    if len(data) == 0:
        print("❌ ERROR: No valid sequences remaining after filtering!")
        return
    
    # Select only sequence and label columns as requested
    data = data[['sequence', 'label']]
    
    # Report class balance
    label_counts = data['label'].value_counts()
    print(f"Class distribution: Enhancers={label_counts.get(1, 0)}, Non-enhancers={label_counts.get(0, 0)}")
    
    n = len(data); tr = int(n * train_ratio); va = int(n * (train_ratio + val_ratio))
    train_data = data.iloc[:tr]
    val_data = data.iloc[tr:va]
    test_data = data.iloc[va:]
    
    train_data.to_csv(splits/"train.csv", index=False)
    val_data.to_csv(splits/"dev.csv", index=False)
    test_data.to_csv(splits/"test.csv", index=False)
    
    print(f"Final dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("✔ Enhancer dataset saved to", splits.resolve())
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build enhancer vs non-enhancer binary classification dataset")
    ap.add_argument("--out_dir", type=Path, default=Path("enhancer_dataset"),
                    help="Output directory for dataset (default: enhancer_dataset)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--train_ratio", type=float, default=0.8,
                    help="Fraction of data for training (default: 0.8)")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of data for validation (default: 0.1)")
    
    args = ap.parse_args()
    
    # Validate ratios
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        ap.error("train_ratio must be between 0 and 1")
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        ap.error("val_ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        ap.error("train_ratio + val_ratio must be < 1.0 to leave room for test set")
    
    build(args.out_dir, args.seed, args.train_ratio, args.val_ratio)
'''


PROCESS_PROMOTER_CODE = r'''#!/usr/bin/env python3
# Robust promoter-vs-non-promoter builder (hg38 / mm10 / danRer11)
# ---------------------------------------------------------------
# 1. Downloads plain-text BED tracks from EPDnew  (no .gz)
# 2. Downloads compressed reference genomes (.fa.gz) from UCSC
# 3. Extracts −2000..+500 bp around each TSS as positives
# 4. Samples equal-sized random negatives, no overlap with positives
# 5. Shuffles and writes train/val/test CSV (80/10/10)

import argparse, gzip, random, shutil
from pathlib import Path
from typing import List
import requests, pandas as pd
from tqdm import tqdm
from intervaltree import Interval, IntervalTree          # BED overlap logic :contentReference[oaicite:2]{index=2}
from pyfaidx import Fasta

# ---------- data sources (HTTPS only) ----------
SOURCES = {
    "hg38": {
        "bed": "https://epd.expasy.org/ftp/epdnew/H_sapiens/current/Hs_EPDnew.bed",
        "fa":  "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    },
    "mm10": {
        "bed": "https://epd.expasy.org/ftp/epdnew/M_musculus/current/Mm_EPDnew.bed",
        "fa":  "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    },
    "danRer11": {
        "bed": "https://epd.expasy.org/ftp/epdnew/zebrafish/current/Dr_EPDnew.bed",
        "fa":  "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz",
    },
}

UP, DOWN = 2000, 500                          # promoter window  :contentReference[oaicite:3]{index=3}
WIN_LEN   = UP + DOWN + 1                     # 2501 bp

# ---------- helpers ----------
def download(url: str, dest: Path, chunk: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): return
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=f"↓ {dest.name}") as bar, open(dest, "wb") as fh:
            for blk in r.iter_content(chunk):
                fh.write(blk); bar.update(len(blk))

def gunzip(src: Path, dst: Path):
    if dst.exists(): return
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)

# ---------- BED parsing ----------
def expand_promoter(line: str):
    """Return (chrom, start, end, strand) after padding missing cols."""
    parts: List[str] = line.rstrip("\n").split("\t")
    # If tab splitting doesn't work (less than 3 columns), try space splitting
    if len(parts) < 3:
        parts = line.rstrip("\n").split()
    if len(parts) < 3:
        raise ValueError(f"BED line has <3 cols: {parts!r}")
    chrom, start, end = parts[:3]
    start, end = int(start), int(end)
    strand = parts[5] if len(parts) >= 6 else "+"   # default plus strand
    if strand not in {"+", "-"}: strand = "+"
    if strand == "+":
        s, e = max(0, start - UP), start + DOWN + 1
    else:
        s, e = max(0, end - DOWN - 1), end + UP
    return chrom, s, e, strand

def load_promoters(bed: Path):
    with open(bed) as fh:
        for ln in fh:
            if ln.startswith(("track", "browser", "#")) or not ln.strip():   # skip headers :contentReference[oaicite:4]{index=4}
                continue
            yield expand_promoter(ln)

# ---------- negative sampling ----------
def make_index(regs):
    idx={}
    for c,s,e,_ in regs:
        idx.setdefault(c, IntervalTree()).add(Interval(s, e))
    return idx

def rand_neg(fa: Fasta, idx, n: int, rng: random.Random):
    chroms = list(fa.keys()); lens = {c: len(fa[c]) for c in chroms}
    out=[]
    while len(out) < n:
        c   = rng.choice(chroms)
        if lens[c] < WIN_LEN: continue
        s   = rng.randint(0, lens[c] - WIN_LEN); e = s + WIN_LEN
        if c in idx and idx[c].overlap(s, e): continue
        out.append((c, s, e))
    return out

def rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]

def get_seq(fa, c, s, e, strand):
    try:
        seq = fa[c][s:e].upper()  # Remove .seq since as_raw=True returns strings directly
        return rc(seq) if strand == "-" else seq
    except (KeyError, IndexError):
        return ""  # Return empty string if chromosome not found or coordinates invalid

def is_valid_sequence(seq: str, min_len: int = WIN_LEN, max_n_ratio: float = 0.1) -> bool:
    """Check if sequence is valid: not empty, correct length, not too many Ns"""
    if not seq or len(seq) != min_len:
        return False
    n_count = seq.count('N')
    n_ratio = n_count / len(seq)
    return n_ratio <= max_n_ratio

def filter_valid_sequences(regs, fa, strand_info=None):
    """Filter regions that produce valid sequences"""
    valid_regs = []
    for i, reg in enumerate(regs):
        if strand_info:
            c, s, e, strand = reg
            seq = get_seq(fa, c, s, e, strand)
        else:
            c, s, e = reg
            seq = get_seq(fa, c, s, e, "+")
        
        if is_valid_sequence(seq):
            valid_regs.append(reg)
    
    return valid_regs

# ---------- builder ----------
def build(out_dir: Path, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Dataset split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
    rng = random.Random(seed)
    raw   = out_dir / "raw"
    split = out_dir / "splits"; split.mkdir(parents=True, exist_ok=True)
    dfs   = []
    
    for sp, urls in SOURCES.items():
        bed = raw / f"{sp}.bed"
        fa_gz, fa_plain = raw / f"{sp}.fa.gz", raw / f"{sp}.fa"
        download(urls["bed"], bed)
        download(urls["fa"],  fa_gz); gunzip(fa_gz, fa_plain)
        fa = Fasta(fa_plain, as_raw=True, sequence_always_upper=True)
    
        pos_regs = list(load_promoters(bed))
        print(f"Loaded {len(pos_regs)} promoter regions for {sp}")
        
        # Filter valid positive regions
        pos_regs = filter_valid_sequences(pos_regs, fa, strand_info=True)
        print(f"Kept {len(pos_regs)} valid promoter regions for {sp}")
        
        idx = make_index(pos_regs)
        pos_df = pd.DataFrame({
            "id":      [f"{sp}:POS:{i}" for i in range(len(pos_regs))],
            "species": sp, "label": 1,
            "sequence":[get_seq(fa,c,s,e,strand) for c,s,e,strand in pos_regs],
        })
    
        # Generate more negative regions to account for filtering
        neg_regs = rand_neg(fa, idx, len(pos_regs) * 2, rng)  # Generate 2x to account for filtering
        
        # Filter valid negative regions
        neg_regs = filter_valid_sequences(neg_regs, fa, strand_info=False)
        print(f"Generated {len(neg_regs)} valid negative regions for {sp}")
        
        # Take only the number we need
        neg_regs = neg_regs[:len(pos_regs)]
        
        neg_df = pd.DataFrame({
            "id":      [f"{sp}:NEG:{i}" for i in range(len(neg_regs))],
            "species": sp, "label": 0,
            "sequence":[get_seq(fa,c,s,e,"+") for c,s,e in neg_regs],
        })
        dfs.extend([pos_df, neg_df])
    
    data = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed)
    
    # Final cleanup: remove any rows with empty sequences
    print(f"Total sequences before cleanup: {len(data)}")
    data = data[data['sequence'].str.len() == WIN_LEN]  # Keep only sequences of correct length
    data = data[data['sequence'].str.contains('^[ACGTN]+$')]  # Keep only valid DNA sequences
    print(f"Total sequences after cleanup: {len(data)}")
    
    # Select only sequence and label columns as requested
    data = data[['sequence', 'label']]
    
    # Report class balance
    label_counts = data['label'].value_counts()
    print(f"Class distribution: Promoters={label_counts[1]}, Non-promoters={label_counts[0]}")
    
    n = len(data); tr = int(n * train_ratio); va = int(n * (train_ratio + val_ratio))
    train_data = data.iloc[:tr]
    val_data = data.iloc[tr:va] 
    test_data = data.iloc[va:]
    
    train_data.to_csv(split/"train.csv", index=False)
    val_data.to_csv(split/"val.csv", index=False)
    test_data.to_csv(split/"test.csv", index=False)
    
    print(f"Final dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("✔ Dataset ready:", split.resolve())
    
# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build promoter vs non-promoter classification dataset")
    p.add_argument("--out_dir", type=Path, default=Path("promoter_dataset"), 
                   help="Output directory for dataset (default: promoter_dataset)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Fraction of data for training (default: 0.8)")
    p.add_argument("--val_ratio", type=float, default=0.1, 
                   help="Fraction of data for validation (default: 0.1)")
    
    args = p.parse_args()
    
    # Validate ratios
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        p.error("train_ratio must be between 0 and 1")
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        p.error("val_ratio must be between 0 and 1") 
    if args.train_ratio + args.val_ratio >= 1:
        p.error("train_ratio + val_ratio must be < 1.0 to leave room for test set")
    
    build(args.out_dir, args.seed, args.train_ratio, args.val_ratio)
'''


def main():
    parser = argparse.ArgumentParser(description="Unified runner for data processing scripts")
    parser.add_argument(
        "--type",
        choices=["normal", "emp", "enhancer", "promoter"],
        required=True,
        help="Which pipeline to run: normal(process_data), emp(process_emp), enhancer(process_enhancer), promoter(process_promoter)",
    )
    args, unknown = parser.parse_known_args()

    code_map = {
        "normal": PROCESS_DATA_CODE,
        "emp": PROCESS_EMP_CODE,
        "enhancer": PROCESS_ENHANCER_CODE,
        "promoter": PROCESS_PROMOTER_CODE,
    }

    selected = args.type
    code = code_map[selected]

    # Forward remaining args to the selected script
    sys.argv = [f"{selected}.py"] + unknown

    # Execute the selected code as if it were its own __main__
    exec_globals = {"__name__": "__main__"}
    exec(code, exec_globals, exec_globals)


if __name__ == "__main__":
    main()


