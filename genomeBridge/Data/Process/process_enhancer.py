#!/usr/bin/env python3
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
