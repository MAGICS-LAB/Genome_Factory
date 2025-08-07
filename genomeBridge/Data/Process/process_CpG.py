#!/usr/bin/env python3
"""
Build CpG-island vs non-CpG dataset (hg38 & mm10)

• Downloads UCSC cpgIslandExt.txt.gz tables and reference FASTA
• Extracts island sequences (positives)
• Samples equal-length random windows that do not overlap any island (negatives)
• Shuffles and splits into train/val/test CSV with configurable ratios
"""

import argparse, gzip, random, shutil
from pathlib import Path
from typing import List, Tuple
import requests, pandas as pd
from tqdm import tqdm
from intervaltree import Interval, IntervalTree
from pyfaidx import Fasta

# ---------- UCSC download URLs ----------
SOURCES = {
    "hg38": {
        "cpg": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    },
    "mm10": {
        "cpg": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/cpgIslandExt.txt.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    },
}

# ---------- helpers ----------
def download(url: str, dst: Path, chunk: int = 1 << 20):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=f"↓ {dst.name}") as bar, open(dst, "wb") as fh:
            for blk in r.iter_content(chunk):
                fh.write(blk); bar.update(len(blk))

def gunzip(src: Path, dst: Path):
    if dst.exists(): return
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)

# ---------- load CpG islands ----------
def load_islands(cpg_gz: Path):
    with gzip.open(cpg_gz, "rt") as fh:
        for ln in fh:
            chrom, start, end, *_ = ln.rstrip("\n").split("\t")[:3+1]
            yield chrom, int(start), int(end)       # 0-based half-open

def build_tree(regs: List[Tuple[str,int,int]]):
    idx = {}
    for c, s, e in regs:
        idx.setdefault(c, IntervalTree()).add(Interval(s, e))
    return idx

def rc(seq: str) -> str:    # reverse-complement (if ever needed)
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

def is_valid_sequence(seq: str, min_len: int = 50, max_n_ratio: float = 0.1) -> bool:
    """Check if sequence is valid: not empty, reasonable length, not too many Ns"""
    if not seq or len(seq) < min_len:
        return False
    n_count = seq.count('N')
    n_ratio = n_count / len(seq)
    return n_ratio <= max_n_ratio

def random_windows(fa: Fasta, idx, length: int, n: int, rng, fa_chroms):
    chroms = list(fa_chroms); sizes = {c: len(fa[c]) for c in chroms}
    out = []
    attempts = 0
    max_attempts = n * 10  # Avoid infinite loops
    
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        c = rng.choice(chroms)
        if sizes[c] < length: continue
        s = rng.randint(0, sizes[c] - length); e = s + length
        if c in idx and idx[c].overlap(s, e): continue
        
        # Check if sequence is valid
        seq = get_seq_safe(fa, c, s, e)
        if is_valid_sequence(seq, min_len=length//2):  # Allow shorter sequences for negatives
            out.append((c, s, e))
            idx[c].add(Interval(s, e))      # avoid neg/neg overlap
    
    return out

# ---------- main ----------
def build(out_dir: Path, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Dataset split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
    rng = random.Random(seed)
    raw  = out_dir / "raw"
    out  = out_dir / "splits"; out.mkdir(parents=True, exist_ok=True)
    frames = []

    for sp, urls in SOURCES.items():
        print(f"\n=== Processing {sp.upper()} ===")
        cpg_gz = raw / f"{sp}.cpg.gz"
        fa_gz  = raw / f"{sp}.fa.gz"
        fa_plain = raw / f"{sp}.fa"

        download(urls["cpg"], cpg_gz)
        download(urls["fa"],  fa_gz); gunzip(fa_gz, fa_plain)
        fa = Fasta(fa_plain, as_raw=True, sequence_always_upper=True)
        
        # Get available chromosomes
        fa_chroms = set(fa.keys())

        # Load and validate CpG islands
        cpg_regions = list(load_islands(cpg_gz))
        print(f"Loaded {len(cpg_regions)} CpG island regions")
        
        # Filter valid CpG islands with chromosome conversion
        valid_cpg_regions = []
        skipped_regions = 0
        
        for c, s, e in cpg_regions:
            # Convert chromosome name if needed
            fa_chrom = convert_chrom_name(c, fa_chroms)
            if fa_chrom is None:
                skipped_regions += 1
                continue
                
            # Extract sequence to check quality
            seq = get_seq_safe(fa, fa_chrom, s, e)
            if is_valid_sequence(seq):
                valid_cpg_regions.append((fa_chrom, s, e))
        
        print(f"Kept {len(valid_cpg_regions)} valid CpG regions (skipped {skipped_regions})")
        pos_regs = valid_cpg_regions
        idx = build_tree(pos_regs)

        # Extract positive sequences with validation
        pos_sequences = []
        pos_ids = []
        for i, (c, s, e) in enumerate(pos_regs):
            seq = get_seq_safe(fa, c, s, e)
            if is_valid_sequence(seq):
                pos_sequences.append(seq)
                pos_ids.append(f"{sp}:CPG:{i}")

        pos_df = pd.DataFrame({
            "id": pos_ids,
            "species": sp,
            "label": 1,
            "sequence": pos_sequences,
        })

        neg_regs = []
        for c,s,e in pos_regs:
            neg_regs.extend(random_windows(fa, idx, e-s, 1, rng, fa_chroms))
        
        # Extract negative sequences with validation
        neg_sequences = []
        neg_ids = []
        for i, (c, s, e) in enumerate(neg_regs):
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
        frames.extend([pos_df, neg_df])

    # Comprehensive final cleanup
    data = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    initial_count = len(data)
    data = data[data['sequence'].str.len() >= 50]  # Minimum sequence length
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
        print("❌ ERROR: No valid sequences remaining after filtering!")
        return
    
    # Select only sequence and label columns as requested
    data = data[['sequence', 'label']]
    
    # Report class balance
    label_counts = data['label'].value_counts()
    print(f"Class distribution: CpG islands={label_counts.get(1, 0)}, Non-CpG={label_counts.get(0, 0)}")
    
    # Split data
    n = len(data); tr = int(n * train_ratio); va = int(n * (train_ratio + val_ratio))
    train_data = data.iloc[:tr]
    val_data = data.iloc[tr:va]
    test_data = data.iloc[va:]
    
    train_data.to_csv(out/"train.csv", index=False)
    val_data.to_csv(out/"val.csv", index=False)
    test_data.to_csv(out/"test.csv", index=False)
    
    print(f"Final dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("✔ CpG-island dataset saved to", out.resolve())

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build CpG-island vs non-CpG binary classification dataset")
    p.add_argument("--out_dir", type=Path, default=Path("cpg_dataset"),
                   help="Output directory for dataset (default: cpg_dataset)")
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
