#!/usr/bin/env python3
"""
Build TF-binding vs non-binding dataset from ReMap 2022 ChIP-seq Atlas (hg38/mm10)

• Downloads ReMap 2022 BED files with TF binding peaks
• Downloads UCSC reference genome FASTA  
• Extracts binding regions for specified TFs (e.g., CTCF, POLR2A)
• Samples equal-length random regions without TF binding (negatives)
• Shuffles and splits into train/val/test CSV with configurable ratios
"""

import argparse, gzip, random, shutil, re
from pathlib import Path
import requests, pandas as pd
from tqdm import tqdm
from intervaltree import Interval, IntervalTree
from pyfaidx import Fasta

SOURCES = {
    "hg38": {
        "bed": "https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/"
               "remap2022_all_macs2_hg38_v1_0.bed.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    },
    "mm10": {
        "bed": "https://remap.univ-amu.fr/storage/remap2022/mm10/MACS2/"
               "remap2022_all_macs2_mm10_v1_0.bed.gz",
        "fa" : "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
    },
}

# ---------- utilities ----------
def download(url: str, dst: Path, chunk: int = 1 << 20):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    with requests.get(url, stream=True, timeout=60) as r:
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

def build_tree(regs):
    idx = {}
    for c, s, e in regs:
        idx.setdefault(c, IntervalTree()).add(Interval(s, e))
    return idx

def rc(seq): return seq.translate(str.maketrans("ACGTNacgtn","TGCANtgcan"))[::-1]

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

def rand_window(fa, idx, L, rng, fa_chroms):
    chroms = list(fa_chroms); sizes = {c: len(fa[c]) for c in chroms}
    attempts = 0
    max_attempts = 100  # Prevent infinite loops
    
    while attempts < max_attempts:
        attempts += 1
        c = rng.choice(chroms)
        if sizes[c] < L: continue
        s = rng.randint(0, sizes[c] - L); e = s + L
        if idx.get(c) and idx[c].overlap(s, e): continue
        
        # Check if sequence is valid
        seq = get_seq_safe(fa, c, s, e)
        if is_valid_sequence(seq, min_len=L//2):  # Allow shorter sequences for negatives
            idx.setdefault(c, IntervalTree()).add(Interval(s, e))
            return c, s, e
    
    # Fallback: return a random window without quality check if max attempts reached
    c = rng.choice(chroms)
    if sizes[c] >= L:
        s = rng.randint(0, sizes[c] - L)
        e = s + L
        idx.setdefault(c, IntervalTree()).add(Interval(s, e))
        return c, s, e
    
    return None, None, None  # Failed to generate

# ---------- main ----------
def build(species, tf_list, out_dir, seed=42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Dataset split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
    rng = random.Random(seed)
    raw = out_dir/"raw"; split = out_dir/"splits"; split.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {species.upper()} ===")
    bed_gz = raw/f"{species}.bed.gz"
    fa_gz = raw/f"{species}.fa.gz"; fa_plain = raw/f"{species}.fa"
    download(SOURCES[species]["bed"], bed_gz)
    download(SOURCES[species]["fa"], fa_gz); gunzip(fa_gz, fa_plain)
    fa = Fasta(fa_plain, as_raw=True, sequence_always_upper=True)
    
    # Get available chromosomes
    fa_chroms = set(fa.keys())

    # Read & filter BED with quality control
    tf_pattern = re.compile("|".join(map(re.escape, tf_list)), flags=re.I)
    tf_regions = []
    skipped_regions = 0
    total_regions = 0
    
    with gzip.open(bed_gz, "rt") as fh:
        for ln in fh:
            if not ln.strip() or ln[0] in "#t": continue
            total_regions += 1
            chrom, s, e, name, *_ = ln.rstrip("\n").split("\t")[:5]
            
            if not tf_pattern.search(name): continue
            
            # Convert chromosome name if needed
            fa_chrom = convert_chrom_name(chrom, fa_chroms)
            if fa_chrom is None:
                skipped_regions += 1
                continue
            
            s, e = int(s), int(e)
            # Extract sequence to check quality
            seq = get_seq_safe(fa, fa_chrom, s, e)
            if is_valid_sequence(seq):
                tf_symbol = name.split(".")[1] if "." in name else name  # Extract TF symbol
                tf_regions.append((fa_chrom, s, e, tf_symbol))

    print(f"Loaded {len(tf_regions)} valid TF binding regions for {tf_list} (skipped {skipped_regions})")
    idx = build_tree([(c, s, e) for c, s, e, _ in tf_regions])

    # Extract positive sequences with validation
    pos_sequences = []
    pos_ids = []
    for i, (c, s, e, tf) in enumerate(tf_regions):
        seq = get_seq_safe(fa, c, s, e)
        if is_valid_sequence(seq):
            pos_sequences.append(seq)
            pos_ids.append(f"{species}:TF:{i}")

    pos_df = pd.DataFrame({
        "id": pos_ids,
        "species": species,
        "tf": [tf for c, s, e, tf in tf_regions[:len(pos_sequences)]],
        "label": 1,
        "sequence": pos_sequences,
    })

    # Generate negative samples with validation
    neg_coords = []
    for c, s, e, _ in tf_regions:
        result = rand_window(fa, idx, e-s, rng, fa_chroms)
        if result[0] is not None:  # Valid result
            neg_coords.append(result)
    
    # Extract negative sequences with validation  
    neg_sequences = []
    neg_ids = []
    for i, (c, s, e) in enumerate(neg_coords):
        if c is None: continue  # Skip failed generations
        seq = get_seq_safe(fa, c, s, e)
        if is_valid_sequence(seq):
            neg_sequences.append(seq)
            neg_ids.append(f"{species}:NEG:{i}")

    neg_df = pd.DataFrame({
        "id": neg_ids,
        "species": species,
        "tf": "NA",
        "label": 0,
        "sequence": neg_sequences,
    })
    
    print(f"Generated {len(pos_sequences)} positives and {len(neg_sequences)} negatives")

    # Comprehensive final cleanup
    data = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=seed)
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
    print(f"Class distribution: TF binding={label_counts.get(1, 0)}, Non-binding={label_counts.get(0, 0)}")
    
    # Split data
    n = len(data); tr = int(n * train_ratio); va = int(n * (train_ratio + val_ratio))
    train_data = data.iloc[:tr]
    val_data = data.iloc[tr:va]
    test_data = data.iloc[va:]
    
    train_data.to_csv(split/"train.csv", index=False)
    val_data.to_csv(split/"val.csv", index=False)
    test_data.to_csv(split/"test.csv", index=False)
    
    print(f"Final dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("✔ TF binding dataset saved to", split.resolve())

if __name__=="__main__":
    ap = argparse.ArgumentParser(description="Build TF-binding vs non-binding binary classification dataset from ReMap 2022")
    ap.add_argument("--species", choices=["hg38","mm10"], required=True,
                    help="Species genome to process (hg38 or mm10)")
    ap.add_argument("--tfs", required=True,
                    help="Comma-separated TF symbols (e.g., CTCF,POLR2A,MYC)")
    ap.add_argument("--out_dir", type=Path, default=Path("tf_binding_dataset"),
                    help="Output directory for dataset (default: tf_binding_dataset)")
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
    
    tf_list = [t.strip() for t in args.tfs.split(",")]
    if not tf_list or any(not tf for tf in tf_list):
        ap.error("TF list cannot be empty or contain empty TF names")
    
    build(args.species, tf_list, args.out_dir, args.seed, args.train_ratio, args.val_ratio)
