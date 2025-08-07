#!/usr/bin/env python3
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


