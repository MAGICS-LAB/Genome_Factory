#!/usr/bin/env python3
"""
Build motif-presence vs non-motif dataset using FIMO + JASPAR 2024 vertebrates

• Downloads UCSC reference genome FASTA
• Downloads JASPAR 2024 vertebrate motifs
• Runs FIMO to find motif hits with flanking regions
• Samples equal-length random regions without motifs (negatives)
• Shuffles and splits into train/val/test CSV with configurable ratios
"""

import argparse, gzip, random, shutil, subprocess, tempfile, os
from pathlib import Path
import pandas as pd, requests
from tqdm import tqdm
from intervaltree import Interval, IntervalTree
from pyfaidx import Fasta
from jaspar import JASPAR5  # pyjaspar

FLANK = 50                    # bp each side of motif

UCSC = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
}

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

def fetch_jaspar_meme(out: Path):
    if out.exists(): return
    print("Downloading JASPAR 2024 vertebrates motifs …")
    jdb = JASPAR5()
    motifs = jdb.fetch_motifs(collection="CORE", tax_group="vertebrates")
    with open(out, "w") as fh:
        for m in motifs:
            fh.write(m.to_meme())

def build_bg(fa: Path, bg: Path):
    if bg.exists(): return
    subprocess.check_call(["fasta-get-markov", str(fa), str(bg)])

def run_fimo(meme: Path, fa: Path, bg: Path, out_tsv: Path, p=1e-4):
    if out_tsv.exists(): return
    tmp = tempfile.mkdtemp()
    cmd = [
        "fimo", "--bgfile", str(bg), "--thresh", str(p),
        "--max-strand", "--text", str(meme), str(fa)
    ]
    with open(out_tsv, "w") as fh:
        subprocess.check_call(cmd, stdout=fh)
    shutil.rmtree(tmp, ignore_errors=True)

def to_tree(regs):
    idx = {}
    for c,s,e in regs:
        idx.setdefault(c, IntervalTree()).add(Interval(s, e))
    return idx

def rc(seq): return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]

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

def sample_neg(fa, idx, length, n, rng, fa_chroms):
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
            idx[c].add(Interval(s, e))
    
    return out

def build(species, out_dir, seed=42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"Dataset split ratios: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
    
    rng = random.Random(seed)
    raw = out_dir/"raw"; raw.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing {species.upper()} ===")
    fa_gz = raw/f"{species}.fa.gz"; fa = raw/f"{species}.fa"
    download(UCSC[species], fa_gz); gunzip(fa_gz, fa)
    motif_file = raw/"jaspar2024_vert.meme"
    fetch_jaspar_meme(motif_file)

    bg = raw/f"{species}.bg"
    build_bg(fa, bg)
    hits_tsv = raw/f"{species}_fimo.tsv"
    run_fimo(motif_file, fa, bg, hits_tsv)

    # Load FASTA and get available chromosomes
    fasta = Fasta(fa, as_raw=True, sequence_always_upper=True)
    fa_chroms = set(fasta.keys())

    # Parse FIMO hits with quality control
    hits = []
    skipped_regions = 0
    with open(hits_tsv) as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip(): continue
            seqid, motif, start, stop, *_strand, p = ln.split()[:7]
            chrom = seqid.split(":")[0]
            
            # Convert chromosome name if needed
            fa_chrom = convert_chrom_name(chrom, fa_chroms)
            if fa_chrom is None:
                skipped_regions += 1
                continue
            
            s = max(0, int(start)-1-FLANK)
            e = int(stop)+FLANK      # FIMO stop inclusive
            
            # Extract sequence to check quality
            seq = get_seq_safe(fasta, fa_chrom, s, e)
            if is_valid_sequence(seq):
                hits.append((fa_chrom, s, e))
    
    print(f"Loaded {len(hits)} valid motif regions (skipped {skipped_regions})")
    idx = to_tree(hits)

    # Extract positive sequences with validation
    pos_sequences = []
    pos_ids = []
    for i, (c, s, e) in enumerate(hits):
        seq = get_seq_safe(fasta, c, s, e)
        if is_valid_sequence(seq):
            pos_sequences.append(seq)
            pos_ids.append(f"{species}:MOTIF:{i}")

    pos_df = pd.DataFrame({
        "id": pos_ids,
        "species": species,
        "label": 1,
        "sequence": pos_sequences,
    })

    # Generate negative samples with validation
    neg_coords = []
    for c, s, e in hits:
        neg_coords.extend(sample_neg(fasta, idx, e-s, 1, rng, fa_chroms))
    
    # Extract negative sequences with validation
    neg_sequences = []
    neg_ids = []
    for i, (c, s, e) in enumerate(neg_coords):
        seq = get_seq_safe(fasta, c, s, e)
        if is_valid_sequence(seq):
            neg_sequences.append(seq)
            neg_ids.append(f"{species}:NEG:{i}")
    
    neg_df = pd.DataFrame({
        "id": neg_ids,
        "species": species,
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
    print(f"Class distribution: Motifs={label_counts.get(1, 0)}, Non-motifs={label_counts.get(0, 0)}")
    
    # Split data
    n = len(data); tr = int(n * train_ratio); va = int(n * (train_ratio + val_ratio))
    train_data = data.iloc[:tr]
    val_data = data.iloc[tr:va]
    test_data = data.iloc[va:]
    
    split = out_dir/"splits"; split.mkdir(exist_ok=True)
    train_data.to_csv(split/"train.csv", index=False)
    val_data.to_csv(split/"val.csv", index=False)
    test_data.to_csv(split/"test.csv", index=False)
    
    print(f"Final dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    print("✔ Motif dataset saved to", split.resolve())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build motif-presence vs non-motif binary classification dataset using FIMO + JASPAR")
    ap.add_argument("--species", choices=["hg38","mm10"], required=True,
                    help="Species genome to process (hg38 or mm10)")
    ap.add_argument("--out_dir", type=Path, default=Path("motif_dataset"),
                    help="Output directory for dataset (default: motif_dataset)")
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
    
    build(args.species, args.out_dir, args.seed, args.train_ratio, args.val_ratio)
