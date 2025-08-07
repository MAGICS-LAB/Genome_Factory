#!/usr/bin/env python3
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
