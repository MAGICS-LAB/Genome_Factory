"""
Given a partial gene sequence, generate sequences that are likely to fold into a given structure.
Check the structure of the generated sequences using FoldMason (external program), which can be installed from [here](https://github.com/steineggerlab/foldmason?tab=readme-ov-file#installation)
The script takes as input a gene id, start and end positions, and the start and end positions of the prompt.

# GMP synthetase
python autocomplete_structure_Evo2.py \
    --gen_id NZ_JAYXHC010000003.1 \
    --start 157 \
    --end 1698 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 600 \
    --structure_start 150 \
    --structure_end 500 \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --num 100 \
    --min_seq_len 1000 \
    --max_seq_len 1200 \
    --foldmason_path /home/zpt6685/foldmason/bin/foldmason \
    --output_prefix outputs_Evo2/gmp

# TRAP-like, also explore mutations in the prompt
python autocomplete_structure.py \
    --gen_id OY729418.1 \
    --start 1675256 \
    --end 1676176 \
    --strand -1 \
    --prompt_start 0 \
    --prompt_end 450 \
    --mutate_prompt 1 \
    --structure_start 0 \
    --structure_end 341 \
    --model_dir pGenomeOcean/GenomeOcean-4B \
    --num 200 \
    --min_seq_len 250 \
    --max_seq_len 300 \
    --foldmason_path ~/bin/foldmason \
    --output_prefix outputs/trapl_wt_mutations

"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
import requests
import subprocess
import torch
from evo2 import Evo2

from genomeocean.dnautils import get_nuc_seq_by_id, introduce_mutations
from genomeocean.dnautils import fasta2pdb_api, reverse_complement, LDDT_scoring

from Bio.Seq import translate
from Bio.Seq import Seq

def get_largest_orf(seq):
    seq = Seq(seq)

    # Translate in forward three frames
    orfs = []
    for frame in range(3):
        # translate every substring of the sequence starting with ATG or GTG
        for i in range(frame, len(seq), 3):
            if seq[i:i+3] in ['ATG', 'GTG']:
                orfs.append(str(seq[i:].translate(to_stop=True)))
        # also include the first one that does not start with ATG or GTG
        orfs.append(str(seq[frame:].translate(to_stop=True)))
    # remove extra aa before 'M'
    # orfs = [orf[orf.find('M'):] for orf in orfs]
    return max(orfs, key=len)

def chk_gen_structure(
    gen_id, 
    start, 
    end, 
    prompt_start=0, 
    prompt_end=0,
    mutate_prompt=False,
    strand=1,
    backward=False, # whether to generate sequences in the reverse direction
    ref_pdb='',
    structure_start=0,
    structure_end=0,
    model_dir='',
    foldmason_path='',
    **kwargs,
):
    gene = get_nuc_seq_by_id(gen_id, start=start, end=end)
    if gene is None:
        print(f'Failed to retrieve gene sequence {gen_id} from {start} to {end}')
        sys.exit(1)
    if strand == -1:
        gene=reverse_complement(gene)
    if ref_pdb == '':
        ref_pdb = 'ref_tmp.pdb'
        if os.path.exists(ref_pdb):
            os.remove(ref_pdb)
        fasta2pdb_api(translate(gene, to_stop=True)[structure_start:structure_end], ref_pdb)
    if backward: # start from the end of the gene
        gene = reverse_complement(gene)  

    prompts = [gene[prompt_start:prompt_end]] 
    if mutate_prompt:
        orf_prompt = gene[prompt_start:prompt_end]
        for mutation_rate in range(10, 50, 10):
            mutated = introduce_mutations(orf_prompt, mutation_percentage=mutation_rate, mutation_type='synonymous')
            prompts.append(mutated)
        for mutation_rate in range(10, 50, 10):
            mutated = introduce_mutations(orf_prompt, mutation_percentage=mutation_rate, mutation_type='nonsynonymous')
            prompts.append(mutated)  

    # generate sequences
    pd.DataFrame(prompts).to_csv('tmp_prompts.csv', sep='\t', header=None, index=False)
    
    if 'num' in kwargs:
        num=kwargs['num']
    else:
        num=200
    if 'min_seq_len' in kwargs:
        min_seq_len = kwargs['min_seq_len']
    else:
        min_seq_len = 250
    if 'max_seq_len' in kwargs:
        max_seq_len = kwargs['max_seq_len']
    else:
        max_seq_len = 300

    # Load Evo2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    print("Loading Evo2 model...")
    evo2_model = Evo2('evo2_7b')
    print("Model loaded successfully")
    
    # Read prompts from CSV file
    prompt_df = pd.read_csv('tmp_prompts.csv', sep='\t', header=None)
    csv_prompts = prompt_df[0].tolist()
    
    # Generate sequences - num sequences for each prompt using Evo2 generation
    all_sequences = []
    batch_size = 5  # Set batch size for Evo2 inference
    
    for prompt in csv_prompts:
        # Create batches of the same prompt for parallel generation
        remaining_sequences = num
        
        while remaining_sequences > 0:
            current_batch_size = min(batch_size, remaining_sequences)
            
            # Create batch of identical prompts
            batch_prompts = [prompt] * current_batch_size
            
            # Clear GPU cache periodically to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate sequences using Evo2's generate method
            try:
                output = evo2_model.generate(
                    prompt_seqs=batch_prompts,
                    n_tokens=max_seq_len,
                    temperature=1.0,
                    top_k=4
                )
                
                # Process each sequence in the batch
                for i, (output_seq, original_seq) in enumerate(zip(output.sequences, batch_prompts)):
                    # Remove the original sequence to get only generated part
                    if output_seq.startswith(original_seq):
                        generated_part = output_seq[len(original_seq):]
                    else:
                        # Fallback: try to find the original sequence
                        if original_seq in output_seq:
                            start_idx = output_seq.find(original_seq) + len(original_seq)
                            generated_part = output_seq[start_idx:]
                        else:
                            # Last resort: assume it's all generated
                            generated_part = output_seq
                    
                    # Clean up any spaces or special characters
                    generated_part = ''.join([c for c in generated_part if c in 'ATCG'])
                    
                    # Ensure generated part is within length constraints
                    if len(generated_part) > max_seq_len:
                        generated_part = generated_part[:max_seq_len]
                    elif len(generated_part) < min_seq_len:
                        # Pad with 'A' to reach minimum length
                        generated_part = generated_part + 'A' * (min_seq_len - len(generated_part))
                    
                    # Combine prompt with generated part
                    full_sequence = original_seq + generated_part
                    all_sequences.append(full_sequence)
                
            except Exception as e:
                print(f"Error in Evo2 generation: {e}")
                # Create dummy sequences for this batch to continue
                for j in range(current_batch_size):
                    dummy_generated = 'A' * max_seq_len
                    full_sequence = prompt + dummy_generated
                    all_sequences.append(full_sequence)
            
            remaining_sequences -= current_batch_size
            print(f"Generated {len(all_sequences)}/{num * len(csv_prompts)} sequences...")
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create DataFrame with generated sequences
    g_seqs = pd.DataFrame({'seq': all_sequences})
    
    print(f'total {g_seqs.shape[0]} sequences were generated.')  
    os.remove('tmp_prompts.csv')

    if backward:
        g_seqs['seq'] = g_seqs['seq'].apply(lambda x: reverse_complement(x))
    # use biopython to find the longest ORF:
    g_seqs['protein'] = g_seqs['seq'].apply(lambda x: get_largest_orf(x))  
    g_seqs['orf_len'] = g_seqs['protein'].apply(lambda x: 3*len(x))
    g_seqs['length'] = g_seqs['seq'].apply(lambda x: len(x))
    g_seqs = g_seqs[g_seqs['orf_len']>=len(gene)-100].copy()
    print(f'total {g_seqs.shape[0]} sequences has longer ORFs than the original-100.')
    # save the sequences to a file
    g_seqs.to_csv('tmp_generated.csv')
    g_seqs['lddt_score'] = g_seqs['protein'].apply(lambda x: LDDT_scoring(x[structure_start:structure_end], ref_pdb, foldmason_path=foldmason_path))
    os.remove('ref_tmp.pdb')
    return g_seqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_id", help="Gene id")
    parser.add_argument("--start", type=int, help="start position")
    parser.add_argument("--end", type=int, help="end position")
    parser.add_argument("--prompt_start", type=int, default=0, help="start position of the prompt")
    parser.add_argument("--prompt_end", type=int, default=0, help="end position of the prompt")
    parser.add_argument("--mutate_prompt", type=int, default=0, help="mutate the prompt by introducing 10-40 percent synonumous and nonsynumous mutations")
    parser.add_argument("--strand", type=int, default=1, help="strand")
    parser.add_argument("--direction", type=int, default=1, help="set to -1 to generate sequences in the reverse direction")
    parser.add_argument("--ref_pdb", default='', help="reference pdb file")
    parser.add_argument("--structure_start", type=int, default=0, help="start position of the structure")
    parser.add_argument("--structure_end", type=int, default=0, help="end position of the structure")
    parser.add_argument("--model_dir", default='', help="model directory")
    parser.add_argument("--num", type=int, default=200, help="number of sequences to generate")
    parser.add_argument("--min_seq_len", type=int, default=250, help="minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=300, help="maximum sequence length")
    parser.add_argument("--foldmason_path", default='', help="foldmason path")
    parser.add_argument("--output_prefix", default='generated', help="output prefix")
    args = parser.parse_args()
    mutate_prompt = True if args.mutate_prompt == 1 else False

    if args.output_prefix:
        # Expand user home directory (~) if present
        args.output_prefix = os.path.expanduser(args.output_prefix)
        output_dir = os.path.dirname(args.output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

    backward = False
    if args.direction == -1:
        backward = True
        print('Generating sequences in the reverse direction.')
    # max length of the structure prediction is limited to 400
    if args.structure_end - args.structure_start > 400:
        args.structure_end = args.structure_start + 400
        print('The length of the structure prediction is limited to 400.')
        print(f'Structure_end was set to: {args.structure_end}')
    # print out the arguments to standard output
    print(f'Parameters: {args}')
    generated = chk_gen_structure(
        gen_id=args.gen_id, 
        start=args.start, 
        end=args.end, 
        prompt_start=args.prompt_start, 
        prompt_end=args.prompt_end,
        mutate_prompt=mutate_prompt,
        strand=args.strand,
        backward=backward,
        ref_pdb=args.ref_pdb,
        structure_start=args.structure_start,
        structure_end=args.structure_end,
        model_dir=args.model_dir,
        num=args.num,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        foldmason_path=args.foldmason_path
    )
    # save the results to a file
    generated.to_csv(args.output_prefix + '.csv', sep='\t', index=False)

if __name__ == '__main__':
    main()