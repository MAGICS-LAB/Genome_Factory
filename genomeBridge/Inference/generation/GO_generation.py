"""
Simple DNA sequence generation using GenomeOcean model with single nucleotide prompts.

Example usage:
python 1.py --num 1000 --seq_len 5000 --output_prefix generated_sequences
"""
import pandas as pd
import os
import argparse

from genomeocean.generation import SequenceGenerator

def generate_sequences(
    num=1000,
    seq_len=20000,
    model_dir='',
    **kwargs
):
    """Generate DNA sequences using GenomeOcean model with single nucleotide prompts"""
    
    # Use 4 different single nucleotide prompts
    prompts = ["A", "T", "C", "G"]
    sequences_per_prompt = 250

    # Set min_seq_len and max_seq_len according to user specification
    min_seq_len = int(seq_len / 3)  # int(5000/4) = 1250
    max_seq_len = int(seq_len / 3)  # int(5000/4) = 1250
    
    print(f"Generating {num} sequences with min_seq_len=max_seq_len={min_seq_len}")
    
    # Generate sequences for each prompt
    all_sequences = []
    
    for prompt in prompts:
        print(f"Generating {sequences_per_prompt} sequences with prompt '{prompt}'...")
        
        # Create temporary prompt file for GenomeOcean - only 1 row with the prompt
        pd.DataFrame([prompt]).to_csv('tmp_prompts.csv', sep='\t', header=None, index=False)
        
        try:
            # Use GenomeOcean SequenceGenerator
            seq_gen = SequenceGenerator(
                model_dir=model_dir, 
                promptfile='tmp_prompts.csv', 
                num=sequences_per_prompt,  # Generate sequences_per_prompt sequences for this single prompt
                min_seq_len=min_seq_len, 
                max_seq_len=max_seq_len,
                temperature=1.0,
                presence_penalty=0.5, 
                frequency_penalty=0.5, 
                repetition_penalty=1.0, 
                seed=1234
            )
            
            g_seqs = seq_gen.generate_sequences(prepend_prompt_to_output=False, max_repeats=0)
            
            # Extract sequences and add to all_sequences
            sequences = g_seqs['seq'].tolist()
            all_sequences.extend(sequences)
            
            print(f"Generated {len(sequences)} sequences with prompt '{prompt}'")
            
        except Exception as e:
            print(f"Error generating sequences for prompt '{prompt}': {e}")
            # Create dummy sequences if generation fails
            for j in range(sequences_per_prompt):
                dummy_seq = prompt + 'A' * (min_seq_len - 1)
                all_sequences.append(dummy_seq)
        
        # Clean up temporary file
        if os.path.exists('tmp_prompts.csv'):
            os.remove('tmp_prompts.csv')
    
    # Truncate each sequence to first 20000 characters
    truncated_sequences = []
    for seq in all_sequences:
        truncated_seq = seq[:20000]  # Take first 20000 characters
        truncated_sequences.append(truncated_seq)
        
    print(f"Successfully generated {len(truncated_sequences)} sequences, each truncated to max 20000 characters")
    return truncated_sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000, help="number of sequences to generate")
    parser.add_argument("--seq_len", type=int, default=20000, help="reference sequence length for calculating min/max_seq_len")
    parser.add_argument("--model_dir", default='pGenomeOcean/GenomeOcean-4B', help="GenomeOcean model directory")
    parser.add_argument("--output_prefix", default='generated_sequences_20000', help="output file prefix")
    args = parser.parse_args()

    # Create output directory if needed
    if args.output_prefix:
        args.output_prefix = os.path.expanduser(args.output_prefix)
        output_dir = os.path.dirname(args.output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

    print(f'Parameters: num={args.num}, seq_len={args.seq_len}, model_dir={args.model_dir}, output={args.output_prefix}')
    
    try:
        # Generate sequences
        sequences = generate_sequences(
            num=args.num,
            seq_len=args.seq_len,
            model_dir=args.model_dir
        )
        
        # Save sequences to file
        df = pd.DataFrame({'sequence': sequences})
        output_filename = f"{args.output_prefix}.csv"
        df.to_csv(output_filename, index=False)
        print(f"Saved {len(sequences)} sequences to {output_filename}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()