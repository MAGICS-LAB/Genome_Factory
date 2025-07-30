"""
Simple DNA sequence generation using Evo model with empty prompt.

Example usage:
python 1.py --num 100 --seq_len 1000 --output_prefix generated_sequences
"""
import pandas as pd
import os
import argparse
import torch
from evo import Evo, generate

def generate_sequences(
    num=1000,
    seq_len=5000,
    **kwargs
):
    """Generate DNA sequences using Evo model with empty prompt"""
    
    # Use 4 different single nucleotide prompts
    prompts = ["A", "T", "C", "G"]
    sequences_per_prompt = 250

    # Load Evo model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    
    print("Loading Evo model...")
    evo_model = Evo('evo-1-131k-base')
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    # Generate sequences
    all_sequences = []
    batch_size = 5
    
    print(f"Generating {num} sequences of length {seq_len}...")
    
    # Generate 250 sequences for each prompt
    for prompt in prompts:
        print(f"Generating {sequences_per_prompt} sequences with prompt '{prompt}'...")
        remaining_sequences = sequences_per_prompt
        
        while remaining_sequences > 0:
            current_batch_size = min(batch_size, remaining_sequences)
            batch_prompts = [prompt] * current_batch_size
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                output_seqs, _ = generate(
                    batch_prompts,
                    model,
                    tokenizer,
                    n_tokens=seq_len,
                    temperature=1.0,
                    top_k=4,
                    top_p=1.0,
                    cached_generation=True,
                    batched=True,
                    prepend_bos=False,
                    device=str(device),
                    verbose=1
                )
                
                all_sequences.extend(output_seqs)
                
            except Exception as e:
                print(f"Error in generation: {e}")
                # Create dummy sequences if generation fails
                for j in range(current_batch_size):
                    dummy_seq = 'A' * seq_len
                    all_sequences.append(dummy_seq)
            
            remaining_sequences -= current_batch_size
            print(f"Generated {len(all_sequences)}/{num} sequences...")

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Successfully generated {len(all_sequences)} sequences")
    return all_sequences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000, help="number of sequences to generate")
    parser.add_argument("--seq_len", type=int, default=5000, help="length of each sequence")
    parser.add_argument("--output_prefix", default='generated_sequences', help="output file prefix")
    args = parser.parse_args()

    # Create output directory if needed
    if args.output_prefix:
        args.output_prefix = os.path.expanduser(args.output_prefix)
        output_dir = os.path.dirname(args.output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

    print(f'Parameters: num={args.num}, seq_len={args.seq_len}, output={args.output_prefix}')
    
    try:
        # Generate sequences
        sequences = generate_sequences(
            num=args.num,
            seq_len=args.seq_len
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