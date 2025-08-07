"""
Simple DNA sequence generation using Generator model with single nucleotide prompts.

Example usage:
python 1.py --num 1000 --seq_len 5000 --output_prefix generated_sequences
"""
import pandas as pd
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def left_truncation(sequence, multiple=6):
    remainder = len(sequence) % multiple
    if remainder != 0:
        return sequence[remainder:]
    return sequence

def generate_sequences(
    num=1000,
    seq_len=833,
    **kwargs
):
    """Generate DNA sequences using Generator model with single nucleotide prompts"""
    
    # Use 4 different single nucleotide prompts
    prompts = ["A", "T", "C", "G"]
    sequences_per_prompt = 250

    # Load Generator model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    
    print("Loading Generator model...")
    tokenizer = AutoTokenizer.from_pretrained("GenerTeam/GENERator-eukaryote-3b-base", trust_remote_code=True, cache_dir="/projects/p32572")
    model = AutoModelForCausalLM.from_pretrained("GenerTeam/GENERator-eukaryote-3b-base", cache_dir="/projects/p32572")
    model = model.to(device)
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    config = model.config
    max_length = config.max_position_embeddings
    max_seq_len = seq_len  # Generator uses k-mers (6-character tokens)
    
    # Generate sequences
    all_sequences = []
    batch_size = 10
    
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
            
            # Process prompts with Generator-specific preprocessing
            processed_prompts = []
            for sequence in batch_prompts:
                # Apply truncation to ensure sequence is divisible by 6
                truncated_sequence = left_truncation(sequence)
                # Add BOS token
                processed_sequence = tokenizer.bos_token + truncated_sequence
                processed_prompts.append(processed_sequence)
            
            # Tokenize the batch
            tokenizer.padding_side = "left"
            inputs = tokenizer(
                processed_prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Generate sequences for the batch
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_seq_len,
                    temperature=2, 
                    top_k=-1,
                )
            
            # Decode the generated sequences
            decoded_seqs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_sequences.extend(decoded_seqs)
            
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