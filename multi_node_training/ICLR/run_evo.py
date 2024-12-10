import argparse

import os
import csv
from evo import Evo, generate


def main():

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Generate sequences using the Evo model.')

    parser.add_argument('--model-name', type=str, default='evo-1-131k-base', help='Evo model name')
    parser.add_argument('--prompt', type=str, default='ACGT', help='Prompt for generation')
    parser.add_argument('--n-samples', type=int, default=1, help='Number of sequences to sample at once')
    parser.add_argument('--n-tokens', type=int, default=2000, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature during sampling')
    parser.add_argument('--top-k', type=int, default=4, help='Top K during sampling')
    parser.add_argument('--top-p', type=float, default=1., help='Top P during sampling')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for slicing')
    parser.add_argument('--end_idx', type=int, default=1, help='End index for slicing')
    parser.add_argument('--cached-generation', type=bool, default=True, help='Use KV caching during generation')
    parser.add_argument('--batched', type=bool, default=True, help='Use batched generation')
    parser.add_argument('--prepend-bos', type=bool, default=False, help='Prepend BOS token')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for generation')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--output_dir', type=str, default="/root/MOE_DNA/ICLR/generated/coding_non_coding", help='Output directory for generated sequences')
    parser.add_argument('--data_dir', type=str, default="/root/data/cami2/marine_plant_20_unknown.tsv", help='Data directory for prompts')

    args = parser.parse_args()
    
    # Load data.
    with open(args.data_dir, "r") as f:
        delimiter = "," if args.data_dir.endswith(".csv") else "\t"
        lines = list(csv.reader(f, delimiter=delimiter))[args.start_idx:args.end_idx]
        dna_sequences = [line[0][-2000:] for line in lines]
        labels = [line[1:] for line in lines]
        num_generation_from_each_prompt = 1
        augmented_labels = []
        for label in labels:
            aug_label = [label] * num_generation_from_each_prompt
            augmented_labels.extend(aug_label)
    
    print(f"Loaded {len(dna_sequences)} DNA sequences from {args.data_dir}")

    # Load model.

    evo_model = Evo(args.model_name)
    model, tokenizer = evo_model.model, evo_model.tokenizer

    model.to(args.device)
    model.eval()
    

    dataset_name = args.data_dir.split("/")[-1].split(".")[0]
    
    # Sample sequences.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print('Generated sequences:')
    all_seqs = []
    prompts = dna_sequences * num_generation_from_each_prompt
    for seq, label in zip(prompts, augmented_labels):
        output_seqs, output_scores = generate(
            [seq],
            model,
            tokenizer,
            n_tokens=args.n_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            cached_generation=args.cached_generation,
            batched=args.batched,
            prepend_bos=args.prepend_bos,
            device=args.device,
            verbose=args.verbose,
        )
                
        with open(os.path.join(args.output_dir, f"evo_{dataset_name}_{args.temperature}_{args.start_idx}_{args.end_idx}.txt"), "a") as f:
            combined = output_seqs + label
            delimiter = "\t"
            f.write(f"{combined[0]}{delimiter}{delimiter.join(combined[1:])}\n")




if __name__ == '__main__':
    main()