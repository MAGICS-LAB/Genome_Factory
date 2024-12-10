import argparse
from genslm import GenSLM, SequenceDataset
import torch
import tqdm
import time
import csv


def seq_to_3mer(seq):
    seq = seq.replace("\n", "")
    kmers = [seq[i:i+3] for i in range(0, len(seq), 3)]
    if len(kmers[-1]) < 3:
        kmers = kmers[:-1]
    seq = " ".join(kmers)
    return seq

def decode_3mer(seq):
    seq = seq.replace(" ", "")
    return seq

def main():

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='Generate sequences using the Evo model.')

    parser.add_argument('--model-name', type=str, default='genslm_25M_patric', help='Evo model name')
    parser.add_argument('--prompt', type=str, default='ACGT', help='Prompt for generation')
    parser.add_argument('--n-samples', type=int, default=1, help='Number of sequences to sample at once')
    parser.add_argument('--n-tokens', type=int, default=2000, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature during sampling')
    parser.add_argument('--top-k', type=int, default=4, help='Top K during sampling')
    parser.add_argument('--top-p', type=float, default=1., help='Top P during sampling')
    parser.add_argument('--cached-generation', type=bool, default=True, help='Use KV caching during generation')
    parser.add_argument('--batched', type=bool, default=True, help='Use batched generation')
    parser.add_argument('--prepend-bos', type=bool, default=False, help='Prepend BOS token')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for generation')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Load model.

    model = GenSLM(args.model_name, model_cache_dir="/root/Downloads")
    model.eval()

    # Select GPU device if it is available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    with open("/root/data/cami2/marine_plant_20_unknown.tsv", "r") as f:
        lines = list(csv.reader(f, delimiter="\t"))
        dna_sequences = [line[0] for line in lines]
        labels = [line[1] for line in lines]
        num_generation_from_each_prompt = 1
        augmented_labels = []
        for label in labels:
            aug_label = [label] * num_generation_from_each_prompt
            augmented_labels.extend(aug_label)

    # Sample sequences.
    
    print('Generated sequences:')
    all_seqs = []
    prompts = dna_sequences * num_generation_from_each_prompt

    seq_len = args.n_tokens // 3 + 5
    print(seq_len)
    model_size = args.model_name.split("_")[1]
    
    with open(f"/root/MOE_DNA/ICLR/generated/from_marine_plant_20_unknown_genslm_{model_size}.txt", "a") as f:
        for i, seq in enumerate(tqdm.tqdm(dna_sequences)):
            if i == 1:
                start = time.time()
            
            seq = seq_to_3mer(seq)
            prompt = model.tokenizer.encode(seq, return_tensors="pt").to(device)
            
            tokens = model.model.generate(
                prompt,
                max_new_tokens=seq_len,  # Increase this to generate longer sequences
                min_new_tokens=seq_len,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=num_generation_from_each_prompt,  # Change the number of sequences to generate
                remove_invalid_values=True,
                use_cache=True,
                pad_token_id=model.tokenizer.encode("[PAD]")[0],
                temperature=args.temperature,
            )
        

            sequences = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            output_seqs = [decode_3mer(seq)[len(seq):] for seq in sequences]
            
            
            f.write(f"{output_seqs[0]}\t{label}\n")




if __name__ == '__main__':
    main()