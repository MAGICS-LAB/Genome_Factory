import os
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import argparse
from sklearn.preprocessing import normalize

# Calculate the averaged embedding with different resolution
def ave_reso_step(model_output, attention_mask, resolution=10):
    batch_size, feature_dim = model_output.shape[0], model_output.shape[2]
    model_output = model_output.view(batch_size, -1, resolution, feature_dim)
    attention_mask = attention_mask.view(batch_size, -1, resolution, 1)
    
    weighted_sum = torch.sum(model_output * attention_mask, dim=2)
    weight_sum = torch.sum(attention_mask, dim=2)
    weighted_average = weighted_sum / (weight_sum + 1e-6)
    
    return weighted_average

# Calculate the complementary dna sequence for BPE embedding
def complement_dna(dna_sequence):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    complementary_sequence = ''.join(complement.get(base, '') for base in dna_sequence)
    return complementary_sequence

def calculate_llm_embedding(dna_sequences, model_tokenizer_path, model_max_length=10240, batch_size=25, resolution=10240):
    
    # reorder the sequences by length
    # process sequences with similar lengths in the same batch can greatly speed up the computation
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_tokenizer_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )


    model = transformers.AutoModel.from_pretrained(
            model_tokenizer_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")

    train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='max_length', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
            
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            
            if resolution<model_max_length:
                embedding = ave_reso_step(model_output, attention_mask, resolution)
            else:
                embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            
            if j==0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().float().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings

def calculate_bpe_embedding(dna_sequences, tokenizer_path, model_max_length=10240):
    # reorder the sequences by length
    # process sequences with similar lengths in the same batch can greatly speed up the computation
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    # Number of token types
    token_types = 4091
    embeddings = np.zeros((len(dna_sequences), token_types))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )
    
    for i, seq in enumerate(tqdm.tqdm(dna_sequences)):
        token_feat = tokenizer(
                seq, 
                max_length=model_max_length, 
                return_tensors='pt', 
                padding='max_length', 
                truncation=True
            )

        for token in token_feat['input_ids'].tolist()[0]:
            if token >=5:
                embeddings[i, token-5] += 1
    
        # Complement sequence
        com_seq = complement_dna(seq)
        token_feat = tokenizer(
                com_seq, 
                max_length=model_max_length, 
                return_tensors='pt', 
                padding='max_length', 
                truncation=True
            )
        for token in token_feat['input_ids'].tolist()[0]:
            if token >=5:
                embeddings[i, token-5] += 1
    
        # Normalize to get fractions
        embeddings[i, :] /= np.sum(embeddings[i, :])
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings

def calculate_tnf_embedding(dna_sequences):
    # Define all possible tetra-nucleotides
    nucleotides = ['A', 'T', 'C', 'G']
    tetra_nucleotides = [a+b+c+d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]
    
    # build mapping from tetra-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}        

    # Iterate over each sequence and update counts
    embeddings = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    for j, seq in enumerate(tqdm.tqdm(dna_sequences)):
        for i in range(len(seq) - 3):
            tetra_nuc = seq[i:i+4]
            if tetra_nuc in tetra_nucleotides:
                embeddings[j, tnf_index[tetra_nuc]] += 1
    
    # Convert counts to frequencies
    total_counts = np.sum(embeddings, axis=1)
    embeddings = embeddings / total_counts[:, None]

    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="use which type of embedding: Genome Ocean; BPE; TNF")
    parser.add_argument("--model_tokenizer_path", type=str, default=None, help="model and tokenizer path")
    parser.add_argument("--data_path", type=str, default=None, help="path to the DNA sequences. Expect a txt file with each line as a DNA sequence")
    parser.add_argument("--output_path", type=str, default=None, help="path to save the embedding")
    parser.add_argument("--batch_size", type=int, default=4)  # adjust this to fit your GPU
    parser.add_argument("--model_max_length", type=int, default=10240)  # set this as 0.25 * DNA_length
    parser.add_argument("--resolution", type=int, default=10240)
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        dna_sequences = f.read().splitlines()
    
    print(f"Get {len(dna_sequences)} sequences from {args.data_path}")
    print(f"Model: {args.model_tokenizer_path}")
    print(f"Max length: {args.model_max_length}")
    print(f"Batch size: {args.batch_size}")    
    
    if args.type == "GO":
        embedding = calculate_llm_embedding(dna_sequences, 
                                            model_tokenizer_path=args.model_tokenizer_path, 
                                            model_max_length=args.model_max_length, 
                                            batch_size=args.batch_size,
                                            resolution=args.resolution)
    elif args.type == "BPE":
        embedding = calculate_bpe_embedding(dna_sequences, 
                                            tokenizer_path=args.model_tokenizer_path, 
                                            model_max_length=args.model_max_length)
    elif args.type == "TNF":
        embedding = calculate_tnf_embedding(dna_sequences)
    
    if args.output_path is None:
        args.output_path = args.data_path.replace(".txt", "_embedding.npy")
    
    print(f"Embeddings shape {embedding.shape}")
    print(f"Save embeddings to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    np.save(os.path.join(args.output_path, "embedding.npy"), embedding)

if __name__ == "__main__":
    main()