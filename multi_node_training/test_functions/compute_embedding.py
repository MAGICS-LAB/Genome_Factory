import os
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import argparse
from sklearn.preprocessing import normalize



def calculate_llm_embedding(dna_sequences, model_name_or_path, model_max_length=400, batch_size=25):
    # reorder the sequences by length
    # process sequences with similar lengths in the same batch can greatly speed up the computation
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )


    model = transformers.AutoModel.from_pretrained(
            model_name_or_path,
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
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
                
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            
            if j==0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().float().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="path to the DNA sequences. Expect a txt file with each line as a DNA sequence")
    parser.add_argument("--batch_size", type=int, default=16)  # adjust this to fit your GPU
    parser.add_argument("--model_max_length", type=int, default=1024)  # set this as 0.25 * DNA_length
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        dna_sequences = f.read().splitlines()
    
    print(f"Get {len(dna_sequences)} sequences from {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"Max length: {args.model_max_length}")
    print(f"Batch size: {args.batch_size}")    
    
    embedding = calculate_llm_embedding(dna_sequences, 
                                        model_name_or_path=args.model_path, 
                                        model_max_length=args.model_max_length, 
                                        batch_size=args.batch_size)
    embedding_norm = normalize(embedding)
    
    if args.output_path is None:
        args.output_path = args.data_path.replace(".txt", "_embedding.npy")
    
    print(f"Embeddings shape {embedding_norm.shape}")
    print(f"Save embeddings to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    np.save(os.path.join(args.output_path, "embedding.npy"), embedding_norm)


if __name__ == "__main__":
    main()