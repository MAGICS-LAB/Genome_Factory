import argparse
import torch
import numpy as np
from evo import Evo
from transformers import AutoModel, AutoTokenizer
from torch import nn

def main():
    # ---------------------------
    # 1. Parse command-line args
    # ---------------------------
    parser = argparse.ArgumentParser(
        description="Extract and save token embeddings from a pretrained model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="evo-1-131k-base",
        help="Name or path of the Hugging Face model"
    )
    parser.add_argument(
        "--dna",
        type=str,
        default="GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
        help="One or more DNA sequences to process"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./embeddings.npy",
        help="Output file (must end with .npy)"
    )
    args = parser.parse_args()

    device = 'cuda:0'

    # Load Evo model and tokenizer
    evo_model = Evo(args.model_path)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()  # disable dropout, layernorm in training mode

    # Replace unembed so that model returns embeddings directly
    class CustomEmbedding(nn.Module):
        def unembed(self, u):
            return u

    model.unembed = CustomEmbedding()

    # Prepare input
    sequence = args.dna
    input_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).to(device).unsqueeze(0)

    # Inference without gradient tracking
    with torch.no_grad():
        embed, _ = model(input_ids)

    # Convert to NumPy and save
    emb_array = embed.float().cpu().numpy()
    np.save(args.output_file, emb_array)

    print(f"Embeddings saved to {args.output_file} (shape: {emb_array.shape})")

if __name__ == "__main__":
    main()