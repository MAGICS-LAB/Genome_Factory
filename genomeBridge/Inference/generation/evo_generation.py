from evo import Evo
import torch
import argparse

def main():
    device = 'cuda:0'
    parser = argparse.ArgumentParser(
        description="Unified generation Script"
    )
    parser.add_argument("--model_path", type=str, default="evo-1-131k-base",
                        help="Path to the generation model.")
    parser.add_argument("--dna", type=str, default="GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
                        help="DNA sequence to keep generating.")
    args = parser.parse_args()
    evo_model = Evo(args.model_path)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()

    input_ids = torch.tensor(
        tokenizer.tokenize(args.dna),
        dtype=torch.int,
    ).to(device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(input_ids) # (batch, length, vocab)
    logits = logits.squeeze(0)
    id = logits.argmax(dim=-1)
    print(tokenizer.detokenize(id))

if __name__ == "__main__":
    main()