import argparse
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

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
        default="pGenomeOcean/GenomeOcean-100M",
        help="Name or path of the Hugging Face model"
    )
    parser.add_argument(
        "--dna",
        type=str,
        nargs='+',
        default=[
            "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
            "GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT"
        ],
        help="One or more DNA sequences to process"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./embeddings.npy",
        help="Output file (must end with .npy)"
    )
    args = parser.parse_args()

    # ---------------------------
    # 2. Select device
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force ATen backend on CPU to avoid Triton errors
    if device.type == "cpu":
        try:
            import torch._inductor as _inductor
            _inductor.config.cpu_backend = "aten"
        except ImportError:
            pass

    # ---------------------------
    # 3. Load model & tokenizer
    # ---------------------------
    
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model.to(device)

    # ---------------------------
    # 4. Tokenize inputs
    # ---------------------------
    inputs = tokenizer(
        args.dna,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ---------------------------
    # 5. Forward & extract
    # ---------------------------
    model.eval()
    with torch.no_grad():
        if "GenomeOcean" in args.model_path:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        else:
            outputs = model(**inputs)

    last_hidden_state = outputs[0]  # shape: (batch_size, seq_len, hidden_size)

    # ---------------------------
    # 6. Upcast, convert to NumPy, and save
    # ---------------------------
    # ← Here’s the only change: upcast to float32 before numpy()
    emb_array = last_hidden_state.float().cpu().numpy()  # numpy doesn’t support bfloat16 :contentReference[oaicite:1]{index=1}
    np.save(args.output_file, emb_array)

    print(f"Embeddings saved to {args.output_file} (shape: {emb_array.shape})")

if __name__ == "__main__":
    main()