
# genomeAI/inference.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser(
        description="Unified Inference Script"
    )
    parser.add_argument("--model_path", type=str, default="./Trained_model",
                        help="Path to the merged model directory (full or LoRA).")
    parser.add_argument("--dna", type=str, default="ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG...",
                        help="DNA sequence to infer.")
    args = parser.parse_args()

    # Load tokenizer + model from the same folder
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # Inference
    model.eval()  # recommended for consistent output
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits
    predicted_classes = torch.argmax(logits, dim=-1)
    print("Predicted classes:", predicted_classes)

if __name__ == "__main__":
    main()
