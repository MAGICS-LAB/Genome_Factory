
# genomeAI/inference.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser(
        description="Unified Inference Script"
    )
    parser.add_argument("--model_path", type=str, default="./Trained_model666",
                        help="Path to the merged model directory (full or LoRA).")
    parser.add_argument("--dna", type=str, default="ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG",
                        help="DNA sequence to infer.")
    parser.add_argument("--model_max_length", type=str, default="128",
                        help="Maximum length of the input sequence.")
    args = parser.parse_args()

    # Load tokenizer + model from the same folder
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference
    model.eval()  # recommended for consistent output
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=int(args.model_max_length))
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        #output = model(inputs["input_ids"])
        
        try:
            output = model(**inputs)
        except:
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            
    logits = output.logits
    predicted_classes = torch.argmax(logits, dim=-1)
    print("Predicted classes:", predicted_classes)

if __name__ == "__main__":
    main()
