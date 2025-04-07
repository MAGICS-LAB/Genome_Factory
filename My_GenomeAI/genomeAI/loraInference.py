

# genomeAI/loraInference.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./lora_Trained_model",
                        help="Path to the LoRA adapter directory.")
    parser.add_argument("--dna", type=str, default="GATAATTCTGGAGATGGCAGATG...",
                        help="DNA sequence to infer.")
    args = parser.parse_args()

    # Try loading tokenizer from model_path first, fallback base_model
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=128)
    output = model(**inputs)
    logits = output.logits
    predicted_classes = torch.argmax(logits, dim=-1)
    print("Predicted classes:", predicted_classes)

if __name__ == "__main__":
    main()

