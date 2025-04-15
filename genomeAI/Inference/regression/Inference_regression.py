

# genomeAI/loraInference.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
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
        trust_remote_code=True,
    )
    model.eval()
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=128)
    output = model(**inputs).logits
    #logits = output.logits
    predicted_value = output.squeeze(-1)
    print("Predicted value:", predicted_value)
    

if __name__ == "__main__":
    main()

