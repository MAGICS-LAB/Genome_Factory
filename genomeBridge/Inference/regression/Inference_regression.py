

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
    parser.add_argument("--model_max_length", type=str, default="128",
                        help="Maximum length of the input sequence.")
    args = parser.parse_args()

    # Try loading tokenizer from model_path first, fallback base_model
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=int(args.model_max_length))
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        try:
            output = model(**inputs).logits
        except:
            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
    #logits = output.logits
    predicted_value = output.squeeze(-1)
    print("Predicted value:", predicted_value)
    

if __name__ == "__main__":
    main()

