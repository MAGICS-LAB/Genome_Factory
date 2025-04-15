import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel
from genomeAI.Train.workflow.adapter.workflow_adapter_regression import AdapterModel

def main():
    parser = argparse.ArgumentParser(
        description="Unified Inference Script"
    )
    parser.add_argument("--model_path", type=str, default="./Trained_model",
                        help="Path to the merged model directory (full or LoRA).")
    parser.add_argument("--dna", type=str, default="ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG...",
                        help="DNA sequence to infer.")
    parser.add_argument("--num_labels", type=str, default="1",
                        help="Number of labels.")
    args = parser.parse_args()

    # Load tokenizer + model from the same folder
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    model = AdapterModel(pretrained_model=model, num_labels=int(args.num_labels))
    model.load_state_dict(torch.load(f"{args.model_path}/pytorch_model.bin"), strict=False)
    # Inference
    model.eval()  # recommended for consistent output
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs)[1]
    print("Predicted value:", output.squeeze(-1))

if __name__ == "__main__":
    main()