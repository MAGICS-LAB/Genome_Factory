import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModel
from genomeAI.Train.workflow.adapter.workflow_adapter_regression import AdapterModel
from safetensors.torch import load_file
def main():
    parser = argparse.ArgumentParser(
        description="Unified Inference Script"
    )
    parser.add_argument("--model_path", type=str, default="/mnt/c/Users/11817/GenomeAI/Trained_model666",#"./Trained_model666",#"./Trained_model666",
                        help="Path to the merged model directory (full or LoRA).")
    parser.add_argument("--dna", type=str, default="ATTGGTGGAATGCACAGGATATTGTGAAGGAGTACAG",
                        help="DNA sequence to infer.")
    parser.add_argument("--num_labels", type=str, default="1",
                        help="Number of labels.")
    parser.add_argument("--model_max_length", type=str, default="128",
                        help="Maximum length of the input sequence.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer + model from the same folder
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "pGenomeOcean/GenomeOcean-100M",#args.model_path,
        trust_remote_code=True
    )
    model = AdapterModel(pretrained_model=model, num_labels=int(args.num_labels))
    try:
        model.load_state_dict(torch.load(f"{args.model_path}/pytorch_model.bin"), strict=False)
    except:
        model.load_state_dict(load_file(f"{args.model_path}/model.safetensors"), strict=False)
    model = model.to(device)
    #pytorch_model.bin
    # Inference
    model.eval()  # recommended for consistent output
    inputs = tokenizer(args.dna, return_tensors='pt', truncation=True, max_length=int(args.model_max_length))
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)[1]
    print("Predicted value:", output.squeeze(-1))

if __name__ == "__main__":
    main()