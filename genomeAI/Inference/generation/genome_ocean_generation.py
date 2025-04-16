import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
def main():
    parser = argparse.ArgumentParser(
        description="Unified generation Script"
    )
    parser.add_argument("--model_path", type=str, default="pGenomeOcean/GenomeOcean-100M",
                        help="Path to the generation model.")
    parser.add_argument("--dna", type=str, default="GCCGCTAAAAAGCGACCAGAATGATCCAAAAAAGAAGGCAGGCCAGCACCATCCGTTTTTTACAGCTCCAGAACTTCCTTT",
                        help="DNA sequence to keep generating.")
    parser.add_argument("--min_new_tokens", type=str, default="10",
                        help="Minimum number of tokens to generate.")
    parser.add_argument("--max_new_tokens", type=str, default="10",
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--do_sample", type=str, default="true",
                        help="Whether to sample from the model.")
    parser.add_argument("--top_p", type=str, default="0.9",
                        help="Top-p value for sampling.")
    parser.add_argument("--temperature", type=str, default="1.0",
                        help="Temperature value for sampling.")
    parser.add_argument("--num_return_sequences", type=str, default="1",
                        help="Number of return sequences.")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="left",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    ).to(device) 

    # Generation
    
    input_ids = tokenizer(args.dna, return_tensors='pt', padding=True)["input_ids"]
    input_ids = input_ids[:, :-1].to(device)   # remove the [SEP] token at the end
    model_output = model.generate(
        input_ids=input_ids,
        min_new_tokens=int(args.min_new_tokens),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        top_p=float(args.top_p),
        temperature=float(args.temperature),
        num_return_sequences=int(args.num_return_sequences),
            )
    generated = tokenizer.decode(model_output[0]).replace(" ", "")[5+len(args.dna):]
    print(f"Generated sequence: {generated}")

if __name__ == "__main__":
    main()