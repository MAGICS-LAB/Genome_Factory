from typing import List
import csv
import os
import time
import torch
import transformers
import argparse
from vllm import LLM, SamplingParams


def bad_word_processor(token_ids, logits):
    # To suppress 'N's from being generated:	
    logits[8] = float("-inf")
    return logits


def generate_sequences_vllm(
    model_dir, 
    prompts=[""],
    num_generation_from_each_prompt=100,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,     
    top_p=0.95,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    repetition_penalty=1.0,
):  
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="slow",
        trust_remote_code=True,
        max_model_len=10240,
        seed=0,
        dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    
    sampling_params = SamplingParams(
        n=num_generation_from_each_prompt,
        temperature=temperature, 
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=[2],
        max_tokens=max_length,
        min_tokens=min_length,
        detokenize=False,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        logits_processors=[bad_word_processor], #To suppress 'N's from being generated, remove it if not needed
    )
    
    prompts = ["[CLS]"+p for p in prompts]
    prompt_token_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

    start = time.time()
    all_outputs = llm.generate(
        prompts=None, 
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    print(time.time()-start)


    generated_sequences = []
    for outputs in all_outputs:
        for output in outputs.outputs:
            text = tokenizer.decode(output.token_ids, skip_special_tokens=True).replace(" ", "").replace("\n", "")
            generated_sequences.append(text)

    
    return generated_sequences


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="4B", help="Model size [100M, 500M, 4B]")
    parser.add_argument("--data_dir", type=str, default="/root/data/cami2/marine_plant_20_unknown.tsv")
    parser.add_argument("--output_dir", type=str, default="/root/MOE_DNA/ICLR/generated")
    parser.add_argument("--num_generation_from_each_prompt", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_length", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--max_output_bp", type=int, default=-1)
    parser.add_argument("--max_prompt_bp", type=int, default=2000)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()
    

    delimiter = "," if args.data_dir.endswith(".csv") else "\t"
    with open(args.data_dir, "r") as f:
        lines = list(csv.reader(f, delimiter=delimiter))
        dna_sequences = [line[0][-args.max_prompt_bp:] for line in lines]
        labels = [line[1:] for line in lines]
        augmented_labels = []
        for label in labels:
            aug_label = [label] * args.num_generation_from_each_prompt
            augmented_labels.extend(aug_label)
        
    model_dir = os.path.join("/root/DNABERT_3/models", f"{args.model_size}")
    generated_sequences = generate_sequences_vllm(
        model_dir,
        prompts=dna_sequences,
        num_generation_from_each_prompt=args.num_generation_from_each_prompt,
        temperature=args.temperature,
        min_length=args.min_length,
        max_length=args.max_length, 
        top_k=args.top_k,     
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        
    )

    if args.max_output_bp > 0:
        print(f"Trimming sequences to {args.max_output_bp} bp")
        generated_sequences = [seq[:args.max_output_bp] for seq in generated_sequences]
    
    print(f"Saving generated sequences to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    file_type = "tsv" if delimiter == "\t" else "csv"
    with open(f"{args.output_dir}/go_{args.temperature}_{args.top_k}_{args.top_p}_{args.min_length}_{args.max_length}_{args.presence_penalty}_{args.frequency_penalty}_{args.repetition_penalty}_{args.num_generation_from_each_prompt}.{file_type}", "w") as f:
        for seq, label in zip(generated_sequences, augmented_labels):
            combines = [seq] + label
            f.write(f"{combines[0]}{delimiter}{delimiter.join(combines[1:])}\n")


