import os
import tqdm
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# set it to prevent the warning message when use the model
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Change this to the sequences you want to score
dna_sequences = None
# with open("/root/data/pre-train/metagenomics/gre_15k.txt", "r") as f:
#     dna_sequences = f.readlines()[:2]
    

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B"

def generate_sequences(
    model_dir,
    dna_sequences=None,
    num_generation_in_total=10,
    num_generation_from_each_prompt=4,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,          
    penalty_alpha=0.6,
):
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )
    
    model.to("cuda")
    
    if dna_sequences is None:
        dna_sequences = ["" for _ in range(num_generation_in_total)]
        num_generation_from_each_prompt = 1
        print(f"No prompt provided. Will generate {num_generation_in_total} sequences")
    else:
        print(f"Will generate {num_generation_from_each_prompt * len(dna_sequences)} sequences from {len(dna_sequences)} prompts")
        
    class MyDataset(Dataset):
        def __init__(self, dna_sequences):
            self.dna_sequences = dna_sequences
            
        def __len__(self):
            return len(self.dna_sequences)

        def __getitem__(self, i):
            return self.dna_sequences[i]

    outputs = []
    dataset = MyDataset(dna_sequences=dna_sequences)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, sequence in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            input_ids = tokenizer(sequence, return_tensors='pt')["input_ids"][:, :-1].to("cuda")       
            model_output = model.generate(
                input_ids=input_ids,
                min_length=min_length,
                max_length=max_length,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_generation_from_each_prompt,
                bad_words_ids=[[8]],
                penalty_alpha=penalty_alpha,
            ).cpu().numpy().tolist()
            for text in model_output:
                text = tokenizer.decode(text, skip_special_tokens=True).replace(" ", "").replace("\n", "")
                outputs.append(text)
    
    return outputs

""" 
This functions runs on single GPU using 4-bits quanlitization.
The best practice is to start one process per GPU.

Example usage:

1. If you want the model to generate sequences from scratch:
        let dna_sequences = None, num_generation_in_total be the number of sequences you want to generate
        generate "num_generation_in_total" sequences
    If you want the model to generate sequences from a prompt:
        let dna_sequences to be a list of sequences, num_generation_from_each_prompt be the number of sequences you want to generate from each prompt
        generate "num_generation_from_each_prompt * len(dna_sequences)" sequences
        
2. model_dir is the directory of the model you want to use for generation

3. min_length / max_length are the minimum and maximum lengths of the generated sequences in BPE tokens

4. top_k is the number of top tokens to consider for sampling

5. temperature is the sampling temperature. Lower values will result in more deterministic outputs, while higher values will result in more random outputs. 
    Evo use [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3]
"""


generated_sequences = generate_sequences(
    model_dir=model_dir,
    dna_sequences=dna_sequences,
    num_generation_in_total=100,
    num_generation_from_each_prompt=2,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,       
    penalty_alpha=0.6,   
)