import tqdm
import numpy as np
import transformers
import torch
import torch.nn as nn
from torch.utils.data import Dataset



# Change this to the sequences you want to score
dna_sequences = None
# with open("/root/data/pre-train/metagenomics/gre_15k.txt", "r") as f:
#     dna_sequences = f.readlines()[:2]
    

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B"

def generate_sequences(
    model_dir,
    dna_sequences=None,
    num_generation_from_each_prompt=4,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,          
):
    

    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )


    class MyDataset(Dataset):
        def __init__(self, dna_sequences):
            self.dna_sequences = dna_sequences
            if self.dna_sequences is None:
                self.len = num_generation_from_each_prompt
                print(f"No prompt provided. Will generate {self.len} sequences")
            else:
                self.len = len(self.dna_sequences) * num_generation_from_each_prompt
                print(f"Will generate {self.len} sequences for {len(self.dna_sequences)} prompts")
            

        def __len__(self):
            return self.len // num_generation_from_each_prompt

        def __getitem__(self, i):
            if self.dna_sequences is None:
                return ""
            else:
                return self.dna_sequences[i]


    dataset = MyDataset(dna_sequences=dna_sequences)


    outputs = []
    generator = transformers.pipeline(
        'text-generation', 
        model = model,
        tokenizer = tokenizer,
    )

    dna_generator = generator(
        dataset,
        min_length=min_length,
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        temperature=temperature,
        num_return_sequences=num_generation_from_each_prompt,
    )
    for generated_text in tqdm.tqdm(dna_generator):
        for text in generated_text:
            outputs.append(text["generated_text"].replace(" ", "").replace("\n", ""))

    
    return outputs

""" 
This functions runs on single GPU using 4-bits quanlitization.
The best practice is to start one process per GPU.

Example usage:

1. If you want the model to generate sequences from scratch:
        let dna_sequences = None, then the model with generate in total "num_generation_from_each_prompt" sequences
    If you want the model to generate sequences from a prompt:
        let dna_sequences to be a list of sequences, then the model will generate total "num_generation_from_each_prompt * len(dna_sequences)" sequences
        
2. model_dir is the directory of the model you want to use for generation

3. min_length / max_length are the minimum and maximum lengths of the generated sequences in BPE tokens

4. top_k is the number of top tokens to consider for sampling

5. temperature is the sampling temperature. Lower values will result in more deterministic outputs, while higher values will result in more random outputs. 
    Evo use [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3]
"""


generated_sequences = generate_sequences(
    model_dir=model_dir,
    dna_sequences=dna_sequences,
    num_generation_from_each_prompt=10,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,          
)