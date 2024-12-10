import os
import csv
import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import time
import argparse
from sklearn.preprocessing import normalize



def calculate_llm_embedding(dna_sequences, model_name_or_path="/root/DNABERT_3/models/4B", model_max_length=400, batch_size=25):
    # reorder the sequences by length
    # process sequences with similar lengths in the same batch can greatly speed up the computation
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    if "InstaDeepAI" in model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #         model_name_or_path,
    #         trust_remote_code=True,
    #         torch_dtype=torch.bfloat16, 
    #         attn_implementation="flash_attention_2",
    #     )
    
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
        )
    

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")

    train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        if j == 1:
            start = time.time()
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
            
    print(time.time()-start)



def calculate_genslm_embedding(dna_sequences, model_name_or_path="/root/Downloads/", model_max_length=400, batch_size=25):
    from genslm import GenSLM, SequenceDataset

    # Load model
    model = GenSLM("genslm_2.5B_patric", model_cache_dir=model_name_or_path)
    model.eval()

    # Select GPU device if it is available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Input data is a list of gene sequences
    n_gpu = torch.cuda.device_count()
    dataset = SequenceDataset(dna_sequences, model_max_length, model.tokenizer)
    dataloader = util_data.DataLoader(dataset, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)

    # Compute averaged-embeddings for each input sequence
    embeddings = []
    with torch.no_grad():
        for j, batch in enumerate(tqdm.tqdm(dataloader)):
            if j == 1:
                start = time.time()
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
            # # outputs.hidden_states shape: (layers, batch_size, sequence_length, hidden_size)
            # # Use the embeddings of the last layer
            # emb = outputs.hidden_states[-1].detach().cpu().numpy()

    print(time.time()-start)

data_dir = "/root/data/cami2/reference/clustering_0.tsv"
with open(data_dir, "r") as f:
    lines = list(csv.reader(f, delimiter="\t"))[1:]
    lines = [line[0] for line in lines]
    
# batch_size = 8
# lines = lines[:128+batch_size]
# lines = [l*5 for l in lines]
# print(len(lines[0]))

# seq_len = 4096
# lines = [line[:seq_len] for line in lines]
# print(seq_len)
# # embeddings = calculate_llm_embedding(lines, model_max_length=50000, batch_size=batch_size)
# embeddings = calculate_llm_embedding(lines, model_name_or_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species", model_max_length=seq_len//6, batch_size=batch_size*4)
# # embeddings = calculate_llm_embedding(lines, model_name_or_path='togethercomputer/evo-1-131k-base', model_max_length=seq_len, batch_size=batch_size)
# # embeddings = calculate_genslm_embedding(lines, model_max_length=seq_len//3, batch_size=batch_size)




def generate_sequences(
    model_dir,
    dna_sequences=None,
    num_generation_from_each_prompt=1,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=1,          
    penalty_alpha=0.6,
    batch_size=50,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
    if "evo" in model_dir:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
            revision="1.1_fix",
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2",
        )
    
    model.to("cuda")
    
    print(f"Will generate {num_generation_from_each_prompt * len(dna_sequences)} sequences from {len(dna_sequences)} prompts")
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    outputs = []
    train_loader = torch.utils.data.DataLoader(dna_sequences, batch_size=batch_size, shuffle=False, num_workers=1)
    with torch.no_grad():
        for i, sequence in enumerate(tqdm.tqdm(train_loader)):
            if i == 1:
                start = time.time()
        
        
            input_ids = tokenizer(sequence, return_tensors='pt', padding=True)["input_ids"].to("cuda")       
            model_output = model.generate(
                input_ids=input_ids,
                min_new_tokens=min_length,
                max_new_tokens=max_length,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_generation_from_each_prompt,
                penalty_alpha=penalty_alpha,
            )
            # ).cpu().numpy().tolist()
            # for text in model_output:
            #     text = tokenizer.decode(text, skip_special_tokens=True).replace(" ", "").replace("\n", "")
            #     outputs.append(text)
    print(time.time()-start)
    
    return outputs

def generate_sequences_vllm(
    model_dir, 
    prompts=[""],
    num_generation_from_each_prompt=100,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=50,     
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
        stop_token_ids=[2],
        max_tokens=max_length,
        min_tokens=min_length,
        detokenize=False,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
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
            print(len(text))
            generated_sequences.append(text)

    return generated_sequences

def generate_evo_sequences(
    dna_sequences=None,
    num_generation_from_each_prompt=1,
    max_length=1024,
    batch_size=2,
    ):
    from evo import Evo, generate
    
    evo_model = Evo('evo-1-131k-base')
    model, tokenizer = evo_model.model, evo_model.tokenizer

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # Sample sequences.
    
    print('Generated sequences:')
    prompts = dna_sequences * num_generation_from_each_prompt
    train_loader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False, num_workers=1)
    for i, seq in enumerate(train_loader):
        if i == 1:
            start = time.time()
        
            
            
        output_seqs, output_scores = generate(
            seq,
            model,
            tokenizer,
            n_tokens=max_length,
            temperature=1.0,
            top_k=4,
            top_p=1.0,
            cached_generation=True,
            batched=True,
            prepend_bos=False,
            device=device,
            verbose=1,
        )
    print(time.time()-start)

def seq_to_3mer(seq):
    seq = seq.replace("\n", "")
    kmers = [seq[i:i+3] for i in range(0, len(seq), 3)]
    if len(kmers[-1]) < 3:
        kmers = kmers[:-1]
    seq = " ".join(kmers)
    return seq

def decode_3mer(seq):
    seq = seq.replace(" ", "")
    return seq

def generate_genslm_sequences(
    model_dir,
    dna_sequences=None,
    num_generation_from_each_prompt=1,
    temperature=0.7,
    min_length=128,
    max_length=1024, 
    top_k=1,          
    penalty_alpha=0.6,
    batch_size=2,
):
    from genslm import GenSLM, SequenceDataset

    # Load model
    model = GenSLM("genslm_250M_patric", model_cache_dir=model_dir)
    model.eval()

    # Select GPU device if it is available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    outputs = []
    prompts = [seq_to_3mer(seq) for seq in dna_sequences]
    train_loader = torch.utils.data.DataLoader(prompts, batch_size=batch_size, shuffle=False, num_workers=1)
    for i, batch in enumerate(tqdm.tqdm(train_loader)):
        if i == 1:
            start = time.time()
        
        prompt = model.tokenizer(batch, return_tensors="pt", padding=True)["input_ids"].to(device)
        
        print(prompt.shape)
        
        tokens = model.model.generate(
            prompt,
            max_new_tokens=max_length,  # Increase this to generate longer sequences
            min_new_tokens=min_length,
            do_sample=True,
            top_k=50,
            top_p=1,
            num_return_sequences=1,  # Change the number of sequences to generate
            remove_invalid_values=True,
            use_cache=True,
            pad_token_id=model.tokenizer.encode("[PAD]")[0],
            temperature=1.0,
        )
    

        sequences = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        # print(sequences)
        sequences = [decode_3mer(seq)[len(seq):] for seq in sequences]
        print(len(sequences[0]))
        # outputs.append(sequences[0])
        
    print(time.time()-start)

    return outputs

lines = [line[:1024] for line in lines]


### GenomeOcean
seq_len = 230
batch_size = 100
lines = lines[:200+batch_size]
outputs = generate_sequences("/root/DNABERT_3/models/4B", dna_sequences=lines, num_generation_from_each_prompt=1, temperature=0.7, min_length=seq_len, max_length=seq_len, top_k=1, batch_size=batch_size)

### Evo - HF
# seq_len = 1024
# batch_size = 2
# lines = lines[:100+batch_size]
# outputs = generate_sequences("togethercomputer/evo-1-131k-base", dna_sequences=lines, num_generation_from_each_prompt=1, temperature=0.7, min_length=seq_len, max_length=seq_len, top_k=1, penalty_alpha=0.6, batch_size=batch_size)

### Evo - Github
# seq_len = 1024
# batch_size = 2
# lines = lines[:10+batch_size]
# outputs = generate_evo_sequences(dna_sequences=lines, num_generation_from_each_prompt=1, max_length=seq_len)

### GenSLM
# seq_len = 1024 // 3
# batch_size = 10
# lines = lines[:100+batch_size]
# outputs = generate_genslm_sequences("/root/Downloads/", dna_sequences=lines, num_generation_from_each_prompt=1, temperature=0.7, min_length=seq_len, max_length=seq_len, top_k=1, penalty_alpha=0.6, batch_size=batch_size)




# outputs = generate_sequences_vllm("/root/DNABERT_3/models/4B", prompts=lines, num_generation_from_each_prompt=1, temperature=0.7, min_length=seq_len, max_length=seq_len, top_k=1)
