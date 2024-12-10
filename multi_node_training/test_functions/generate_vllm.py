import os
import transformers
import torch
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset

# set it to prevent the warning message when use the model
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Change this to the directory of the model
model_dir = "/root/DNABERT_3/models/4B"


"""
Parameters:
- model_dir: str
    The directory of the model
- prompts: List[str]   
    A list of prompts to generate from. If not provided, the model will generate from scratch
- num_generation_from_each_prompt: int
    The number of sequences to generate from each prompt
- temperature: float
    The temperature of the sampling
- min_length: int
    The minimum length of the generated sequence (in tokens)
- max_length: int
    The maximum length of the generated sequence (in tokens)
- top_k: int   
    The top_k parameter for sampling
- presence_penalty: float
    Penalizes new tokens based on whether they appear in the generated text so far. 
    Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
- frequency_penalty: float
    Penalizes new tokens based on their frequency in the generated text so far. 
    Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.
- repetition_penalty: float
    Penalizes new tokens based on whether they appear in the prompt and the generated text so far. 
    Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens.
"""
def generate_sequences(
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

    all_outputs = llm.generate(
        prompts=None, 
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )


    generated_sequences = []
    for outputs in all_outputs:
        for output in outputs.outputs:
            text = tokenizer.decode(output.token_ids, skip_special_tokens=True).replace(" ", "").replace("\n", "")
            generated_sequences.append(text)

    return generated_sequences


"""
Example usage:
"""

# Generate from scratch
# generated_sequences = generate_sequences(model_dir)

# Generate from existing dna sequences 
dna_sequences = [
        "TGAAGTGGTACAGAGATGTTCATTAAGCCAACAACTAGACCCATTGCTACGAAGAAGATCATGATGACACCGTGTGCGGTGAA",
        "CCCATAATGGAGGCCGACAAGCGATGCGAATTACCAATCACAAAGGTAACCGCCATGGTTTCGCCCAATGCCCGGCCCAGCCCC"
        ] # save the sequences in a list
generate_sequences(model_dir, prompts=dna_sequences, min_length=5, max_length=30, num_generation_from_each_prompt=1)
