"""
Type 1: directly copy the weights from mistral to initialize Qwen-MoE
"""
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

moe_config = AutoConfig.from_pretrained("/root/MOE_DNA/model/model_qwenmoe_8_100M", trust_remote_code=True)
moe_model = AutoModelForCausalLM.from_config(moe_config, trust_remote_code=True)
mistral = AutoModelForCausalLM.from_pretrained("/root/weiminwu/dnabert-3/llm2vec/model/meta-100M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/root/weiminwu/dnabert-3/llm2vec/model/meta-100M", trust_remote_code=True)

all_moe_keys = set(moe_model.state_dict().keys())
changed_moe_keys = set()
left_mistral_keys = set()

for key in mistral.state_dict().keys():
    if key in all_moe_keys:
        moe_model.state_dict()[key].copy_(mistral.state_dict()[key])
        changed_moe_keys.add(key)
        all_moe_keys.remove(key)
    else:
        left_mistral_keys.add(key)

num_layers = moe_config.num_hidden_layers
num_experts = moe_config.num_experts

for l in range(num_layers):
    for sec in ["gate_proj", "up_proj", "down_proj"]:
        key_mistral = f"model.layers.{l}.mlp.{sec}.weight"
        mistral_weight = mistral.state_dict()[key_mistral]
        if sec != "down_proj":
            dim = mistral_weight.shape[0] // 4
            mistral_weight_1 = mistral_weight[:dim]
            mistral_weight_2 = mistral_weight[dim:2*dim]
            mistral_weight_3 = mistral_weight[2*dim:3*dim]
            mistral_weight_4 = mistral_weight[3*dim:]
        else:
            dim = mistral_weight.shape[1] // 4
            mistral_weight_1 = mistral_weight[:, :dim]
            mistral_weight_2 = mistral_weight[:, dim:2*dim]
            mistral_weight_3 = mistral_weight[:, 2*dim:3*dim]
            mistral_weight_4 = mistral_weight[:, 3*dim:]
        
        mapping = {
            0: mistral_weight_1,
            1: mistral_weight_2,
            2: mistral_weight_3,
            3: mistral_weight_4
        }
        
        key_share = f"model.layers.{l}.mlp.shared_expert.{sec}.weight"
        moe_model.state_dict()[key_share].copy_(mistral_weight)
        changed_moe_keys.add(key_share)
        all_moe_keys.remove(key_share)
        
        for e in range(num_experts):
            key_moe = f"model.layers.{l}.mlp.experts.{e}.{sec}.weight"
            moe_model.state_dict()[key_moe].copy_(mapping[e%4])
            changed_moe_keys.add(key_moe)
            all_moe_keys.remove(key_moe)
            
        left_mistral_keys.remove(key_mistral)

# count model parameters
num_params = sum(p.numel() for p in moe_model.parameters())
print(f"Number of parameters in M: {num_params/1e6:.2f}M")

# save model and tokenizer
moe_model.save_pretrained("/root/MOE_DNA/trained_model/qwenmoe_8_100M")
tokenizer.save_pretrained("/root/MOE_DNA/trained_model/qwenmoe_8_100M")



"""
Type 2: directly copy the weights from mistral to initialize Qwen-MoE, add additional dimensions that are randomly initialized
"""
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

moe_config = AutoConfig.from_pretrained("/root/MOE_DNA/model/model_qwenmoe_8_100M_random", trust_remote_code=True)
moe_model = AutoModelForCausalLM.from_config(moe_config, trust_remote_code=True)
mistral = AutoModelForCausalLM.from_pretrained("/root/weiminwu/dnabert-3/llm2vec/model/meta-100M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/root/weiminwu/dnabert-3/llm2vec/model/meta-100M", trust_remote_code=True)

all_moe_keys = set(moe_model.state_dict().keys())
changed_moe_keys = set()
left_mistral_keys = set()

for key in mistral.state_dict().keys():
    if key in all_moe_keys:
        moe_model.state_dict()[key].copy_(mistral.state_dict()[key])
        changed_moe_keys.add(key)
        all_moe_keys.remove(key)
    else:
        left_mistral_keys.add(key)

num_layers = moe_config.num_hidden_layers
num_experts = moe_config.num_experts

mistral_dim = mistral.config.intermediate_size
moe_dim = moe_config.intermediate_size
random_dim = (moe_dim - mistral_dim) // 4
pretrained_dim = mistral_dim // 4
print(f"Randomly initialized dimensions per expert: {random_dim}")
print(f"Pretrained dimensions per expert: {pretrained_dim}")

for l in range(num_layers):
    for sec in ["gate_proj", "up_proj", "down_proj"]:
        key_mistral = f"model.layers.{l}.mlp.{sec}.weight"
        mistral_weight = mistral.state_dict()[key_mistral]
        if sec != "down_proj":
            dim = mistral_weight.shape[0] // 4
            mistral_weight_1 = mistral_weight[:dim]
            mistral_weight_2 = mistral_weight[dim:2*dim]
            mistral_weight_3 = mistral_weight[2*dim:3*dim]
            mistral_weight_4 = mistral_weight[3*dim:]
        else:
            dim = mistral_weight.shape[1] // 4
            mistral_weight_1 = mistral_weight[:, :dim]
            mistral_weight_2 = mistral_weight[:, dim:2*dim]
            mistral_weight_3 = mistral_weight[:, 2*dim:3*dim]
            mistral_weight_4 = mistral_weight[:, 3*dim:]
        
        mapping = {
            0: mistral_weight_1,
            1: mistral_weight_2,
            2: mistral_weight_3,
            3: mistral_weight_4
        }
        
        # Each expert with 768 (from mistral) + 16 (random) weights
        for e in range(num_experts):
            key_moe = f"model.layers.{l}.mlp.experts.{e}.{sec}.weight"
            if sec != "down_proj":
                moe_model.state_dict()[key_moe][:pretrained_dim].copy_(mapping[e%4])
            else:
                moe_model.state_dict()[key_moe][:, :pretrained_dim].copy_(mapping[e%4])
            changed_moe_keys.add(key_moe)
            all_moe_keys.remove(key_moe)
        
        key_share = f"model.layers.{l}.mlp.shared_expert.{sec}.weight"
        # Split each shared expert into 4 parts
        # Treat each part as an expert and fill it in the same way as the above expert
        for i in range(4):
            if sec != "down_proj":
                moe_model.state_dict()[key_share][(pretrained_dim+random_dim)*i:(pretrained_dim+random_dim)*i + pretrained_dim].copy_(mapping[i])
            else:
                moe_model.state_dict()[key_share][:, (pretrained_dim+random_dim)*i:(pretrained_dim+random_dim)*i + pretrained_dim].copy_(mapping[i])
        changed_moe_keys.add(key_share)
        all_moe_keys.remove(key_share)

            
        left_mistral_keys.remove(key_mistral)
        
print(f"Left mistral keys: {left_mistral_keys}")
print(f"Left moe keys: {sorted(list(all_moe_keys))}")

assert all(["gate" in key for key in all_moe_keys]), "Some keys are not initialized"

# count model parameters
num_params = sum(p.numel() for p in moe_model.parameters())
print(f"Number of parameters in M: {num_params/1e6:.2f}M")

# save model and tokenizer
moe_model.save_pretrained("/root/MOE_DNA/trained_model/qwenmoe_8_100M_random")
tokenizer.save_pretrained("/root/MOE_DNA/trained_model/qwenmoe_8_100M_random")
