# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .Mamba.modeling_caduceus import CaduceusMixerModel
# from transformers import AutoConfig

# class BiMamba_S(nn.Module):
#     def __init__(self, embedding_dim=768*2, output_dim=768, input_len=125, feat_dim=128):
#         super(BiMamba_S, self).__init__()
#         config_overrides = {
#             "d_model": embedding_dim,
#             "n_layer": 3
#         }
#         config = AutoConfig.from_pretrained(
#             "/root/weiminwu/Cell_free_DNA/Head_test/cfDNA/models/Mamba/",
#             **config_overrides,
#             trust_remote_code=True
#             ) 
#         self.model = CaduceusMixerModel(config)
#         self.emb_size = output_dim
#         self.feat_dim = feat_dim
        
#         self.dense_layer = nn.Linear(embedding_dim, output_dim)

#         self.contrast_head = nn.Sequential(
#             nn.Linear(self.emb_size, self.emb_size, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.emb_size, self.feat_dim, bias=False))
        
#     def forward(self, inputs, labels=None):        
#         input_mean_1, input_mean_2 = torch.unbind(inputs[0], dim=1)
#         input_std_1, input_std_2 = torch.unbind(inputs[1], dim=1)
        
#         inputs_1 = torch.concatenate([input_mean_1, input_std_1], dim=-1)
#         inputs_2 = torch.concatenate([input_mean_2, input_std_2], dim=-1)

#         output_1 = self.model(inputs_1)[0]
#         output_2 = self.model(inputs_2)[0]
        
#         output_1 = self.dense_layer(torch.mean(output_1, dim=1))
#         output_2 = self.dense_layer(torch.mean(output_2, dim=1))
        
#         cnst_feat1, cnst_feat2 = self.contrast_logits(output_1, output_2)
     
#         return cnst_feat1, cnst_feat2, output_1, output_2
            
#     def contrast_logits(self, embd1, embd2):
#         feat1 = self.contrast_head(embd1)
#         feat2 = self.contrast_head(embd2)
#         return feat1, feat2

#     def save_pretrained(self, dir):
#         pass