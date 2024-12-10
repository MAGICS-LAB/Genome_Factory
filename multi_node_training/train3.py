import torch
import deepspeed
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
# deepspeed.init_distributed()

# 创建一个随机生成的数据集
class RandomTextDataset(Dataset):
    def __init__(self, tokenizer, length=16, num_samples=1000):
        self.tokenizer = tokenizer
        self.length = length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机生成一个长度为length的token id序列
        random_text = "".join([chr(random.randint(65, 90)) for _ in range(self.length)])
        inputs = self.tokenizer(
            random_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.length,
            truncation=True,
        )
        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0)


# 设置分布式训练的配置
def setup_deepspeed(args):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model.cuda()

    # 创建随机数据集
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = RandomTextDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # ds_config = {
    #     "train_batch_size": 2,
    #     "zero_optimization": {
    #         "stage": 3,
    #         "overlap_comm": True,
    #         "contiguous_gradients": True,
    #         "sub_group_size": 1e9,
    #         "reduce_bucket_size": "auto",
    #         "stage3_prefetch_bucket_size": "auto",
    #         "stage3_param_persistence_threshold": "auto",
    #         "stage3_max_live_parameters": 1e9,
    #         "stage3_max_reuse_distance": 1e9,
    #         "stage3_gather_16bit_weights_on_model_save": True,
    #     },
    #     "optimizer": {
    #         "type": "Adam",
    #         "params": {"lr": 1e-5},
    #     },
    #     "activation_checkpointing": {
    #         "partition_activations": True,
    #     },
    #      "allgather_partitions": True,
    # "allgather_bucket_size": 5e8,
    # "overlap_comm": False,
    # "reduce_scatter": True,
    # "reduce_bucket_size": 5e8,
    # "contiguous_gradients" : True

    # }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters()
    )

    return model_engine, optimizer, dataloader

from tqdm import tqdm

# 主训练函数
def train(args):
    model_engine, optimizer, dataloader = setup_deepspeed(args)

    model_engine.train()
    for epoch in range(3):  # 假设训练3个epoch
        for step, (input_ids, attention_mask) in tqdm(enumerate(dataloader)):
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

            outputs = model_engine(
                input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher",
    )


    # DeepSpeed 分布式相关的参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    train(args)
