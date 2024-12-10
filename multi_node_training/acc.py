import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_scheduler,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator
from transformers import GPT2Config  # 引入 GPT2Config


# 加载数据集
raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

# 加载分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


custom_config = GPT2Config(
    n_embd=3072,  # 隐藏层维度，从 768 增加到 3072
    n_layer=12,  # Transformer 层数，从 12 增加到 48
    n_head=64,  # 注意力头数量，从 12 增加到 32
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GPT2LMHeadModel(custom_config)  # 使用自定义配置初始化 GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# 分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"])


# 对数据集进行分词
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)

# 定义块大小
block_size = 128


# 将文本分组为固定长度
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# 应用分组函数
lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)

# 创建数据加载器
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)

eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

# 设置优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 初始化 Accelerator
accelerator = Accelerator()

# 准备模型、优化器和数据加载器
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
from tqdm import tqdm
for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch in tqdm(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # 评估
    model.eval()
    losses = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch["input_ids"].size(0))))
    loss = torch.mean(torch.cat(losses))
    perplexity = math.exp(loss)
    print(f"Epoch {epoch + 1}: Perplexity: {perplexity}")
