import random
from datasets import Dataset


# 生成随机文本数据
def generate_random_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        text = " ".join(
            random.choices(["Hello", "world", "test", "data", "Qwen", "Instruct"], k=5)
        )
        data.append({"input": text, "label": random.randint(0, 1)})
    return Dataset.from_dict(
        {"input": [d["input"] for d in data], "label": [d["label"] for d in data]}
    )


train_dataset = generate_random_data(num_samples=1000)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载Qwen模型和tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = tokenizer.eos_token_id


# 数据预处理
def preprocess_function(examples):
    return tokenizer(
        examples["input"], truncation=True, padding="max_length", max_length=128
    )


tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
from transformers import TrainingArguments, Trainer

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # 每个设备上的batch size
    num_train_epochs=3,
    deepspeed="ds_config.json",  # 指定deepspeed配置文件
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,  # 启用混合精度训练
    report_to="none",  # 不需要汇报到任何平台，如WandB
)

# 使用Trainer进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

# 开始训练
trainer.train()
