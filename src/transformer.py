import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk

# 获取当前文件所在的目录（即 src 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定义项目根目录，并定位到 data/cnn_dailymail
dataset_path = os.path.join(current_dir, "../data/cnn_dailymail")

# 加载模型和tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载本地数据集
dataset = load_from_disk(dataset_path)

# 数据预处理
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # 设置label
    labels = tokenizer(examples["highlights"], max_length=150, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 多GPU训练设置
results_dir = os.path.join(current_dir, "../results")
training_args = TrainingArguments(
    output_dir=results_dir,  # 将结果存储到项目根目录的 results 文件夹
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # 使用混合精度训练（节省显存）
    dataloader_num_workers=4,
    gradient_accumulation_steps=2,  # 增大批次大小的有效方法
)

# Trainer 设置
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 启动训练
trainer.train()
