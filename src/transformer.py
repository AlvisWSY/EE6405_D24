import os
from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import multiprocessing
import torch

# Step 1: 动态定义目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径拼接目录
data_dir = os.path.join(current_dir, "../data", "processed")
output_model_dir = os.path.join(current_dir, "../models", "t5_finetuned")
output_dir = os.path.join(current_dir, "../output", "t5_summary")
cache_dir = os.path.join(current_dir, "../tmp")  # 缓存目录

# 设置缓存目录
os.environ["HF_DATASETS_CACHE"] = cache_dir

# 确保必要的目录存在
os.makedirs(output_model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 打印目录以确认
print(f"Data directory: {data_dir}")
print(f"Model output directory: {output_model_dir}")
print(f"Test results output directory: {output_dir}")
print(f"Cache directory: {cache_dir}")

# **指定可用的 GPU**
# 您可以使用第 1、2、3 号 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Step 2: 加载数据
def load_and_print_dataset(path, split_name):
    dataset = load_from_disk(path)
    print(f"Loaded {split_name}: {len(dataset)} samples")
    return dataset

# ArXiv 数据
arxiv_train = load_and_print_dataset(os.path.join(data_dir, "ArXiv/train"), "ArXiv train")
arxiv_validation = load_and_print_dataset(os.path.join(data_dir, "ArXiv/validation"), "ArXiv validation")
arxiv_test = load_and_print_dataset(os.path.join(data_dir, "ArXiv/test"), "ArXiv test")

# CNN/DailyMail 数据
cnn_train = load_and_print_dataset(os.path.join(data_dir, "cnn_dailymail/train"), "CNN train")
cnn_validation = load_and_print_dataset(os.path.join(data_dir, "cnn_dailymail/validation"), "CNN validation")
cnn_test = load_and_print_dataset(os.path.join(data_dir, "cnn_dailymail/test"), "CNN test")

# 合并训练集和验证集
train_dataset = concatenate_datasets([arxiv_train, cnn_train])
validation_dataset = concatenate_datasets([arxiv_validation, cnn_validation])
print(f"Combined train dataset size: {len(train_dataset)} samples")
print(f"Combined validation dataset size: {len(validation_dataset)} samples")

# Step 3: 加载预训练模型和 tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 4: 数据预处理
max_input_length = 512
max_target_length = 128

# 定义预处理函数
def preprocess_function(examples):
    try:
        inputs = ["summarize: " + text for text in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

        labels = tokenizer(
            examples["abstract"], max_length=max_target_length, padding="max_length", truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise e

# 优化数据预处理
def preprocess_dataset(dataset, num_proc, batch_size):
    max_threads = multiprocessing.cpu_count()
    num_proc = min(num_proc, max_threads)
    print(f"Using {num_proc} threads to process data, batch size: {batch_size}")

    def print_progress(batch):
        print(f"Processing batch of size: {len(batch['text'])}")
        return preprocess_function(batch)

    return dataset.map(
        print_progress,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text", "abstract"]
    )

# 调整线程数和批量大小
num_proc = 16  # 根据系统允许的最大线程数进行调整
batch_size = 10000  # 根据内存情况适当调整

train_dataset = preprocess_dataset(train_dataset, num_proc=num_proc, batch_size=batch_size)
validation_dataset = preprocess_dataset(validation_dataset, num_proc=num_proc, batch_size=batch_size)

# Step 5: 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=output_model_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # 增大批量大小以充分利用 GPU
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    predict_with_generate=True,
    load_best_model_at_end=True,
    fp16=True,  # 启用混合精度训练
    dataloader_num_workers=4,  # 根据允许的线程数进行调整
    # **启用分布式训练**
    # 设置 fp16_backend 为 "amp"
    fp16_backend="amp",
    # 如果需要，可以启用 gradient_checkpointing 以节省显存
    # gradient_checkpointing=True,
)

# Step 6: 定义数据收集器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# **确保模型可以使用多 GPU 进行训练**
# Trainer 会自动检测并使用多个 GPU，无需额外设置

# Step 7: 微调模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()
print("Training completed.")

# 保存模型
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model saved to {output_model_dir}")

# Step 8: 测试集推理并保存输出
def generate_predictions(test_dataset, dataset_name):
    processed_test = test_dataset.map(preprocess_function, batched=True, remove_columns=["text", "abstract"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 调整批量大小
    batch_size = 16  # 根据显存情况调整

    # 创建数据加载器
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"], device=device) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"], device=device) for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    dataloader = DataLoader(processed_test, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)

    predictions = []

    for batch in dataloader:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_target_length
            )
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded_preds)

    results = Dataset.from_dict({
        "abstract": test_dataset["abstract"],
        "output": predictions,
    })

    save_path = os.path.join(output_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    results.save_to_disk(save_path)
    print(f"Test results saved to {save_path}")

# 推理 ArXiv 和 CNN/DailyMail 测试集
generate_predictions(arxiv_test, "ArXiv")
generate_predictions(cnn_test, "cnn_dailymail")
