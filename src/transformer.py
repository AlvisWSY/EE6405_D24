import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from datasets import load_from_disk, Dataset as HFDataset

# 初始化分布式环境
def init_distributed_training(rank, world_size):
    print(f"[Rank {rank}] Initializing distributed training with world size {world_size}...")
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Distributed training initialized.")

# 数据加载函数
def load_datasets(data_path, datasets):
    print(f"[INFO] Loading datasets from {data_path}...")
    train_datasets = []
    val_datasets = []
    test_datasets = {}
    for dataset_name in datasets:
        print(f"[INFO] Loading dataset: {dataset_name}")
        train_datasets.append(load_from_disk(os.path.join(data_path, dataset_name, "train")))
        val_datasets.append(load_from_disk(os.path.join(data_path, dataset_name, "val")))
        test_datasets[dataset_name] = load_from_disk(os.path.join(data_path, dataset_name, "test"))
    print("[INFO] Datasets loaded successfully.")
    return train_datasets, val_datasets, test_datasets

# 动态分段函数
def split_long_text(text, tokenizer, max_length=512):
    tokens = tokenizer(text, truncation=False)["input_ids"]
    print(f"[DEBUG] Text length: {len(tokens)}, Splitting into segments of max length {max_length}...")
    segments = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    print(f"[DEBUG] Generated {len(segments)} segments.")
    return segments

# 数据集预处理函数（支持动态分段）
def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=150):
    print("[DEBUG] Preprocessing data with dynamic segmentation...")
    inputs = examples["text"]
    targets = examples["abstract"]

    # 动态分段
    input_segments = split_long_text(inputs, tokenizer, max_input_length)
    target_segments = split_long_text(targets, tokenizer, max_target_length)

    # 每段单独生成输入和标签
    model_inputs = []
    for inp, tgt in zip(input_segments, target_segments):
        input_data = tokenizer(
            tokenizer.convert_ids_to_tokens(inp),
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )
        label_data = tokenizer(
            tokenizer.convert_ids_to_tokens(tgt),
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )
        input_data["labels"] = label_data["input_ids"]
        model_inputs.append(input_data)

    print(f"[DEBUG] Preprocessing completed, total segments: {len(model_inputs)}")
    return model_inputs

# 文本生成并保存函数
def generate_and_save_results(rank, model, tokenizer, test_datasets, output_dir):
    print(f"[Rank {rank}] Generating summaries and saving results...")
    model.eval()
    for dataset_name, dataset in test_datasets.items():
        print(f"[Rank {rank}] Processing test set for {dataset_name}...")
        results = {"abstract": [], "output": []}

        for idx, example in enumerate(dataset):
            input_segments = split_long_text(example["text"], tokenizer)
            generated_summary = []

            # 对每个分段生成摘要
            for segment in input_segments:
                input_ids = torch.tensor(segment).unsqueeze(0).to(rank)
                with torch.no_grad():
                    generated_ids = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
                generated_summary.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

            # 拼接所有分段的摘要
            final_summary = " ".join(generated_summary)

            # 存储结果
            results["abstract"].append(example["abstract"])
            results["output"].append(final_summary)

            if idx % 10 == 0:
                print(f"[Rank {rank}] Processed {idx + 1}/{len(dataset)} samples for {dataset_name}.")

        # 保存为 .arrow 格式
        output_path = os.path.join(output_dir, "t5-base", dataset_name)
        os.makedirs(output_path, exist_ok=True)
        arrow_dataset = HFDataset.from_dict(results)
        arrow_dataset.save_to_disk(os.path.join(output_path, "test_results.arrow"))
        print(f"[Rank {rank}] Results saved for {dataset_name}.")

# 分布式训练的核心函数
def train(rank, world_size, data_path, datasets, output_dir, model_dir, epochs=3, batch_size=8, lr=5e-5):
    init_distributed_training(rank, world_size)

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    print(f"[Rank {rank}] Loaded tokenizer.")

    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print(f"[Rank {rank}] Model initialized and wrapped in DDP.")

    # 加载数据
    train_datasets, val_datasets, test_datasets = load_datasets(data_path, datasets)

    # 动态分段预处理数据
    print(f"[Rank {rank}] Preprocessing training and validation data...")
    train_data = sum([d.map(lambda x: preprocess_data(x, tokenizer), batched=True) for d in train_datasets], [])
    val_data = sum([d.map(lambda x: preprocess_data(x, tokenizer), batched=True) for d in val_datasets], [])
    print(f"[Rank {rank}] Preprocessing completed for training and validation data.")

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    print(f"[Rank {rank}] Optimizer initialized.")

    for epoch in range(epochs):
        print(f"[Rank {rank}] Starting epoch {epoch + 1}/{epochs}...")
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = torch.tensor(batch["input_ids"]).to(rank)
            attention_mask = torch.tensor(batch["attention_mask"]).to(rank)
            labels = torch.tensor(batch["labels"]).to(rank)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")

        # 验证模型
        if rank == 0:
            print(f"[Rank {rank}] Validating model...")
            model.eval()
            total_loss = 0
            for batch in val_loader:
                input_ids = torch.tensor(batch["input_ids"]).to(rank)
                attention_mask = torch.tensor(batch["attention_mask"]).to(rank)
                labels = torch.tensor(batch["labels"]).to(rank)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    total_loss += outputs.loss.item()

            print(f"[Rank {rank}] Validation Loss after Epoch {epoch + 1}: {total_loss / len(val_loader)}")

    # 保存模型
    if rank == 0:
        model_save_path = os.path.join(model_dir, "t5-base")
        os.makedirs(model_save_path, exist_ok=True)
        model.module.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"[Rank {rank}] Model saved to {model_save_path}.")

    # 生成并保存测试结果
    generate_and_save_results(rank, model.module, tokenizer, test_datasets, output_dir)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_path, "../data")
    output_dir = os.path.join(base_path, "../output")
    model_dir = os.path.join(base_path, "../models")
    datasets = ["cnn_daily_mail", "arxiv"]

    world_size = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.multiprocessing.spawn(train, args=(world_size, data_path, datasets, output_dir, model_dir), nprocs=world_size)
