import os

# 禁用 TensorFlow，强制使用 PyTorch
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁用 TensorFlow 日志

# 指定可见的 GPU 设备，仅使用编号为 2 的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths to your directories
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data", "processed")
model_dir = os.path.join(current_dir, "../models", "t5_finetuned")
output_dir = os.path.join(current_dir, "../output", "t5_summary")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test datasets
arxiv_test = load_from_disk(os.path.join(data_dir, "ArXiv/test"))
cnn_test = load_from_disk(os.path.join(data_dir, "cnn_dailymail/test"))

# Preprocessing parameters
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = ["summarize: " + text for text in examples["text"]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding="max_length", truncation=True
    )
    return model_inputs

# Preprocess test datasets and set format to torch
arxiv_test_processed = arxiv_test.map(
    preprocess_function, batched=True, remove_columns=["text", "abstract"]
).with_format("torch")

cnn_test_processed = cnn_test.map(
    preprocess_function, batched=True, remove_columns=["text", "abstract"]
).with_format("torch")

def generate_predictions(processed_dataset, original_dataset, dataset_name):
    # Create DataLoader
    test_dataloader = DataLoader(
        processed_dataset,
        batch_size=64,  # 根据您的 GPU 内存调整批量大小
        shuffle=False,
    )

    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Generating predictions for {dataset_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_beams=4,          # 根据需要调整
                early_stopping=True,
            )

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_preds)

    # Save predictions along with abstracts
    results = Dataset.from_dict({
        "abstract": original_dataset["abstract"],
        "output": predictions,
    })

    # Save results
    save_path = os.path.join(output_dir, dataset_name)
    os.makedirs(save_path, exist_ok=True)
    results.save_to_disk(save_path)
    print(f"Predictions saved to {save_path}")


print("Is CUDA available:", torch.cuda.is_available())
print("Current device:", device)
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# Generate predictions for ArXiv test set
generate_predictions(arxiv_test_processed, arxiv_test, "ArXiv")

# Generate predictions for CNN/DailyMail test set
generate_predictions(cnn_test_processed, cnn_test, "cnn_dailymail")
