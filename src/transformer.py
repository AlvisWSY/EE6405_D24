import os
from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import multiprocessing
import torch

# Step 1: Dynamically define directories
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use relative paths to construct directories
data_dir = os.path.join(current_dir, "../data", "processed")
output_model_dir = os.path.join(current_dir, "../models", "t5_finetuned")
output_dir = os.path.join(current_dir, "../output", "t5_summary")
cache_dir = os.path.join(current_dir, "../tmp")  # Cache directory

# Set cache directory
os.environ["HF_DATASETS_CACHE"] = cache_dir

# Ensure necessary directories exist
os.makedirs(output_model_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Print directory paths to confirm
print(f"Data directory: {data_dir}")
print(f"Model output directory: {output_model_dir}")
print(f"Test results output directory: {output_dir}")
print(f"Cache directory: {cache_dir}")

# **Specify available GPUs**
# You can use GPU 2 and 3
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Step 2: Load datasets
def load_and_print_dataset(path, split_name):
    dataset = load_from_disk(path)
    print(f"Loaded {split_name}: {len(dataset)} samples")
    return dataset

# Load ArXiv dataset
arxiv_train = load_and_print_dataset(os.path.join(data_dir, "ArXiv/train"), "ArXiv train")
arxiv_validation = load_and_print_dataset(os.path.join(data_dir, "ArXiv/validation"), "ArXiv validation")
arxiv_test = load_and_print_dataset(os.path.join(data_dir, "ArXiv/test"), "ArXiv test")

# Load CNN/DailyMail dataset
cnn_train = load_and_print_dataset(os.path.join(data_dir, "cnn_dailymail/train"), "CNN train")
cnn_validation = load_and_print_dataset(os.path.join(data_dir, "cnn_dailymail/validation"), "CNN validation")
cnn_test = load_and_print_dataset(os.path.join(data_dir, "cnn_dailymail/test"), "CNN test")

# Combine training and validation datasets
train_dataset = concatenate_datasets([arxiv_train, cnn_train])
validation_dataset = concatenate_datasets([arxiv_validation, cnn_validation])
print(f"Combined train dataset size: {len(train_dataset)} samples")
print(f"Combined validation dataset size: {len(validation_dataset)} samples")

# Step 3: Load the pre-trained model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 4: Preprocess data
max_input_length = 512
max_target_length = 128

# Define the preprocessing function
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

# Optimize data preprocessing
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

# Adjust the number of threads and batch size
num_proc = 16  # Adjust based on the maximum number of threads available
batch_size = 10000  # Adjust based on memory availability

train_dataset = preprocess_dataset(train_dataset, num_proc=num_proc, batch_size=batch_size)
validation_dataset = preprocess_dataset(validation_dataset, num_proc=num_proc, batch_size=batch_size)

# Step 5: Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_model_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Increase batch size to fully utilize GPUs
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    predict_with_generate=True,
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,  # Adjust based on the number of threads available
    # **Enable distributed training**
    # Set fp16_backend to "amp"
    fp16_backend="amp",
    # Enable gradient checkpointing if needed to save memory
    # gradient_checkpointing=True,
)

# Step 6: Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# **Ensure that multiple GPUs can be used for training**
# Trainer will automatically detect and utilize multiple GPUs without extra configuration

# Step 7: Fine-tune the model
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

# Save the fine-tuned model
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model saved to {output_model_dir}")

# Step 8: Inference on the test dataset and save results
def generate_predictions(test_dataset, dataset_name):
    processed_test = test_dataset.map(preprocess_function, batched=True, remove_columns=["text", "abstract"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Adjust batch size
    batch_size = 16  # Adjust based on GPU memory

    # Create a data loader
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

# Inference on ArXiv and CNN/DailyMail test datasets
generate_predictions(arxiv_test, "ArXiv")
generate_predictions(cnn_test, "cnn_dailymail")
