# src Directory

This directory contains various Python scripts used for different NLP tasks in the group project. Below is a description of each script and its functionality.

## Scripts

### 1. `TextRank.py`

- **Description**: Implements the TextRank algorithm for text summarization.
- **Functionality**:
  - Preprocesses text by tokenizing sentences and words, removing stop words, and converting to lowercase.
  - Uses TF-IDF and cosine similarity to build a similarity matrix.
  - Ranks sentences using the PageRank algorithm and generates a summary.
  - Processes ArXiv and CNN/DailyMail datasets and saves the generated summaries.

### 2. `t5_prediction.py`

- **Description**: Generates text summaries using a fine-tuned T5 model.
- **Functionality**:
  - Disables TensorFlow to use PyTorch and sets up the GPU environment.
  - Loads a fine-tuned T5 model and tokenizer.
  - Preprocesses the input text and generates summaries for the ArXiv and CNN/DailyMail datasets.
  - Saves the generated summaries to the output directory.

### 3. `test.py`

- **Description**: Tests the fine-tuning and evaluation of the T5 model.
- **Functionality**:
  - Loads and preprocesses ArXiv and CNN/DailyMail datasets.
  - Fine-tunes the T5 model on the combined training dataset and evaluates on the validation dataset.
  - Generates predictions for the test datasets and saves the results.

### 4. `traditionalMethods.py`

- **Description**: Contains traditional methods for text processing and analysis.
- **Functionality**:
  - Implements keyword extraction, HITS, and TF-IDF + LSA methods.
  - Generates summaries of length 5 using the specified method.
  - Processes and saves results for ArXiv and CNN/DailyMail datasets.

### 5. `transformer.py`

- **Description**: Implements transformer-based models for NLP tasks.
- **Functionality**:
  - Loads and preprocesses ArXiv and CNN/DailyMail datasets.
  - Fine-tunes a T5 model and evaluates it on the validation dataset.
  - Generates predictions for the test datasets and saves the results.

## Usage

1. **TextRank Summarization**:
   - Run `TextRank.py` to generate text summaries using the TextRank algorithm.

2. **T5 Model Prediction**:
   - Run `t5_prediction.py` to generate text summaries using a fine-tuned T5 model.

3. **Model Testing**:
   - Run `test.py` to fine-tune the T5 model and generate predictions on test datasets.

4. **Traditional Methods Summarization**:
   - Run `traditionalMethods.py` to generate text summaries using keyword extraction, HITS, and TF-IDF + LSA methods.

5. **Transformer-Based Summarization**:
   - Run `transformer.py` to fine-tune and evaluate transformer-based models and generate predictions on test datasets.

Please ensure the necessary dependencies are installed and the required datasets are available in the specified directories before running the scripts.

# src 目录

该目录包含用于项目中不同 NLP 任务的各种 Python 脚本。以下是每个脚本的描述及其功能。

## 脚本

### 1. `TextRank.py`

- **描述**：实现用于文本摘要的 TextRank 算法。
- **功能**：
  - 通过分句和分词、去除停用词和小写化来预处理文本。
  - 使用 TF-IDF 和余弦相似度构建相似度矩阵。
  - 使用 PageRank 算法对句子进行排序并生成摘要。
  - 处理 ArXiv 和 CNN/DailyMail 数据集并保存生成的摘要。

### 2. `t5_prediction.py`

- **描述**：使用微调的 T5 模型生成文本摘要。
- **功能**：
  - 禁用 TensorFlow 以使用 PyTorch，并设置 GPU 环境。
  - 加载微调的 T5 模型和 tokenizer。
  - 预处理输入文本并为 ArXiv 和 CNN/DailyMail 数据集生成摘要。
  - 将生成的摘要保存到输出目录。

### 3. `test.py`

- **描述**：测试 T5 模型的微调和评估。
- **功能**：
  - 加载并预处理 ArXiv 和 CNN/DailyMail 数据集。
  - 在合并的训练数据集上微调 T5 模型，并在验证数据集上进行评估。
  - 生成测试数据集的预测结果并保存。

### 4. `traditionalMethods.py`

- **描述**：包含传统的文本处理和分析方法。
- **功能**：
  - 实现关键词提取、HITS 和 TF-IDF + LSA 方法。
  - 使用指定的方法生成长度为 5 的摘要。
  - 处理 ArXiv 和 CNN/DailyMail 数据集并保存结果。

### 5. `transformer.py`

- **描述**：实现基于 transformer 的模型用于 NLP 任务。
- **功能**：
  - 加载并预处理 ArXiv 和 CNN/DailyMail 数据集。
  - 微调 T5 模型并在验证数据集上进行评估。
  - 生成测试数据集的预测结果并保存。

## 使用说明

1. **TextRank 摘要生成**：
   - 运行 `TextRank.py` 使用 TextRank 算法生成文本摘要。

2. **T5 模型预测**：
   - 运行 `t5_prediction.py` 使用微调的 T5 模型生成文本摘要。

3. **模型测试**：
   - 运行 `test.py` 微调 T5 模型并生成测试数据集的预测结果。

4. **传统方法摘要生成**：
   - 运行 `traditionalMethods.py` 使用关键词提取、HITS 和 TF-IDF + LSA 方法生成文本摘要。

5. **基于 Transformer 的摘要生成**：
   - 运行 `transformer.py` 微调和评估基于 transformer 的模型，并生成测试数据集的预测结果。

请确保在运行脚本之前安装必要的依赖项，并在指定目录中提供所需的数据集。
