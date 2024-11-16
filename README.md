# Automatic Text Summarization (EE6405 - Group D24)

## Project Overview

This project is part of the course **EE6405**, developed by **Group D24**. The goal is to generate automatic summaries from various types of datasets, including news articles, research papers and their abstracts. We explored and compared the performance of different summarization methods: traditional extractive techniques (such as TF-IDF and TextRank) and modern abstractive methods based on transformer model (such as T5).

Additionally, we built a **Graphical User Interface (GUI)** that allows users to choose between different summarization methods. This tool will allow users to upload datasets, select summarization approaches, and visualize the results.

## Project Workflow

1. **Data Preprocessing**
   - Clean and preprocess the text data.
   - Sentence tokenization and removal of irrelevant information.
   - Datasets:
     - [CNN/Daily Mail Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail) for news articles.
     - [ArXiv](https://huggingface.co/datasets/ccdv/arxiv-summarization) for research papers.

2. **Traditional Methods for Extractive Summarization**
   - **Keyword-based Summarization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to calculate the importance of words in sentences.
   - **Graph-based HITS Summarization**: Applies the HITS (Hyperlink-Induced Topic Search) algorithm to rank sentences based on authority scores.
   - **Latent Semantic Analysis (LSA) Summarization**: Applies TF-IDF to represent sentences as vectors. Reduces dimensionality using Truncated Singular Value Decomposition (SVD) and ranks sentences based on their significance in the reduced space.
   - **TextRank**: Apply graph-based TextRank for extractive summarization.

3. **Transformer Models for Abstract Summarization**
   - **T5 (Text-to-Text Transfer Transformer)**: Finetune pre-trained T5-base model for atrical summarization.

4. **Evaluation and Comparison**
   - Evaluate the summarization results based on both traditional and transformer-based methods
   - Compare the scores and performance of CNN/Daily Mail Dataset and ArXiv Dataset.
   - Metrics for evaluation:
     - **BLEU**: Evaluate overlap between generated summaries and reference summaries.
     - **ROUGE**: Measure recall-oriented evaluation of generated summaries.
     - **METEOR**: Evaluate the sequence of words in the output sentence.

5. **GUI Development**
   - Develop a user-friendly GUI to allow users to:
     - Upload datasets (news articles, research papers, etc.).
     - Select summarization methods (TF-IDF, TextRank, T5, GPT).
     - View and compare the generated summaries.

6. **Visualization and Display**
   - Visualize the comparison between original texts and generated summaries.
   - Present graphical comparisons of summarization quality across different methods.

## Project Structure

The project structure is organized as follows:

- **`data/`**               Directory for storing datasets (datasets are not uploaded but can be downloaded via scripts)
- **`docs/`**               Documentation and project notes
- **`gui/`**                GUI related code and assets
- **`models/`**             Directory for storing trained models
- **`results/`**            Output results such as generated summaries
- **`scripts/`**            Automation scripts (e.g., data download, model training)
- **`setup/`**              Directory for setup
- **`src/`**                Source code for summarization methods
- **`README.md`**           Project overview and setup guide
- **`environment.yml`**     Python dependencies

## Getting Started

### Clone the repository:
```bash
git clone https://github.com/AlvisWSY/EE6405_D24.git
```

### Initialize the project:
```bash
python initialize.py
```
This script initializes the project by setting up the conda environment, creating the necessary directory structures, and downloading datasets to the `data/` directory.

# 自动文本摘要生成 (EE6405 - Group D24)

## 项目概述

本项目是 EE6405 课程的一部分，由 D24 小组开发。目标是对多种类型的数据集（包括新闻文章、研究论文及其摘要）进行自动摘要生成。我们探索并比较了不同的摘要生成方法：传统的提取式技术（如 TF-IDF 和 TextRank）和基于 transformer 模型的现代抽象式方法（如 T5）。

此外，我们还开发了一个**图形用户界面（GUI）**，允许用户在不同的摘要生成方法之间进行选择。这个工具允许用户上传数据集、选择摘要方法，并可视化结果。

## 项目工作流程

1. **数据预处理**
   - 清理和预处理文本数据。
   - 分句处理并去除无关信息。
   - 数据集：
     - [CNN/Daily Mail 数据集](https://huggingface.co/datasets/abisee/cnn_dailymail) 适用于新闻文章。
     - [ArXiv 数据集](https://huggingface.co/datasets/ccdv/arxiv-summarization) 适用于研究论文。

2. **传统方法的提取式摘要生成**
   - **基于关键词的摘要生成**：使用 TF-IDF（词频-逆文档频率）计算句子中词的重要性。
   - **基于图的 HITS 摘要生成**：应用 HITS（超链接诱导主题搜索）算法根据权威评分对句子进行排序。
   - **潜在语义分析（LSA）摘要生成**：应用 TF-IDF 将句子表示为向量。使用截断奇异值分解（SVD）降维，并根据降维空间中的重要性对句子进行排序。
   - **TextRank**：使用基于图的 TextRank 进行提取式摘要生成。

3. **基于 Transformer 模型的抽象摘要生成**
   - **T5（文本到文本转换 Transformer）**：根据任务使用数据调整预训练的T5-base模型。

4. **评估与比较**
   - 评估基于传统提取式方法和基于 transformer 模型生成的摘要结果。
   - 比较 CNN/Daily Mail 数据集和 ArXiv 数据集的得分和效果。
   - 评估指标：
     - **BLEU**：评估生成的摘要与参考摘要之间的重叠。
     - **ROUGE**：衡量生成摘要的回忆率。
     - **METEOR**：评估输出句子中的单词序列。

5. **GUI 开发**
   - 开发用户友好的 GUI，允许用户：
     - 上传数据集（新闻文章、研究论文等）。
     - 选择摘要方法（TF-IDF、TextRank、T5、GPT）。
     - 查看并比较生成的摘要。

6. **可视化与展示**
   - 可视化比较原文与生成的摘要。
   - 图形化展示不同方法的摘要质量。

## 项目结构

项目的文件结构如下：

- **`data/`**               用于存储数据集的目录（数据集不会上传，但可通过脚本下载）
- **`docs/`**               项目文档和笔记
- **`gui/`**                GUI 相关代码和资源
- **`models/`**             用于存储训练好的模型的目录
- **`results/`**            输出结果如生成的摘要
- **`scripts/`**            自动化脚本（如数据下载、模型训练）
- **`setup/`**              用于设置的目录
- **`src/`**                摘要生成方法的源代码
- **`README.md`**           项目概述与设置指南
- **`environment.yml`**     Python 依赖库

## 快速开始

### 克隆项目：
```bash
git clone https://github.com/AlvisWSY/EE6405_D24.git
```

### 初始化项目：
```bash
python initialize.py
```
这个脚本会通过设置 conda 环境，创建必要的目录结构，并将数据集下载到 `data/` 目录，来初始化项目。
