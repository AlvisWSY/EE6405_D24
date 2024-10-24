# Automatic Text Summarization (EE6405 - Group D24)

## Project Overview

This project is part of the course **EE6405**, developed by **Group D24**. The goal is to generate automatic summaries from various types of datasets, including news articles, research papers and their abstracts, and potentially another type of article-summary dataset. We will explore and compare the performance of different summarization methods: traditional extractive techniques (such as TF-IDF and TextRank) and modern abstractive methods based on transformer models (such as T5 and GPT).

Additionally, we aim to build a **Graphical User Interface (GUI)** that allows users to choose between different summarization methods. This tool will allow users to upload datasets, select summarization approaches, and visualize the results.

## Project Workflow

1. **Data Preprocessing**
   - Clean and preprocess the text data.
   - Sentence tokenization and removal of irrelevant information.
   - Datasets:
     - [CNN/Daily Mail Dataset](https://www.kaggle.com/datasets) for news articles.
     - Research papers and their abstracts (e.g., from arXiv or other open repositories).
     - A third dataset (yet to be determined) involving article-summary pairs.

2. **Traditional Methods for Extractive Summarization**
   - **TF-IDF**: Implement term frequency-inverse document frequency to extract key sentences.
   - **TextRank**: Apply graph-based TextRank for extractive summarization.

3. **Transformer Models for Abstract Summarization**
   - Implement abstractive summarization using modern transformer models:
     - **T5 (Text-to-Text Transfer Transformer)**: Use for generating abstract summaries.
     - **GPT (Generative Pre-trained Transformer)**: Generate summaries based on pre-trained models.

4. **Evaluation and Comparison**
   - Compare both traditional and transformer-based summarization methods.
   - Metrics for evaluation:
     - **BLEU**: Evaluate overlap between generated summaries and reference summaries.
     - **ROUGE**: Measure recall-oriented evaluation of generated summaries.

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

- **data/**               # Directory for storing datasets (datasets are not uploaded but can be downloaded via scripts)
- **src/**                # Source code for summarization methods
- **models/**             # Directory for storing trained models
- **setup/**              # Directory for setup
- **scripts/**            # Automation scripts (e.g., data download, model training)
- **results/**            # Output results such as generated summaries
- **gui/**                # GUI related code and assets
- **docs/**               # Documentation and project notes
- **README.md**           # Project overview and setup guide
- **enviroment.yml**    # Python dependencies

## Getting Started

### Clone the repository:
```bash
git clone https://github.com/your-repo/automatic-text-summarization.git
```

### Initialize the project:
```bash
python initialize.py
```

# 自动文本摘要生成 (EE6405 - Group D24)

## 项目概述
本项目是 EE6405 课程的一部分，由 D24 小组开发。目标是对多种类型的数据集（包括新闻文章、研究论文及其摘要，以及可能的其他类型的文章摘要数据集）进行自动摘要生成。我们将探索并比较不同的摘要生成方法：传统的提取式方法（如 TF-IDF 和 TextRank）与基于 transformer 模型的抽象式方法（如 T5 和 GPT）。

此外，我们将开发一个用户友好的图形用户界面（GUI），允许用户在不同的摘要生成方法之间进行选择。该工具将允许用户上传数据集、选择摘要方法，并查看生成的结果。

## 项目工作流程

### 数据预处理
- 清理和预处理文本数据。
- 进行分句处理，并去除无关信息。
- **数据集：**
  - CNN/Daily Mail 数据集 适用于新闻文章。
  - 学术论文及其摘要（例如来自 arXiv 或其他公开仓库）。
  - 第三类数据集（尚未确定），可能包含文章-摘要对。

### 提取式摘要生成的传统方法
- **TF-IDF：** 实现基于词频-逆文档频率的关键句提取。
- **TextRank：** 使用基于图的 TextRank 算法进行提取式摘要生成。

### 基于 Transformer 模型的抽象摘要生成
- 使用现代 transformer 模型进行抽象摘要生成：
  - **T5 (Text-to-Text Transfer Transformer)：** 生成抽象式摘要。
  - **GPT (Generative Pre-trained Transformer)：** 基于预训练模型生成摘要。

### 模型评价与比较
- 比较传统方法和 transformer 模型生成的摘要。
- **评价指标：**
  - **BLEU：** 评估生成摘要与参考摘要之间的重叠程度。
  - **ROUGE：** 衡量生成摘要的回忆率。

### GUI 开发
- 开发用户友好的 GUI，允许用户：
  - 上传数据集（新闻文章、学术论文等）。
  - 选择摘要方法（TF-IDF、TextRank、T5、GPT）。
  - 查看并比较生成的摘要。

### 可视化与展示
- 可视化展示原文与生成摘要的对比。
- 图形化比较不同方法的摘要质量。

## 项目结构
项目的文件结构如下：
- **data/** # 用于存储数据集的目录（数据集不会上传，但可通过脚本下载）
- **src/** # 摘要生成方法的源代码
- **models/** # 用于存储训练好的模型的目录
- **setup/** # 用于setup的脚本
- **scripts/** # 自动化脚本（如数据下载、模型训练）
- **results/** # 输出结果（例如生成的摘要）
- **gui/** # GUI 相关代码和资源
- **docs/** # 项目文档和笔记
- **README.md** # 项目概述与设置指南
- **enviroment.yml** # Python 依赖库

## 快速开始

### 克隆项目：
```bash
git clone https://github.com/your-repo/automatic-text-summarization.git
```

### 初始化项目：
```bash
python initialize.py
```
