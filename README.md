# Automatic Text Summarization (EE6405 - Group D24)

## Project Overview

This project is part of the course **EE6405**, developed by **Group D24**. The goal is to generate automatic summaries from various types of datasets, including news articles, research papers and their abstracts, and potentially another type of article-summary dataset. We will explore and compare the performance of different summarization methods: traditional extractive techniques (such as TF-IDF and TextRank) and modern abstractive methods based on transformer models (such as T5 and GPT).

The project also includes building a **Graphical User Interface (GUI)**, allowing users to choose between different summarization methods. This will turn the experiment into a usable application where users can select a summarization approach and dataset type.

## Project Workflow

1. **Data Preprocessing**
   - Clean and preprocess the text data.
   - Sentence tokenization and removal of irrelevant information.
   - Datasets:
     - [CNN/Daily Mail Dataset](https://www.kaggle.com/datasets) for news articles.
     - Research papers and their abstracts (e.g., from arXiv or other open repositories).
     - A third dataset (yet to be determined) potentially involving article-summary pairs.

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

## Key Tools and Technologies

- **Datasets**:
   - CNN/Daily Mail dataset for news articles.
   - Research papers and abstracts from publicly available sources (arXiv, etc.).
   - A third dataset (yet to be determined).
- **Traditional Methods**: TF-IDF, TextRank.
- **Transformers**:
   - T5 model (via Hugging Face Transformers)
   - GPT model (via OpenAI GPT or Hugging Face)
- **GUI**: Frameworks like `Tkinter`, `PyQt`, or `Flask` (for web-based GUI).
- **Metrics**: BLEU, ROUGE.
- **Python Libraries**:
   - `transformers` (Hugging Face for Transformer models)
   - `nltk` (for TextRank, preprocessing)
   - `scikit-learn` (for TF-IDF)
   - `matplotlib` or `seaborn` (for visualization)
   - `rouge_score` and `nltk.translate.bleu_score` (for evaluation metrics)
   - `Tkinter`, `PyQt`, or `Flask` (for GUI development)

## Project Goals

- Build and implement both extractive and abstractive summarization methods for multiple datasets.
- Compare the results of traditional techniques and transformer-based models.
- Develop a GUI for users to interact with the summarization methods.
- Evaluate the generated summaries using BLEU and ROUGE metrics.
- Visualize the performance and summarize findings in a final report.

---

## Team Responsibilities

1. **Data Preprocessing**: Clean and prepare different types of text data for summarization.
2. **Traditional Method Implementation**: Implement and fine-tune extractive summarization using TF-IDF and TextRank.
3. **Transformer Model Implementation**: Develop abstractive summarization using T5 and GPT.
4. **GUI Development**: Create a user-friendly interface to let users choose between summarization methods.
5. **Model Evaluation**: Compare the performance using BLEU, ROUGE, and other metrics.
6. **Visualization**: Present the comparison between different summarization methods and models with visual tools.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/automatic-text-summarization.git
   ```
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the pipeline**:
   ```bash
   python main.py
   ```
