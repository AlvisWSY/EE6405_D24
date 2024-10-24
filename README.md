# EE6405_D24 
# Automatic Text Summarization

## Project Overview

This project aims to generate automatic summaries from a given text dataset (e.g., news articles) using both traditional and modern NLP methods. We will compare the performance of extractive summarization techniques, such as TF-IDF and TextRank, with abstract summarization methods based on transformer models like T5 and GPT.

The objective is to produce concise summaries while retaining the most important information from the original text. We will evaluate the effectiveness of each approach using metrics like BLEU and ROUGE and visualize the results for better comparison.

## Project Workflow

1. **Data Preprocessing**
   - Clean and preprocess the text data.
   - Sentence tokenization and removal of irrelevant information.
   - Dataset: [CNN/Daily Mail Dataset](https://www.kaggle.com/datasets) or other online news sources.

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

5. **Visualization and Display**
   - Visualize the comparison between original texts and generated summaries.
   - Present graphical comparisons of the summarization quality across models and methods.

## Key Tools and Technologies

- **Datasets**: CNN/Daily Mail dataset from Kaggle or other publicly available datasets.
- **Traditional Methods**: TF-IDF, TextRank.
- **Transformers**: 
   - T5 model (via Hugging Face Transformers)
   - GPT model (via OpenAI GPT or Hugging Face)
- **Metrics**: BLEU, ROUGE.
- **Python Libraries**:
   - `transformers` (Hugging Face for Transformer models)
   - `nltk` (for TextRank, preprocessing)
   - `scikit-learn` (for TF-IDF)
   - `matplotlib` or `seaborn` (for visualization)
   - `rouge_score` and `nltk.translate.bleu_score` (for evaluation metrics)

## Project Goals

- Build and implement both extractive and abstractive summarization methods.
- Compare the results of traditional techniques and transformer-based models.
- Evaluate the generated summaries using BLEU and ROUGE metrics.
- Visualize the performance and summarize findings in a final report.

---

## Team Responsibilities

1. **Data Preprocessing**: Clean and prepare text data for summarization.
2. **Traditional Method Implementation**: Implement and fine-tune extractive summarization using TF-IDF and TextRank.
3. **Transformer Model Implementation**: Develop abstractive summarization using T5 and GPT.
4. **Model Evaluation**: Compare the performance using BLEU, ROUGE, and other metrics.
5. **Visualization**: Present the comparison between different summarization methods and models with visual tools.

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
