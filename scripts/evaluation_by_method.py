import spacy
from rouge_score import rouge_scorer
import pandas as pd
import pyarrow as pa
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from datasets import load_from_disk
import os
import time
import numpy as np
import nltk

nltk.download('wordnet')

# Path to the output directory containing multiple datasets
output_dir = "../output"


# Evaluation Functions
# BLEU score calculation
def calculate_bleu(generated, ground_truth):
    smoothie = SmoothingFunction().method4  # Smoothing for BLEU to handle short texts
    bleu_scores = []
    total = len(generated)
    for i, (gen, gt) in enumerate(zip(generated, ground_truth)):
        score = sentence_bleu([gt.split()], gen.split(), smoothing_function=smoothie)
        bleu_scores.append(score)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} BLEU scores")
    return bleu_scores


# ROUGE score calculation
def calculate_rouge(generated, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    total = len(generated)
    for i, (gen, gt) in enumerate(zip(generated, ground_truth)):
        scores = scorer.score(gt, gen)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} ROUGE scores")
    return rouge1, rouge2, rougeL


# METEOR score calculation using SpaCy tokenizer
nlp = spacy.load('en_core_web_sm')


def calculate_meteor(generated, ground_truth):
    meteor_scores = []
    total = len(generated)
    for i, (gen, gt) in enumerate(zip(generated, ground_truth)):
        score = meteor_score([list(map(str, nlp(gt)))], list(map(str, nlp(gen))))
        meteor_scores.append(score)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} METEOR scores")
    return meteor_scores


# Visualization
def visualize_scores(bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, meteor_scores, dataset_name):
    # Plotting BLEU, ROUGE, and METEOR scores for comparison
    plt.figure(figsize=(18, 10))

    # BLEU Scores
    plt.subplot(2, 3, 1)
    plt.hist(bleu_scores, bins=10, color='skyblue')
    plt.title(f'BLEU Score Distribution - {dataset_name}')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(bleu_scores), color='r', linestyle='dashed', linewidth=1,
                label=f'Mean: {np.mean(bleu_scores):.2f}')
    plt.axvline(np.median(bleu_scores), color='g', linestyle='dotted', linewidth=1,
                label=f'Median: {np.median(bleu_scores):.2f}')
    plt.legend()

    # ROUGE-1 Scores
    plt.subplot(2, 3, 2)
    plt.hist(rouge1_scores, bins=10, color='lightgreen')
    plt.title(f'ROUGE-1 Score Distribution - {dataset_name}')
    plt.xlabel('ROUGE-1 F1 Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(rouge1_scores), color='r', linestyle='dashed', linewidth=1,
                label=f'Mean: {np.mean(rouge1_scores):.2f}')
    plt.axvline(np.median(rouge1_scores), color='g', linestyle='dotted', linewidth=1,
                label=f'Median: {np.median(rouge1_scores):.2f}')
    plt.legend()

    # ROUGE-2 Scores
    plt.subplot(2, 3, 3)
    plt.hist(rouge2_scores, bins=10, color='salmon')
    plt.title(f'ROUGE-2 Score Distribution - {dataset_name}')
    plt.xlabel('ROUGE-2 F1 Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(rouge2_scores), color='r', linestyle='dashed', linewidth=1,
                label=f'Mean: {np.mean(rouge2_scores):.2f}')
    plt.axvline(np.median(rouge2_scores), color='g', linestyle='dotted', linewidth=1,
                label=f'Median: {np.median(rouge2_scores):.2f}')
    plt.legend()

    # ROUGE-L Scores
    plt.subplot(2, 3, 4)
    plt.hist(rougeL_scores, bins=10, color='plum')
    plt.title(f'ROUGE-L Score Distribution - {dataset_name}')
    plt.xlabel('ROUGE-L F1 Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(rougeL_scores), color='r', linestyle='dashed', linewidth=1,
                label=f'Mean: {np.mean(rougeL_scores):.2f}')
    plt.axvline(np.median(rougeL_scores), color='g', linestyle='dotted', linewidth=1,
                label=f'Median: {np.median(rougeL_scores):.2f}')
    plt.legend()

    # METEOR Scores
    plt.subplot(2, 3, 5)
    plt.hist(meteor_scores, bins=10, color='gold')
    plt.title(f'METEOR Score Distribution - {dataset_name}')
    plt.xlabel('METEOR Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(meteor_scores), color='r', linestyle='dashed', linewidth=1,
                label=f'Mean: {np.mean(meteor_scores):.2f}')
    plt.axvline(np.median(meteor_scores), color='g', linestyle='dotted', linewidth=1,
                label=f'Median: {np.median(meteor_scores):.2f}')
    plt.legend()

    plt.tight_layout()
    output_evaluation_path = os.path.join("../output_evaluation", dataset_name)
    os.makedirs(os.path.dirname(output_evaluation_path), exist_ok=True)
    plt.savefig(f"{output_evaluation_path}.png")
    plt.show()


# Main function to evaluate and visualize all datasets
def main():
    start_time = time.time()

    # Loop through each subdirectory in the output directory
    for method_dir in os.listdir(output_dir):
        method_path = os.path.join(output_dir, method_dir)
        if os.path.isdir(method_path):
            for dataset_name in os.listdir(method_path):
                dataset_path = os.path.join(method_path, dataset_name)
                if os.path.isdir(dataset_path):
                    print(f"Evaluating dataset: {method_dir}/{dataset_name}")

                    # Load the dataset
                    dataset = load_from_disk(dataset_path)
                    generated_summaries = dataset["output"]  # Column name in the .arrow file
                    ground_truth_summaries = dataset["abstract"]  # Column name for ground truth

                    # Calculate scores
                    bleu_scores = calculate_bleu(generated_summaries, ground_truth_summaries)
                    rouge1_scores, rouge2_scores, rougeL_scores = calculate_rouge(generated_summaries,
                                                                                  ground_truth_summaries)
                    meteor_scores = calculate_meteor(generated_summaries, ground_truth_summaries)

                    # Visualize scores
                    visualize_scores(bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, meteor_scores,
                                     f"{method_dir}/{dataset_name}")

    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")


# Run the main function
if __name__ == "__main__":
    main()
