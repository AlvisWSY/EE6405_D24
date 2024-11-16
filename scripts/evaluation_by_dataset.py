import numpy as np
import spacy
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
from datasets import load_from_disk
import os
import time
import nltk

nltk.download('wordnet')

# Path to the output directory containing multiple datasets
output_dir = "../results"


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

def visualize_scores(score_dict, dataset_name):
    plt.figure(figsize=(18, 10))
    methods = list(score_dict.keys())
    colors = plt.get_cmap("tab10").colors

    # BLEU Scores
    plt.subplot(2, 3, 1)
    for i, method in enumerate(methods):
        plt.hist(score_dict[method]['bleu'], bins=20, alpha=0.4, label=method, color=colors[i % len(colors)])
    plt.title(f'BLEU Score Distribution - {dataset_name}')
    plt.xlabel('BLEU Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # ROUGE-1 Scores
    plt.subplot(2, 3, 2)
    for i, method in enumerate(methods):
        plt.hist(score_dict[method]['rouge1'], bins=20, alpha=0.4, label=method, color=colors[i % len(colors)])
    plt.title(f'ROUGE-1 Score Distribution - {dataset_name}')
    plt.xlabel('ROUGE-1 F1 Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # ROUGE-2 Scores
    plt.subplot(2, 3, 3)
    for i, method in enumerate(methods):
        plt.hist(score_dict[method]['rouge2'], bins=20, alpha=0.4, label=method, color=colors[i % len(colors)])
    plt.title(f'ROUGE-2 Score Distribution - {dataset_name}')
    plt.xlabel('ROUGE-2 F1 Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # ROUGE-L Scores
    plt.subplot(2, 3, 4)
    for i, method in enumerate(methods):
        plt.hist(score_dict[method]['rougeL'], bins=20, alpha=0.4, label=method, color=colors[i % len(colors)])
    plt.title(f'ROUGE-L Score Distribution - {dataset_name}')
    plt.xlabel('ROUGE-L F1 Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # METEOR Scores
    plt.subplot(2, 3, 5)
    for i, method in enumerate(methods):
        plt.hist(score_dict[method]['meteor'], bins=20, alpha=0.4, label=method, color=colors[i % len(colors)])
    plt.title(f'METEOR Score Distribution - {dataset_name}')
    plt.xlabel('METEOR Score')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 3, 6)
    average_scores = {}
    for method in methods:
        bleu_avg = np.mean(score_dict[method]['bleu'])
        rouge1_avg = np.mean(score_dict[method]['rouge1'])
        rouge2_avg = np.mean(score_dict[method]['rouge2'])
        rougeL_avg = np.mean(score_dict[method]['rougeL'])
        meteor_avg = np.mean(score_dict[method]['meteor'])
        average_scores[method] = (bleu_avg + rouge1_avg + rouge2_avg + rougeL_avg + meteor_avg) / 5

    plt.bar(average_scores.keys(), average_scores.values(), color=plt.get_cmap("Paired").colors[:len(methods)])
    plt.title(f'Average Score Comparison - {dataset_name}')
    plt.xlabel('Method')
    plt.ylabel('Average Score')

    plt.tight_layout()
    output_evaluation_path = os.path.join("../results_evaluation", dataset_name)
    os.makedirs(os.path.dirname(output_evaluation_path), exist_ok=True)
    plt.savefig(f"{output_evaluation_path}.png")
    plt.show()

def main():
    start_time = time.time()

    # Loop through each subdirectory in the output directory
    for dataset_name in os.listdir(output_dir):
        all_scores_across_methods = []
        dataset_path = os.path.join(output_dir, dataset_name)
        if os.path.isdir(dataset_path):
            print(f"Evaluating dataset: {dataset_name}")

            # Load the dataset
            dataset = load_from_disk(dataset_path)
            score_dict = {}

            # Loop through each column in the dataset (excluding 't5_demo')
            for method in dataset.column_names:
                if method in ['t5_demo', 'abstract']:
                    continue
                print(f"Evaluating dataset method: {dataset_name}/{method}")

                generated_summaries = dataset[method]  # Summaries generated by the specific method
                ground_truth_summaries = dataset['abstract']  # Ground truth summaries

                # Calculate scores for each method
                bleu_scores = calculate_bleu(generated_summaries, ground_truth_summaries)
                rouge1_scores, rouge2_scores, rougeL_scores = calculate_rouge(generated_summaries,
                                                                              ground_truth_summaries)
                meteor_scores = calculate_meteor(generated_summaries, ground_truth_summaries)

                # Calculate average score for each row
                scores = [
                    (bleu + rouge1 + rouge2 + rougeL + meteor) / 5
                    for bleu, rouge1, rouge2, rougeL, meteor in
                    zip(bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, meteor_scores)
                ]

                # 如果 all_scores_across_methods 是空的，初始化为 scores
                if not all_scores_across_methods:
                    all_scores_across_methods = [[score] for score in scores]
                else:
                    # 如果已有内容，将新评分添加到每个数据点的列表中
                    for i, score in enumerate(scores):
                        all_scores_across_methods[i].append(score)

                score_dict[method] = {
                    'bleu': bleu_scores,
                    'rouge1': rouge1_scores,
                    'rouge2': rouge2_scores,
                    'rougeL': rougeL_scores,
                    'meteor': meteor_scores
                }

            # Visualize scores for the current dataset
            visualize_scores(score_dict, f"{dataset_name}")

            average_scores = [sum(scores) / len(scores) for scores in all_scores_across_methods]

            # Add the average score as a new column to the dataset
            dataset = dataset.add_column("score", average_scores)
            new_dataset_path = os.path.join("../results_evaluation/scored_dataset", dataset_name)
            os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)
            dataset.save_to_disk(new_dataset_path)

    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")


# Run the main function
if __name__ == "__main__":
    main()