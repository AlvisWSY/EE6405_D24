import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.sparse import csr_matrix
from datasets import load_from_disk, Dataset, DatasetDict
import os
from multiprocessing import Pool

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")

def preprocess_text(text):
    """
    Preprocess the input text, including sentence splitting, tokenization, stopword removal, and lowercasing.
    """
    # Split into sentences
    sentences = sent_tokenize(text)

    # Tokenization, stopword removal, and lowercasing
    stop_words = set(stopwords.words("english"))
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())  # Convert to lowercase
        words = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric tokens
        preprocessed_sentences.append(" ".join(words))
    return sentences, preprocessed_sentences

def textrank_summary(text, num_sentences=5):
    """
    Generate a summary using TextRank.
    :param text: Input text
    :param num_sentences: Number of sentences to include in the summary
    """
    # 1. Preprocess the text
    original_sentences, preprocessed_sentences = preprocess_text(text)

    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)  # If there are not enough sentences, return the original sentences

    # 2. Compute the sparse similarity matrix between sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Sparsify the similarity matrix
    threshold = 0.1
    sparse_matrix = csr_matrix(similarity_matrix * (similarity_matrix > threshold))

    # 3. Build a graph and compute sentence importance scores
    try:
        nx_graph = nx.from_scipy_sparse_array(sparse_matrix)
    except AttributeError:
        nx_graph = nx.from_scipy_sparse_matrix(sparse_matrix)

    scores = nx.pagerank(nx_graph)

    # 4. Rank sentences by importance and generate the summary
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])

    return summary

def process_sample(sample):
    """
    Process a single sample: generate a summary while retaining the original data.
    """
    text = sample["text"]
    abstract = sample["abstract"]
    summary = textrank_summary(text, num_sentences=5)
    return {"text": text, "abstract": abstract, "generated_summary": summary}

# ArXiv Dataset
# Get the actual path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../data/processed/ArXiv/")  # Relative path to the dataset
output_path = os.path.join(script_dir, "../output/TextRank/ArXiv/")  # Output path

# Load the Hugging Face dataset
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]

# Process the dataset in parallel
with Pool(processes=4) as pool:  # Use 4 processes for parallel processing
    processed_samples = pool.map(process_sample, test_dataset)

# Save the results as a new Hugging Face dataset
processed_dataset = Dataset.from_dict({
    "abstract": [sample["abstract"] for sample in processed_samples],
    "output": [sample["generated_summary"] for sample in processed_samples]
})

# Save as Arrow format
processed_dataset.save_to_disk(output_path)

# Print an example
print("Dataset saved to:", output_path)
print("Example original text:", processed_dataset[0]["abstract"])
print("Example summary:", processed_dataset[0]["output"])


# CNN Dataset
# Get the actual path of the current script
dataset_path = os.path.join(script_dir, "../data/processed/cnn_dailymail/")  # Relative path to the dataset
output_path = os.path.join(script_dir, "../output/TextRank/cnn_dailymail/")  # Output path

# Load the Hugging Face dataset
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]

# Process the dataset in parallel
with Pool(processes=4) as pool:  # Use 4 processes for parallel processing
    processed_samples = pool.map(process_sample, test_dataset)

# Save the results as a new Hugging Face dataset
processed_dataset = Dataset.from_dict({
    "abstract": [sample["abstract"] for sample in processed_samples],
    "output": [sample["generated_summary"] for sample in processed_samples]
})

# Save as Arrow format
processed_dataset.save_to_disk(output_path)

# Print an example
print("Dataset saved to:", output_path)
print("Example original text:", processed_dataset[0]["abstract"])
print("Example summary:", processed_dataset[0]["output"])
