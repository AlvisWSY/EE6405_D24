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

# 下载必要的 NLTK 数据
nltk.download("punkt")
nltk.download("stopwords")

def preprocess_text(text):
    """
    对文本进行预处理，包括分句、分词、去停词、小写化。
    """
    # 分句
    sentences = sent_tokenize(text)

    # 分词、去停词、小写化
    stop_words = set(stopwords.words("english"))
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())  # 小写化
        words = [word for word in words if word.isalnum() and word not in stop_words]  # 去停词和非字母数字
        preprocessed_sentences.append(" ".join(words))
    return sentences, preprocessed_sentences

def textrank_summary(text, num_sentences=5):
    """
    使用 TextRank 生成摘要。
    :param text: 输入文本
    :param num_sentences: 摘要中包含的句子数量
    """
    # 1. 文本预处理
    original_sentences, preprocessed_sentences = preprocess_text(text)

    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)  # 如果句子数量不足，直接返回原句子

    # 2. 计算句子之间的稀疏相似性矩阵
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 稀疏化相似性矩阵
    threshold = 0.1
    sparse_matrix = csr_matrix(similarity_matrix * (similarity_matrix > threshold))

    # 3. 构建图并计算句子重要性
    try:
        nx_graph = nx.from_scipy_sparse_array(sparse_matrix)
    except AttributeError:
        nx_graph = nx.from_scipy_sparse_matrix(sparse_matrix)

    scores = nx.pagerank(nx_graph)

    # 4. 根据重要性排序句子并生成摘要
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:num_sentences]])

    return summary

def process_sample(sample):
    """
    处理单个样本：生成摘要并保留原始数据。
    """
    text = sample["text"]
    abstract = sample["abstract"]
    summary = textrank_summary(text, num_sentences=5)
    return {"text": text, "abstract": abstract, "generated_summary": summary}

# ArXiv 数据集
# 获取当前脚本的实际路径
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../data/Processed/ArXiv/")  # 数据集的相对路径
output_path = os.path.join(script_dir, "../output/TextRank/ArXiv/")  # 输出路径

# 加载 Hugging Face 数据集
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]

# 并行处理数据集
with Pool(processes=4) as pool:  # 使用4个进程并行处理
    processed_samples = pool.map(process_sample, test_dataset)

# 将结果保存为新的 Hugging Face 数据集
processed_dataset = Dataset.from_dict({
    "abstract": [sample["abstract"] for sample in processed_samples],
    "output": [sample["generated_summary"] for sample in processed_samples]
})

# 保存为 Arrow 格式
processed_dataset.save_to_disk(output_path)

# 打印示例
print("数据集已保存到:", output_path)
print("示例原文:", processed_dataset[0]["abstract"])
print("示例摘要:", processed_dataset[0]["output"])


# CNN数据集
# 获取当前脚本的实际路径
dataset_path = os.path.join(script_dir, "../data/Processed/cnn_dailymail/")  # 数据集的相对路径
output_path = os.path.join(script_dir, "../output/TextRank/cnn_dailymail/")  # 输出路径

# 加载 Hugging Face 数据集
dataset = load_from_disk(dataset_path)
test_dataset = dataset["test"]

# 并行处理数据集
with Pool(processes=4) as pool:  # 使用4个进程并行处理
    processed_samples = pool.map(process_sample, test_dataset)

# 将结果保存为新的 Hugging Face 数据集
processed_dataset = Dataset.from_dict({
    "abstract": [sample["abstract"] for sample in processed_samples],
    "output": [sample["generated_summary"] for sample in processed_samples]
})

# 保存为 Arrow 格式
processed_dataset.save_to_disk(output_path)

# 打印示例
print("数据集已保存到:", output_path)
print("示例原文:", processed_dataset[0]["abstract"])
print("示例摘要:", processed_dataset[0]["output"])