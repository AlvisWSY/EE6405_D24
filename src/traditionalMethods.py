import os
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import networkx as nx
import nltk

# 下载 NLTK 数据
nltk.download("punkt")
nltk.download("stopwords")

# 文本预处理函数
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    sentences = sent_tokenize(text)
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(" ".join(words))
    return sentences, preprocessed_sentences

# 方法 1: 关键词提取
def keyword_summary(text, num_sentences=5):
    original_sentences, preprocessed_sentences = preprocess_text(text)
    if not preprocessed_sentences:
        return "No content to summarize."
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    scores = tfidf_matrix.sum(axis=1).A.flatten()  # 稀疏矩阵操作
    ranked_sentences = sorted(((scores[i], original_sentences[i]) for i in range(len(original_sentences))), reverse=True)
    return " ".join(s for _, s in ranked_sentences[:num_sentences])

# 方法 2: HITS
def hits_summary(text, num_sentences=5):
    original_sentences, preprocessed_sentences = preprocess_text(text)
    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)
    if not preprocessed_sentences:
        return "No content to summarize."
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, dense_output=False)  # 稀疏相似性矩阵
    sparse_matrix = csr_matrix(similarity_matrix)  # 保持稀疏矩阵
    nx_graph = nx.from_scipy_sparse_array(sparse_matrix)
    hubs, authorities = nx.hits(nx_graph, max_iter=100, tol=1e-4)
    ranked_sentences = sorted(((authorities[i], original_sentences[i]) for i in range(len(original_sentences))), reverse=True)
    return " ".join(s for _, s in ranked_sentences[:num_sentences])

# 方法 3: TF-IDF + LSA
def lsa_summary(text, num_sentences=5):
    original_sentences, preprocessed_sentences = preprocess_text(text)
    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)
    if not preprocessed_sentences:
        return "No content to summarize."
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    if tfidf_matrix.shape[0] < num_sentences:
        return " ".join(original_sentences[:tfidf_matrix.shape[0]])
    svd = TruncatedSVD(n_components=min(tfidf_matrix.shape[0] - 1, 1), random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    scores = svd_matrix.flatten()
    ranked_sentences = sorted(((scores[i], original_sentences[i]) for i in range(len(scores))), reverse=True)
    return " ".join(s for _, s in ranked_sentences[:num_sentences])

# 处理单个数据样本
def process_sample(sample, method_name):
    text = sample["text"]
    abstract = sample["abstract"]
    try:
        if method_name == "keyword_summary":
            summary = keyword_summary(text)
        elif method_name == "hits":
            summary = hits_summary(text)
        elif method_name == "lsa":
            summary = lsa_summary(text)
        else:
            summary = "Invalid method."
        return {"abstract": abstract, "output": summary}
    except Exception as e:
        print(f"Error processing sample: {e}")
        return {"abstract": abstract, "output": "Error generating summary."}

# 数据集处理函数（替代 lambda）
def process_dataset_sample(sample, method_name):
    return process_sample(sample, method_name)

# 主函数
def process_datasets(path, outpath):
    methods = ["keyword_summary", "hits", "lsa"]
    for dataset_name in os.listdir(path):
        dataset_path = os.path.join(path, dataset_name, "test")
        if not os.path.isdir(dataset_path):
            continue
        print(f"Processing dataset: {dataset_name}")
        
        # 加载 test 数据集
        test_dataset = load_from_disk(dataset_path)

        for method_name in methods:
            print(f"Using method: {method_name}")
            # 使用 num_proc 并行处理数据集
            processed_data = test_dataset.map(
                process_dataset_sample,
                fn_kwargs={"method_name": method_name},
                num_proc=4,  # 启用多进程
            )
            
            # 保存结果
            output_dir = os.path.join(outpath, method_name, dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            processed_data = processed_data.remove_columns([col for col in processed_data.column_names if col not in ["abstract", "output"]])
            processed_data.save_to_disk(output_dir)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 配置相对路径
path = os.path.join(current_dir, "../data/processed")  # 输入路径
outpath = os.path.join(current_dir, "../output")  # 输出路径

# 运行
process_datasets(path, outpath)
