from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from summarization.preprocess import preprocess_text

def lsa_summary(text, num_sentences=5):
    """
    使用 TF-IDF + LSA 生成摘要。
    """
    # 对文本进行预处理
    original_sentences, preprocessed_sentences = preprocess_text(text)
    
    # 检查句子数量是否足够
    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)  # 如果句子数量不足，直接返回原文

    try:
        # 计算 TF-IDF 矩阵
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

        # 验证矩阵是否为空
        if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
            return "Error: TF-IDF matrix is empty. Check the input text."

        # 确定 n_components 的值
        n_components = min(num_sentences, tfidf_matrix.shape[0])
        if n_components < 1:
            return "Error: Not enough data for LSA summarization."

        # 使用 LSA 分解
        svd = TruncatedSVD(n_components=n_components, n_iter=100)
        sentence_scores = svd.fit_transform(tfidf_matrix)  # 每个句子的权重

        # 平均权重排序
        avg_scores = sentence_scores.mean(axis=1)

        # 根据权重排序句子
        ranked_indices = avg_scores.argsort()[::-1]
        ranked_sentences = [original_sentences[i] for i in ranked_indices[:num_sentences]]

        return ' '.join(ranked_sentences)

    except Exception as e:
        return f"Error during LSA summarization: {e}"
