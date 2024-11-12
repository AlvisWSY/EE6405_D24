import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarization.preprocess import preprocess_text

def hits_summary(text, num_sentences):
    original_sentences, preprocessed_sentences = preprocess_text(text)
    if len(original_sentences) <= num_sentences:
        return " ".join(original_sentences)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    hubs, authorities = nx.hits(nx_graph)
    ranked_sentences = sorted(((authorities[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    return ' '.join([s for _, s in ranked_sentences[:num_sentences]])
