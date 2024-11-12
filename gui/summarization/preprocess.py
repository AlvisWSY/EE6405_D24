from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("punkt")
nltk.download("stopwords")

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        preprocessed_sentences.append(" ".join(words))
    return sentences, preprocessed_sentences
