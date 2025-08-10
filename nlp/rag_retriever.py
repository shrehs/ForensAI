from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_fir_corpus(folder="data/firs"):
    corpus = []
    files = os.listdir(folder)
    for file in files:
        with open(os.path.join(folder, file), "r") as f:
            corpus.append(f.read())
    return corpus

def find_similar_firs(query_text, corpus, top_k=3):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query_text] + corpus)
    scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
    ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])
    return [x[0] for x in ranked[:top_k]]
