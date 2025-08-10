from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import faiss
import numpy as np
import os

class FIRRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.index = None
        self.embeddings = None
    
    def load_and_index_corpus(self, folder="data/firs"):
        """Load FIR corpus and create FAISS index"""
        self.corpus = []
        files = os.listdir(folder)
        
        for file in files:
            with open(os.path.join(folder, file), "r") as f:
                self.corpus.append(f.read())
        
        # Generate embeddings
        self.embeddings = self.model.encode(self.corpus)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
    
    def find_similar_firs(self, query_text, top_k=3):
        """Find similar FIRs using dense embeddings and FAISS"""
        if self.index is None:
            raise ValueError("Corpus not indexed. Call load_and_index_corpus() first.")
        
        # Encode query
        query_embedding = self.model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for i, score in zip(indices[0], scores[0]):
            results.append({
                'text': self.corpus[i],
                'similarity': float(score)
            })
        
        return results

# Legacy function for backward compatibility
def find_similar_firs(query_text, corpus, top_k=3):
    retriever = FIRRetriever()
    retriever.corpus = corpus
    retriever.embeddings = retriever.model.encode(corpus)
    
    dimension = retriever.embeddings.shape[1]
    retriever.index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(retriever.embeddings)
    retriever.index.add(retriever.embeddings.astype('float32'))
    
    results = retriever.find_similar_firs(query_text, top_k)
    return [r['text'] for r in results]


class FIRIntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])
        self.labels = ['Robbery', 'Domestic Violence', 'Suicide', 'Theft', 'Assault', 'Other']
    
    def train(self, texts, labels):
        """Train the intent classifier"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        accuracy = self.pipeline.score(X_test, y_test)
        print(f"Intent Classification Accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def predict(self, text):
        """Predict intent for a single FIR text"""
        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        
        return {
            'intent': prediction,
            'confidence': max(probabilities),
            'all_probabilities': dict(zip(self.labels, probabilities))
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)