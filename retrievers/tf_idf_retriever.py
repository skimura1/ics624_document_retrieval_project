from loader import load_data
from retrievers.base import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10):
        self.vectorizer = TfidfVectorizer()
        self.top_k = top_k
        self.tf_idf_matrix = None

    def fit(self, passages_text: list[str]):
        self.tf_idf_matrix = self.vectorizer.fit_transform(passages_text)

    def query(self, query: str):
        if self.tf_idf_matrix is None:
            raise ValueError("TF-IDF matrix is not fitted. Please fit the retriever first.")
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tf_idf_matrix)[0]
        top_k_indices = list(map(int, np.argsort(cosine_similarities)[::-1][:self.top_k]))
        return top_k_indices
    
    def score(self, query: str):
        if self.tf_idf_matrix is None:
            raise ValueError("TF-IDF matrix is not fitted. Please fit the retriever first.")
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tf_idf_matrix)[0]
        #normalize the scores [0, 1]
        cosine_similarities = cosine_similarities / np.max(cosine_similarities)
        return cosine_similarities.tolist()

if __name__ == "__main__":
    ds = load_data()
    passages_text = []
    for example in ds:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
    retriever = TFIDFRetriever(top_k=10)
    retriever.fit(passages_text)
    query = "What is the capital of France?"
    top_k_indices = retriever.query(query)
    top_k_passages = [passages_text[i] for i in top_k_indices]
    print(top_k_passages)