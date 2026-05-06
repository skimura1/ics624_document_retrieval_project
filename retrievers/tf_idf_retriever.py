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
        part = np.argpartition(cosine_similarities, -self.top_k)[-self.top_k:]
        top_k_indices = list(map(int, part[np.argsort(cosine_similarities[part])[::-1]]))
        return top_k_indices

    def query_batch(self, queries: list[str], batch_size: int = 64) -> list[list[int]]:
        if self.tf_idf_matrix is None:
            raise ValueError("TF-IDF matrix is not fitted. Please fit the retriever first.")
        query_vectors = self.vectorizer.transform(queries)
        results = []
        for start in range(0, len(queries), batch_size):
            sims = cosine_similarity(query_vectors[start:start + batch_size], self.tf_idf_matrix)
            for row in sims:
                part = np.argpartition(row, -self.top_k)[-self.top_k:]
                results.append(list(map(int, part[np.argsort(row[part])[::-1]])))
        return results
    
    def score_candidates(self, query: str, candidate_indices: list[int]) -> np.ndarray:
        if self.tf_idf_matrix is None:
            raise ValueError("TF-IDF matrix is not fitted. Please fit the retriever first.")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tf_idf_matrix[candidate_indices])[0]
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores

    def score(self, query: str):
        if self.tf_idf_matrix is None:
            raise ValueError("TF-IDF matrix is not fitted. Please fit the retriever first.")
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tf_idf_matrix)[0]
        max_score = np.max(cosine_similarities)
        if max_score > 0:
            cosine_similarities = cosine_similarities / max_score
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
    top_k_scores = retriever.score(query)
    top_k_passages = [passages_text[i] for i in top_k_indices]
    print(top_k_passages)
    print(top_k_scores)