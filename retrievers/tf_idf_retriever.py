from loader import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TFIDFRetriever:
    def __init__(self, top_k: int = 10):
        self.vectorizer = TfidfVectorizer()
        self.top_k = top_k
        self.tf_idf_matrix = None
        self.passages_text = None

    def fit(self, passages_text: list[str]):
        # Take in the list of passages text
        self.passages_text = passages_text
        # Fit the vectorizer on the passages text
        self.tf_idf_matrix = self.vectorizer.fit_transform(passages_text)

    def query(self, query: str):
        if self.tf_idf_matrix is None:
            raise ValueError("TF-IDF matrix is not fitted. Please fit the retriever first.")
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tf_idf_matrix)[0]
        top_k_indices = list(np.argsort(cosine_similarities)[::-1][:self.top_k])
        return top_k_indices

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