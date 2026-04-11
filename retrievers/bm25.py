from rank_bm25 import BM25Okapi
import numpy as np
from loader import load_data
from base import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, top_k: int = 10):
        self.bm25 = None
        self.top_k = top_k
        self.passages_tokens = None

    def fit(self, passages_text: list[str]):
        # Split the passages text into tokens
        self.passages_tokens = [passage.split() for passage in passages_text]
        self.bm25 = BM25Okapi(self.passages_tokens)

    def query(self, query: str):
        if self.bm25 is None:
            raise ValueError("BM25 matrix is not fitted. Please fit the retriever first.")
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = list(map(int, np.argsort(scores)[::-1][:self.top_k]))
        return top_k_indices

if __name__ == "__main__":
    ds = load_data()
    passages_text = []
    for example in ds:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
    retriever = BM25Retriever(top_k=10)
    retriever.fit(passages_text)
    query = "What is the capital of France?"
    top_k_indices = retriever.query(query)
    print(top_k_indices)