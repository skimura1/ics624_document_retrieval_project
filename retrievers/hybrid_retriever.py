from base import BaseRetriever
from bm25 import BM25Retriever
from tf_idf_retriever import TFIDFRetriever
from dense_retriever import DenseRetriever
import numpy as np
from loader import load_data

class HybridRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.bm25_retriever = BM25Retriever(top_k=top_k)
        self.tf_idf_retriever = TFIDFRetriever(top_k=top_k)
        self.dense_retriever = DenseRetriever(top_k=top_k)

    def fit(self, passages_text: list[str]):
        self.bm25_retriever.fit(passages_text)
        self.tf_idf_retriever.fit(passages_text)
        self.dense_retriever.fit("sbert_embeddings.npy")

    def query(self, query: str):
        bm25_indices = self.bm25_retriever.query(query)
        tf_idf_indices = self.tf_idf_retriever.query(query)
        dense_indices = self.dense_retriever.query(query)
        union_indices = set(bm25_indices) | set(tf_idf_indices) | set(dense_indices)
        bm25_scores = np.array(self.bm25_retriever.score(query))
        tfidf_scores = np.array(self.tf_idf_retriever.score(query))
        dense_scores = np.array(self.dense_retriever.score(query))
        combined_scores = []

        for index in union_indices:
            bm25_score = bm25_scores[index]
            tfidf_score = tfidf_scores[index]
            dense_score = dense_scores[index]
            sparse_scores = 0.5 * bm25_score + 0.5 * tfidf_score
            hybrid_score = 0.5 * sparse_scores + 0.5 * dense_score
            combined_scores.append((index, hybrid_score))

        return [index for index, _ in sorted(combined_scores, key=lambda x: x[1], reverse=True)][:self.top_k]

    def score(self, query: str):
        union_indices = set(self.bm25_retriever.query(query)) | set(self.tf_idf_retriever.query(query)) | set(self.dense_retriever.query(query))
        bm25_scores = np.array(self.bm25_retriever.score(query))
        tfidf_scores = np.array(self.tf_idf_retriever.score(query))
        dense_scores = np.array(self.dense_retriever.score(query))
        combined_scores = []

        for index in union_indices:
            bm25_score = bm25_scores[index]
            tfidf_score = tfidf_scores[index]
            dense_score = dense_scores[index]
            sparse_scores = 0.5 * bm25_score + 0.5 * tfidf_score
            hybrid_score = 0.5 * sparse_scores + 0.5 * dense_score
            combined_scores.append((index, hybrid_score))

        return [index for index, _ in sorted(combined_scores, key=lambda x: x[1], reverse=True)][:self.top_k]

if __name__ == "__main__":
    ds = load_data()
    passages_text = []
    for example in ds:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
    retriever = HybridRetriever(top_k=10)
    retriever.fit(passages_text)
    query = "What is the capital of France?"
    top_k_indices = retriever.query(query)
    top_k_passages = [passages_text[i] for i in top_k_indices]
    print(top_k_passages)