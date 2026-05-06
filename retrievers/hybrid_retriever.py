from retrievers.base import BaseRetriever
from retrievers.bm25 import BM25Retriever
from retrievers.tf_idf_retriever import TFIDFRetriever
from retrievers.dense_retriever import DenseRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loader import load_data


class HybridRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10, candidate_k: int = 50):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.bm25_retriever = BM25Retriever(top_k=candidate_k)
        self.tf_idf_retriever = TFIDFRetriever(top_k=candidate_k)
        self.dense_retriever = DenseRetriever(top_k=candidate_k)

    def fit(self, passages_text: list[str]):
        self.bm25_retriever.fit(passages_text)
        self.tf_idf_retriever.fit(passages_text)
        self.dense_retriever.fit("sbert_embeddings.npy", passages_text)

    def query(self, query: str) -> list[int]:
        bm25_cands  = self.bm25_retriever.query(query)
        tfidf_cands = self.tf_idf_retriever.query(query)

        q_emb = self.dense_retriever.encoder.encode([query]).astype(np.float32)
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
        _, dense_top = self.dense_retriever.index.search(q_emb, self.candidate_k)
        dense_cands = dense_top[0].tolist()

        union    = list(set(bm25_cands) | set(tfidf_cands) | set(dense_cands))
        cand_arr = np.array(union)

        tfidf_s = self.tf_idf_retriever.score_candidates(query, union)

        dense_s = (self.dense_retriever.embeddings[cand_arr] @ q_emb.T).squeeze(axis=1)
        dense_s = np.clip(dense_s, 0, None)
        dense_max = dense_s.max()
        if dense_max > 0:
            dense_s /= dense_max

        combined = 0.25 * tfidf_s + 0.75 * dense_s
        order = np.argsort(combined)[::-1]
        return [union[i] for i in order[:self.top_k]]

    def query_batch(self, queries: list[str], chunk_size: int = 512) -> list[list[int]]:
        results = []
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start:start + chunk_size]

            q_embs = self.dense_retriever.encoder.encode(chunk).astype(np.float32)
            q_embs /= np.linalg.norm(q_embs, axis=1, keepdims=True)
            _, dense_batch = self.dense_retriever.index.search(q_embs, self.candidate_k)

            tfidf_batch  = self.tf_idf_retriever.query_batch(chunk)
            bm25_batch   = self.bm25_retriever.query_batch(chunk)
            tfidf_q_vecs = self.tf_idf_retriever.vectorizer.transform(chunk)

            for i in range(len(chunk)):
                union    = list(set(bm25_batch[i]) | set(tfidf_batch[i]) | set(dense_batch[i].tolist()))
                cand_arr = np.array(union)

                dense_s = self.dense_retriever.embeddings[cand_arr] @ q_embs[i]
                dense_s = np.clip(dense_s, 0, None)
                dense_max = dense_s.max()
                if dense_max > 0:
                    dense_s /= dense_max

                tfidf_s = cosine_similarity(tfidf_q_vecs[i], self.tf_idf_retriever.tf_idf_matrix[cand_arr])[0]
                tfidf_max = tfidf_s.max()
                if tfidf_max > 0:
                    tfidf_s /= tfidf_max

                combined = 0.25 * tfidf_s + 0.75 * dense_s
                order = np.argsort(combined)[::-1]
                results.append([union[j] for j in order[:self.top_k]])

            print(f"Hybrid: {min(start + chunk_size, len(queries))}/{len(queries)} queries...")

        return results

    def score(self, query: str) -> list[int]:
        return self.query(query)


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
