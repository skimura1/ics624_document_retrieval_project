import os
import numpy as np
from retrievers.base import BaseRetriever
from encoders.sbert import SentenceBERTEncoder
import faiss

class DenseRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.encoder = SentenceBERTEncoder()
        self.embeddings = None
        self.index = None
        
    def fit(self, file_path: str, passages_text: list[str] = None):
        if os.path.exists(file_path) and passages_text is not None:
            stored = np.load(file_path, mmap_mode='r')
            if stored.shape[0] != len(passages_text):
                print(f"Embeddings count mismatch ({stored.shape[0]} stored vs {len(passages_text)} passages) — regenerating...")
                os.remove(file_path)
        if not os.path.exists(file_path):
            if passages_text is None:
                raise ValueError(f"{file_path} not found and no passages_text provided to encode.")
            print(f"{file_path} not found — encoding passages with SBERT...")
            embeddings = self.encoder.encode(passages_text)
            np.save(file_path, embeddings)
            print(f"Saved embeddings to {file_path}")
        self.embeddings = np.load(file_path).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        # normalize the embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        # build faiss index
        self.index.add(self.embeddings)
        

    def query(self, query: str):
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings or index are not fitted. Please fit the retriever first.")

        # encode the query with sbert encoder
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / norms
        # search the index for the top k most similar embeddings
        _, top_k_indices = self.index.search(query_embedding, self.top_k)
        return top_k_indices[0].tolist()

    def query_batch(self, queries: list[str]) -> list[list[int]]:
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings or index are not fitted. Please fit the retriever first.")
        embeddings = self.encoder.encode(queries)
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        _, indices = self.index.search(embeddings, self.top_k)
        return indices.tolist()

    def score_candidates(self, query: str, candidate_indices: list[int]) -> np.ndarray:
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings or index are not fitted. Please fit the retriever first.")
        q_emb = self.encoder.encode([query]).astype(np.float32)
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
        cand_arr = np.array(candidate_indices)
        scores = (self.embeddings[cand_arr] @ q_emb.T).squeeze(axis=1)
        scores = np.clip(scores, 0, None)
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores

    def score(self, query: str):
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings or index are not fitted. Please fit the retriever first.")
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / norms
        raw_scores, indices = self.index.search(query_embedding, self.index.ntotal)
        # Re-index scores by passage position so scores[i] corresponds to passages_text[i]
        ordered = np.empty(self.index.ntotal, dtype=np.float32)
        ordered[indices[0]] = raw_scores[0]
        ordered = np.clip(ordered, 0, None)
        max_score = np.max(ordered)
        if max_score > 0:
            ordered = ordered / max_score
        return ordered.tolist()

if __name__ == "__main__":
    retriever = DenseRetriever(top_k=10)
    retriever.fit("sbert_embeddings.npy")
    query = "What is the capital of France?"
    top_k_indices = retriever.query(query)
    top_k_scores = retriever.score(query)
    print(top_k_indices)
    print(top_k_scores)