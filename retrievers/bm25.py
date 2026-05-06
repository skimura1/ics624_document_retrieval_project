import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from loader import load_data
from retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.bm25_matrix = None  # CSR (n_docs, n_terms) with BM25 weights

    def fit(self, passages_text: list[str]):
        vectorizer = CountVectorizer()
        tf = vectorizer.fit_transform(passages_text)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean()
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        k1, b = self.k1, self.b
        coo = tf.tocoo()
        row, col, data = coo.row, coo.col, coo.data.astype(np.float64)
        denom = data + k1 * (1.0 - b + b * doc_len[row] / avgdl)
        bm25_vals = idf[col] * data * (k1 + 1.0) / denom
        self.bm25_matrix = csr_matrix((bm25_vals, (row, col)), shape=tf.shape, dtype=np.float32)
        self.vectorizer = vectorizer

    def query(self, query: str) -> list[int]:
        if self.bm25_matrix is None:
            raise ValueError("BM25 matrix is not fitted. Please fit the retriever first.")
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.bm25_matrix.T).toarray().ravel()
        part = np.argpartition(scores, -self.top_k)[-self.top_k:]
        return list(map(int, part[np.argsort(scores[part])[::-1]]))

    def query_batch(self, queries: list[str], chunk_size: int = 64) -> list[list[int]]:
        if self.bm25_matrix is None:
            raise ValueError("BM25 matrix is not fitted. Please fit the retriever first.")
        results = []
        for start in range(0, len(queries), chunk_size):
            chunk = queries[start:start + chunk_size]
            q_vecs = self.vectorizer.transform(chunk)
            scores_mat = (q_vecs @ self.bm25_matrix.T).toarray()
            for scores in scores_mat:
                part = np.argpartition(scores, -self.top_k)[-self.top_k:]
                results.append(list(map(int, part[np.argsort(scores[part])[::-1]])))
            print(f"BM25 batch: {min(start + chunk_size, len(queries))}/{len(queries)}")
        return results

    def score(self, query: str) -> list[float]:
        if self.bm25_matrix is None:
            raise ValueError("BM25 matrix is not fitted. Please fit the retriever first.")
        q_vec = self.vectorizer.transform([query])
        scores = (q_vec @ self.bm25_matrix.T).toarray().ravel()
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores.tolist()

    def score_candidates(self, query: str, candidate_indices: list[int]) -> np.ndarray:
        if self.bm25_matrix is None:
            raise ValueError("BM25 matrix is not fitted. Please fit the retriever first.")
        q_vec = self.vectorizer.transform([query])
        cand = np.asarray(candidate_indices)
        scores = (q_vec @ self.bm25_matrix[cand].T).toarray().ravel()
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores


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
    top_k_passages = [passages_text[i] for i in top_k_indices]
    top_k_scores = retriever.score(query)
    print(top_k_passages)
    print(top_k_scores)
