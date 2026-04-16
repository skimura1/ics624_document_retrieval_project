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
        
    def fit(self, file_path: str):
        # load in the embeddings from the file
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
    
    def score(self, query: str):
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings or index are not fitted. Please fit the retriever first.")
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding.astype(np.float32)
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / norms
        scores, _ = self.index.search(query_embedding, self.index.ntotal)
        #normalize the scores [0, 1]
        scores = scores / np.max(scores)
        return scores[0].tolist()

if __name__ == "__main__":
    retriever = DenseRetriever(top_k=10)
    retriever.fit("sbert_embeddings.npy")
    query = "What is the capital of France?"
    top_k_indices = retriever.query(query)
    print(top_k_indices)