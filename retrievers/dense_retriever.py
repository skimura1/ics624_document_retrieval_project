import numpy as np
from base import BaseRetriever
from encoders.sbert import SentenceBERTEncoder
import faiss

class DenseRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10, encoder = None):
        self.top_k = top_k
        self.embeddings = None
        self.index = None
        
    def fit(self, file_path: str):
        # load in the embeddings from the file
        self.embeddings = np.load(file_path).astype(np.float32)
        print("loaded embeddings shape:", self.embeddings.shape)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        print("index shape:", self.index.d)
        # normalize the embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        print("normalized embeddings shape:", self.embeddings.shape)
        # build faiss index
        self.index.add(self.embeddings)
        print("added", self.index.ntotal, "embeddings")
        

    def query(self, query: str):
        if self.embeddings is None or self.index is None:
            raise ValueError("Embeddings or index are not fitted. Please fit the retriever first.")
        
        # encode the query with sbert encoder
        query_embedding = SentenceBERTEncoder().encode([query])
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        print("normalized query embedding:", query_embedding)
        # search the index for the top k most similar embeddings
        _, top_k_indices = self.index.search(query_embedding, self.top_k)
        return top_k_indices[0].tolist()

if __name__ == "__main__":
    retriever = DenseRetriever(top_k=10)
    retriever.fit("sbert_embeddings.npy")
    print("Index built successfully, total passages:", retriever.index.ntotal)
    query = "What is the capital of France?"
    top_k_indices = retriever.query(query)
    print(top_k_indices)