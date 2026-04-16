from retrievers.base import BaseRetriever
from datasets import Dataset
from loader import load_data
from evaluation.mrr import mrr_at_10
from evaluation.timing import measure_retrieval_time
from retrievers.bm25 import BM25Retriever
from retrievers.tf_idf_retriever import TFIDFRetriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
import pandas as pd

def evaluate_retriever(retriever: BaseRetriever, dataset: Dataset, queries, max_queries: int = 100) -> float:
    mrr = mrr_at_10(retriever, dataset, max_queries=max_queries)
    avg_time = measure_retrieval_time(retriever, queries[:max_queries])
    return {
        "mrr": mrr,
        "avg_time": avg_time
    }

if __name__ == "__main__":
    ds = load_data()
    # load the retrievers
    hybrid_retriever = HybridRetriever(top_k=10)
    bm25_retriever = BM25Retriever(top_k=10)
    tfidf_retriever = TFIDFRetriever(top_k=10)
    dense_retriever = DenseRetriever(top_k=10)
    retrievers = [hybrid_retriever, bm25_retriever, tfidf_retriever, dense_retriever]
    evaluations = []
    # fit the retrievers
    passages_text = []
    queries = []
    for example in ds:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
        queries.append(example["query"])
    hybrid_retriever.fit(passages_text)
    bm25_retriever.fit(passages_text)
    tfidf_retriever.fit(passages_text)
    dense_retriever.fit("sbert_embeddings.npy")
    for retriever in retrievers:
        ds = load_data()
        evaluation = evaluate_retriever(retriever, ds, queries, max_queries=1000)
        evaluations.append({
            "Retriever": retriever.__class__.__name__,
            "MRR@10": evaluation["mrr"],
            "Avg ms/query": evaluation["avg_time"]
        })
    # output the evaluations in a table
    results = pd.DataFrame(evaluations, columns=["Retriever", "MRR@10", "Avg ms/query"])
    results.to_csv("evaluations.csv", index=False)
    print(results)