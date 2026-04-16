import time
from retrievers.base import BaseRetriever

def measure_retrieval_time(retriever: BaseRetriever, queries: list[str]) -> float:
    times = []

    for query in queries:
        start_time = time.perf_counter()
        retriever.query(query)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return sum(times) / len(times)