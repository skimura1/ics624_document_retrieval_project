import time
from retrievers.base import BaseRetriever

_TIMING_WARMUP = 3
_TIMING_SAMPLE = 30

def measure_retrieval_time(retriever: BaseRetriever, queries: list[str]) -> float:
    """Returns average seconds per single query call (consistent across all retrievers)."""
    for q in queries[:_TIMING_WARMUP]:
        retriever.query(q)

    times = []
    for q in queries[_TIMING_WARMUP:_TIMING_WARMUP + _TIMING_SAMPLE]:
        start = time.perf_counter()
        retriever.query(q)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)