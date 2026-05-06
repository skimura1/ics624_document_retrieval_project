from retrievers.base import BaseRetriever
from datasets import Dataset
from loader import load_data

def mrr_at_10(retriever: BaseRetriever, dataset: Dataset, max_queries: int = 100) -> float:
    # Pre-collect all (query, global_id) pairs
    items: list[tuple[str, int]] = []
    global_counter = 0
    for example in dataset:
        if max_queries is not None and len(items) >= max_queries:
            break
        is_selected = example["passages"]["is_selected"]
        base = global_counter
        global_counter += len(is_selected)
        if 1 not in is_selected:
            continue
        global_id = base + is_selected.index(1)
        items.append((example["query"], global_id))

    queries = [q for q, _ in items]
    global_ids = [g for _, g in items]

    # Use batched query path when available (avoids per-query model overhead)
    if hasattr(retriever, "query_batch"):
        print(f"Running query_batch on {len(queries)} queries...")
        all_top_k = retriever.query_batch(queries)
    else:
        all_top_k = []
        for i, q in enumerate(queries):
            all_top_k.append(retriever.query(q))
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} queries...")

    mrr = 0.0
    for global_id, top_k_indices in zip(global_ids, all_top_k):
        if global_id in top_k_indices:
            mrr += 1 / (top_k_indices.index(global_id) + 1)

    return mrr / len(items)


if __name__ == "__main__":
    from retrievers.hybrid_retriever import HybridRetriever
    ds = load_data()
    retriever = HybridRetriever(top_k=10)
    # fit the retriever
    passages_text = []
    for example in ds:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
    retriever.fit(passages_text)
    ds = load_data()
    # evaluate the retriever
    mrr = mrr_at_10(retriever, ds)
    print(f"MRR: {mrr}")