from retrievers.base import BaseRetriever
from datasets import Dataset
from loader import load_data
from retrievers.hybrid_retriever import HybridRetriever

def mrr_at_10(retriever: BaseRetriever, dataset: Dataset, max_queries: int = 100) -> float:
    global_counter = 0
    mrr = 0
    query_count = 0
    # iterate over the dataset
    for example in dataset:
        if max_queries is not None and query_count >= max_queries:
            break
        is_selected = example["passages"]["is_selected"]
        base = global_counter
        global_counter += len(is_selected)
        if 1 not in is_selected:
            # no relevant passage
            continue
        relevant_local_index = is_selected.index(1)
        global_id = base + relevant_local_index  # global index of the relevant passage
        top_k_indices = retriever.query(example["query"])
        query_count += 1
        if query_count % 100 == 0:
            print(f"Processed {query_count} queries...")
        # check if global_id is in top_k_indices
        if global_id in top_k_indices:
            # find the rank of the globa_id in top_k_indices
            rank = top_k_indices.index(global_id) + 1
            mrr += 1 / rank

    return mrr / query_count


if __name__ == "__main__":
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