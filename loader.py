from datasets import load_dataset

def load_data(n: int = 1_000_000):
    ds = load_dataset("microsoft/ms_marco", "v2.1")
    return ds["train"].select(range(min(n, len(ds["train"]))))

if __name__ == "__main__":
    ds = load_data()
    for example in ds:
        print(example["query"])
        for passage in example["passages"]["passage_text"]:
            print(passage)
        break