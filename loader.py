from datasets import load_dataset

def load_data():
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    return ds
