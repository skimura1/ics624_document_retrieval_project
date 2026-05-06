from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from loader import load_data


class SentenceBERTEncoder:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.model = SentenceTransformer(self.MODEL_NAME, device=device)
        print(f"SentenceBERTEncoder using device: {device}")

    def encode(self, passages: list[str]) -> np.ndarray:
        return self.model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

if __name__ == "__main__":
    ds = load_data()
    passages_text = []
    for example in ds:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
    encoder = SentenceBERTEncoder()
    embeddings = encoder.encode(passages_text)
    np.save("sbert_embeddings.npy", embeddings)