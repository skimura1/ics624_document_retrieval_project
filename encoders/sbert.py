from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceBERTEncoder:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, passages: list[str]) -> np.ndarray:
        return self.model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
