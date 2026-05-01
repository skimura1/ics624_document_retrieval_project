import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from loader import load_data


class ColBERTEncoder:
    """Encodes queries and documents into per-token embeddings for late interaction scoring."""

    MODEL_NAME = "bert-base-uncased"
    DIM = 128  # projection dimension (ColBERT default)

    def __init__(self, device: str = None, dim: int = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim or self.DIM
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.bert = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        hidden_size = self.bert.config.hidden_size
        # Linear projection to lower-dimensional space (no bias, as in the ColBERT paper)
        self.projection = nn.Linear(hidden_size, self.dim, bias=False).to(self.device)
        self.bert.eval()
        self.projection.eval()

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
        max_length: int = 128,
        query_max_length: int = 32,
    ) -> list[torch.Tensor]:
        """
        Encode a list of texts into projected token embeddings.
        Returns a list of (seq_len, dim) tensors (L2-normalized).
        """
        max_len = query_max_length if is_query else max_length
        prefix = "[Q] " if is_query else "[D] "
        prefixed = [prefix + t for t in texts]

        results = []
        for text in prefixed:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding=False,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.bert(**inputs)
            token_emb = outputs.last_hidden_state[0]  # (seq_len, hidden)
            projected = self.projection(token_emb)     # (seq_len, dim)
            normalized = nn.functional.normalize(projected, dim=-1)
            results.append(normalized)
        return results


class ColBERTScorer:
    """
    Late-interaction ColBERT scorer.
    Score(q, d) = sum over query tokens of max cosine similarity against document tokens.
    """

    def __init__(self, device: str = None, dim: int = None):
        self.encoder = ColBERTEncoder(device=device, dim=dim)
        self.device = self.encoder.device

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        query_embs = self.encoder.encode([query], is_query=True)
        q = query_embs[0]  # (q_len, dim)

        scores = []
        for passage in passages:
            doc_embs = self.encoder.encode([passage], is_query=False)
            d = doc_embs[0]  # (d_len, dim)
            score = self._maxsim(q, d)
            scores.append(score)
        return scores

    def _maxsim(self, q: torch.Tensor, d: torch.Tensor) -> float:
        """MaxSim: for each query token, take max cosine sim over doc tokens, then sum."""
        # q: (q_len, dim), d: (d_len, dim) — both already L2-normalized
        sim = torch.mm(q, d.T)         # (q_len, d_len)
        max_sim = sim.max(dim=1).values  # (q_len,)
        return max_sim.sum().item()

    def score_preencoded(
        self, q: torch.Tensor, d: torch.Tensor
    ) -> float:
        """Score with pre-encoded tensors (already projected + normalized)."""
        return self._maxsim(q, d)


if __name__ == "__main__":
    ds = load_data()
    passages_text = []
    for example in ds["train"]:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
        if len(passages_text) >= 20:
            break

    scorer = ColBERTScorer()
    query = "What is the capital of France?"
    scores = scorer.score_pairs(query, passages_text[:5])
    for p, s in zip(passages_text[:5], scores):
        print(f"Score: {s:.4f} | {p[:80]}")
