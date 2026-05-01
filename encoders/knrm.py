import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from loader import load_data


class KNRMEncoder:
    """Token-level encoder used by KNRM to produce per-token embeddings."""

    MODEL_NAME = "bert-base-uncased"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    def encode_tokens(self, texts: list[str], max_length: int = 128) -> list[np.ndarray]:
        """Return a list of (seq_len, hidden_dim) token embedding arrays, one per text."""
        results = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # last hidden state: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            results.append(token_embeddings)
        return results


class KNRMRanker(nn.Module):
    """KNRM re-ranker: soft kernel pooling over query-document cosine similarity matrix."""

    def __init__(self, n_kernels: int = 11, sigma: float = 0.1, exact_sigma: float = 0.001):
        super().__init__()
        self.n_kernels = n_kernels
        # Kernel means: n_kernels-1 soft kernels evenly spaced in [-1, 1], plus one exact-match kernel at 1
        mus = np.linspace(-1.0, 1.0, n_kernels - 1).tolist() + [1.0]
        sigmas = [sigma] * (n_kernels - 1) + [exact_sigma]
        self.register_buffer("mus", torch.tensor(mus, dtype=torch.float32))
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.mlp = nn.Linear(n_kernels, 1, bias=False)

    def forward(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """
        query_emb: (q_len, d)
        doc_emb:   (d_len, d)
        returns scalar relevance score
        """
        # Normalize embeddings
        query_emb = nn.functional.normalize(query_emb, dim=-1)  # (q, d)
        doc_emb = nn.functional.normalize(doc_emb, dim=-1)       # (p, d)

        # Cosine similarity matrix: (q, p)
        sim = torch.mm(query_emb, doc_emb.T)

        # Kernel scoring: for each query token, apply n_kernels Gaussian kernels across doc tokens
        # sim: (q, p) -> (q, p, 1) broadcast with mus (k,)
        sim_expanded = sim.unsqueeze(-1)  # (q, p, 1)
        # Gaussian kernel: exp(-(sim - mu)^2 / (2*sigma^2))
        kernel_scores = torch.exp(
            -((sim_expanded - self.mus) ** 2) / (2 * self.sigmas ** 2)
        )  # (q, p, k)

        # Log-sum-exp pooling over document tokens per query token per kernel
        kernel_sums = kernel_scores.sum(dim=1)  # (q, k)
        log_kernel_sums = torch.log(kernel_sums.clamp(min=1e-10))

        # Sum over query tokens -> (k,) kernel feature vector
        kernel_features = log_kernel_sums.sum(dim=0)  # (k,)

        score = self.mlp(kernel_features).squeeze()
        return score


class KNRMScorer:
    """Wraps KNRMEncoder + KNRMRanker for scoring query-passage pairs."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = KNRMEncoder(device=self.device)
        self.ranker = KNRMRanker().to(self.device)
        self.ranker.eval()

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        query_embs = self.encoder.encode_tokens([query])
        q_tensor = torch.tensor(query_embs[0], dtype=torch.float32).to(self.device)

        scores = []
        for passage in passages:
            p_embs = self.encoder.encode_tokens([passage])
            p_tensor = torch.tensor(p_embs[0], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                s = self.ranker(q_tensor, p_tensor).item()
            scores.append(s)
        return scores


if __name__ == "__main__":
    ds = load_data()
    passages_text = []
    for example in ds["train"]:
        for passage in example["passages"]["passage_text"]:
            passages_text.append(passage)
        if len(passages_text) >= 20:
            break

    scorer = KNRMScorer()
    query = "What is the capital of France?"
    scores = scorer.score_pairs(query, passages_text[:5])
    for p, s in zip(passages_text[:5], scores):
        print(f"Score: {s:.4f} | {p[:80]}")
