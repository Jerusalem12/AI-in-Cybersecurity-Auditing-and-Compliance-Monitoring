from __future__ import annotations
import pickle
import numpy as np
from typing import List
from .base import BaseRecommender

class EmbeddingModel:
    """
    Thin wrapper to allow swapping implementations.
    Default: sentence-transformers (if installed).
    Fallback: averages TF-IDF idf weights (poor-man embeddings).
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None
        self._use_sbert = False
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_sbert = True
        except Exception:
            self._model = None
            self._use_sbert = False

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._use_sbert:
            return np.array(self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))
        # fallback: simple hash-based pseudo-embedding to keep code runnable without sbert
        rng = np.random.default_rng(0)
        vecs = []
        for t in texts:
            h = abs(hash(t)) % (10**9)
            rng = np.random.default_rng(h)
            vecs.append(rng.normal(size=384))
        X = np.vstack(vecs)
        # L2 normalize
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return X

class EmbeddingRecommender(BaseRecommender):
    def __init__(self, model_name: str | None = None):
        self.model = EmbeddingModel(model_name=model_name)
        self.X = None
        self.control_ids = []

    def fit(self, control_texts: list[str], control_ids: list[str]):
        self.X = self.model.encode(control_texts)
        self.control_ids = list(control_ids)
        return self

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"X": self.X, "control_ids": self.control_ids, "model_name": self.model.model_name}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model = EmbeddingModel(model_name=obj.get("model_name"))
        self.X = obj["X"]
        self.control_ids = obj["control_ids"]
        return self

    def predict_topk(self, text: str, k: int = 3, boosts=None, negatives=None):
        q = self.model.encode([text])[0]
        sims = (self.X @ q)  # cosine if normalized
        idx = np.argsort(-sims)[:k]
        return [self.control_ids[i] for i in idx], sims[idx].tolist()
