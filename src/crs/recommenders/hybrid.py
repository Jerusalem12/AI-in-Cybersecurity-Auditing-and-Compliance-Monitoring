from __future__ import annotations
import pickle
import numpy as np
from .base import BaseRecommender
from .tfidf import TFIDFRecommender
from .embeddings import EmbeddingRecommender

class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender that combines TF-IDF and Embeddings with weighted fusion.

    TF-IDF: Good at exact keyword/phrase matching
    Embeddings: Good at semantic similarity
    Hybrid: Best of both worlds
    """

    def __init__(self, alpha: float = 0.6, tfidf_params: dict = None, emb_params: dict = None):
        """
        Args:
            alpha: Weight for embeddings (0.0-1.0). Higher = more weight on embeddings.
                   Default 0.6 means 60% embeddings, 40% TF-IDF.
            tfidf_params: Parameters for TFIDFRecommender
            emb_params: Parameters for EmbeddingRecommender
        """
        self.alpha = alpha
        self.tfidf_rec = TFIDFRecommender(**(tfidf_params or {}))
        self.emb_rec = EmbeddingRecommender(**(emb_params or {}))
        self.control_ids = []

    def fit(self, control_texts: list[str], control_ids: list[str]):
        self.control_ids = list(control_ids)
        self.tfidf_rec.fit(control_texts, control_ids)
        self.emb_rec.fit(control_texts, control_ids)
        return self

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "tfidf_rec": self.tfidf_rec,
                "emb_rec": self.emb_rec,
                "control_ids": self.control_ids
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.alpha = obj["alpha"]
        self.tfidf_rec = obj["tfidf_rec"]
        self.emb_rec = obj["emb_rec"]
        self.control_ids = obj["control_ids"]
        return self

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score < 1e-9:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _fuse_scores(self, tfidf_ids, tfidf_scores, emb_ids, emb_scores):
        """Combine scores from both models with weighted fusion"""
        # Create score dictionaries
        tfidf_dict = {cid: score for cid, score in zip(tfidf_ids, tfidf_scores)}
        emb_dict = {cid: score for cid, score in zip(emb_ids, emb_scores)}

        # Get union of all IDs
        all_ids = set(tfidf_ids) | set(emb_ids)

        # Normalize scores separately
        tfidf_scores_norm = self._normalize_scores(list(tfidf_dict.values()))
        emb_scores_norm = self._normalize_scores(list(emb_dict.values()))

        tfidf_dict_norm = {cid: score for cid, score in zip(tfidf_dict.keys(), tfidf_scores_norm)}
        emb_dict_norm = {cid: score for cid, score in zip(emb_dict.keys(), emb_scores_norm)}

        # Compute weighted combination
        combined = {}
        for cid in all_ids:
            tfidf_s = tfidf_dict_norm.get(cid, 0.0)
            emb_s = emb_dict_norm.get(cid, 0.0)
            combined[cid] = self.alpha * emb_s + (1 - self.alpha) * tfidf_s

        # Sort by combined score
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_items], [x[1] for x in sorted_items]

    def predict_topk(self, text: str, k: int = 3, boosts: dict | None = None, negatives: dict | None = None):
        # Get larger candidate pools from both models
        candidate_k = max(k * 3, 20)

        tfidf_ids, tfidf_scores = self.tfidf_rec.predict_topk(text, k=candidate_k, boosts=boosts, negatives=negatives)
        emb_ids, emb_scores = self.emb_rec.predict_topk(text, k=candidate_k, boosts=boosts, negatives=negatives)

        # Fuse scores
        fused_ids, fused_scores = self._fuse_scores(tfidf_ids, tfidf_scores, emb_ids, emb_scores)

        # Return top-k
        return fused_ids[:k], fused_scores[:k]

    def predict_adaptive(self, text: str, threshold: float = 0.3, boosts: dict | None = None, negatives: dict | None = None):
        # Get larger candidate pools from both models
        candidate_k = 20

        tfidf_ids, tfidf_scores = self.tfidf_rec.predict_topk(text, k=candidate_k, boosts=boosts, negatives=negatives)
        emb_ids, emb_scores = self.emb_rec.predict_topk(text, k=candidate_k, boosts=boosts, negatives=negatives)

        # Fuse scores
        fused_ids, fused_scores = self._fuse_scores(tfidf_ids, tfidf_scores, emb_ids, emb_scores)

        # Apply threshold
        idx = np.where(np.array(fused_scores) >= threshold)[0]
        sorted_idx = idx[np.argsort(-np.array(fused_scores)[idx])]

        return [fused_ids[i] for i in sorted_idx], [fused_scores[i] for i in sorted_idx]
