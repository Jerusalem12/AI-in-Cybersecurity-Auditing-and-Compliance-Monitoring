from __future__ import annotations
from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, control_texts: list[str], control_ids: list[str]): ...
    @abstractmethod
    def save(self, path: str): ...
    @abstractmethod
    def load(self, path: str): ...
    @abstractmethod
    def predict_topk(self, text: str, k: int = 3, boosts: dict | None = None, negatives: dict | None = None):
        """Return (ids, scores) ordered by score desc."""
        ...

    @abstractmethod
    def predict_adaptive(self, text: str, threshold: float = 0.3, boosts: dict | None = None, negatives: dict | None = None):
        """Return (ids, scores) for all controls above threshold, ordered by score desc."""
        ...
