from __future__ import annotations
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

class TFIDFRecommender(BaseRecommender):
    def __init__(self, ngram_range=(1,2), min_df=1, stop_words="english"):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words=stop_words)
        self.X = None
        self.control_ids = []

    def fit(self, control_texts: list[str], control_ids: list[str]):
        self.X = self.vectorizer.fit_transform(control_texts)
        self.control_ids = list(control_ids)
        return self

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "X": self.X, "control_ids": self.control_ids}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.vectorizer = obj["vectorizer"]
        self.X = obj["X"]
        self.control_ids = obj["control_ids"]
        return self

    def predict_topk(self, text: str, k: int = 3, boosts: dict | None = None, negatives: dict | None = None):
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.X).ravel()
        if boosts or negatives:
            vocab = self.vectorizer.vocabulary_
            terms = [t for t in text.lower().split() if t in vocab]
            b = sum(boosts.get(t,0.0) for t in terms) if boosts else 0.0
            n = sum(negatives.get(t,0.0) for t in terms) if negatives else 0.0
            sims = sims + b - n
        idx = np.argsort(-sims)[:k]
        return [self.control_ids[i] for i in idx], sims[idx].tolist()

    def predict_adaptive(self, text: str, threshold: float = 0.3, boosts: dict | None = None, negatives: dict | None = None):
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.X).ravel()
        if boosts or negatives:
            vocab = self.vectorizer.vocabulary_
            terms = [t for t in text.lower().split() if t in vocab]
            b = sum(boosts.get(t,0.0) for t in terms) if boosts else 0.0
            n = sum(negatives.get(t,0.0) for t in terms) if negatives else 0.0
            sims = sims + b - n
        # Get all controls above threshold, sorted by score
        idx = np.where(sims >= threshold)[0]
        sorted_idx = idx[np.argsort(-sims[idx])]
        return [self.control_ids[i] for i in sorted_idx], sims[sorted_idx].tolist()
