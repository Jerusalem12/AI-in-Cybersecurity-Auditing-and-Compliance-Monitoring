from __future__ import annotations
from collections import defaultdict
import pandas as pd
from .dataio import parse_gold

def learn_from_train(train_df: pd.DataFrame, recommender, rounds: int = 1, alpha=0.03, beta=0.01):
    """Lightweight heuristic: boost terms if any correct suggestion in top5; penalize otherwise."""
    boosts = defaultdict(float); negatives = defaultdict(float)
    for _ in range(rounds):
        for _, r in train_df.iterrows():
            gold = set(parse_gold(r["gold_controls"]))
            preds, _ = recommender.predict_topk(r["text"], k=5)
            toks = r["text"].lower().split()
            if gold.intersection(preds):
                for t in toks: boosts[t] += alpha
            for cid in preds:
                if cid not in gold:
                    for t in toks: negatives[t] += beta/5.0
    return dict(boosts), dict(negatives)

def aggregate_auditor_feedback(feedback_csv: str) -> tuple[dict, dict]:
    """
    From feedback.csv, build token-level boosts/negatives.
    accept -> boost; reject -> negative; add -> small boost.
    """
    df = pd.read_csv(feedback_csv)
    boosts = defaultdict(float); negatives = defaultdict(float)
    for _, r in df.iterrows():
        action = str(r.get("auditor_action","")).lower()
        explanation = str(r.get("explanation",""))
        toks = [t for t in explanation.lower().split() if t.isalpha() or len(t) > 2]
        if action == "accept":
            for t in toks: boosts[t] += 0.05
        elif action == "add":
            for t in toks: boosts[t] += 0.02
        elif action == "reject":
            for t in toks: negatives[t] += 0.05
    return dict(boosts), dict(negatives)
