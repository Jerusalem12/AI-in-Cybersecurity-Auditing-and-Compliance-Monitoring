from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class Rule:
    control_id: str
    any_keywords: List[str]          # if ANY present → boost
    all_keywords: List[str] = None   # if ALL present → stronger boost
    negative_keywords: List[str] = None  # if ANY present → dampen
    boost: float = 0.08              # add to score
    dampen: float = 0.06             # subtract from score

    def apply(self, text: str, scores: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
        notes = []
        t = text.lower()
        any_hit = any(k in t for k in (self.any_keywords or []))
        all_hit = all(k in t for k in (self.all_keywords or [])) if self.all_keywords else False
        neg_hit = any(k in t for k in (self.negative_keywords or []))
        if any_hit:
            scores[self.control_id] = scores.get(self.control_id, 0.0) + self.boost
            notes.append(f"+{self.boost:.2f} {self.control_id} (any:{','.join(self.any_keywords)})")
        if all_hit:
            scores[self.control_id] = scores.get(self.control_id, 0.0) + self.boost
            notes.append(f"+{self.boost:.2f} {self.control_id} (all:{','.join(self.all_keywords)})")
        if neg_hit:
            scores[self.control_id] = scores.get(self.control_id, 0.0) - self.dampen
            notes.append(f"-{self.dampen:.2f} {self.control_id} (neg:{','.join(self.negative_keywords)})")
        return scores, notes

def apply_post_rules(text: str, top_ids: List[str], top_scores: List[float], rules: List[Rule]) -> Tuple[List[str], List[float], List[str]]:
    """Adjust scores of the current top-k set; return possibly re-ordered lists and human notes."""
    score_map = {cid: float(s) for cid, s in zip(top_ids, top_scores)}
    notes_all: List[str] = []
    for r in rules:
        score_map, notes = r.apply(text, score_map)
        notes_all.extend(notes)
    # reorder by updated score
    items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    ids = [cid for cid,_ in items]
    scores = [s for _,s in items]
    return ids, scores, notes_all
