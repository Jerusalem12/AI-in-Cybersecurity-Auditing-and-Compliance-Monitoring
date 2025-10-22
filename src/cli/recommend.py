from __future__ import annotations
import argparse, yaml
from pathlib import Path
import pandas as pd
from crs.dataio import load_artifacts, save_predictions
from crs.recommenders.tfidf import TFIDFRecommender
from crs.recommenders.embeddings import EmbeddingRecommender
from crs.recommenders.hybrid import HybridRecommender
from crs.postrules import apply_post_rules, Rule

def load_rules(rule_path: str | None):
    if not rule_path or not Path(rule_path).exists():
        return []
    cfg = yaml.safe_load(Path(rule_path).read_text())
    return [
        Rule(
            control_id=r["control_id"],
            any_keywords=r.get("any_keywords", []),
            all_keywords=r.get("all_keywords", []),
            negative_keywords=r.get("negative_keywords", []),
            boost=float(r.get("boost", 0.08)),
            dampen=float(r.get("dampen", 0.06)),
        )
        for r in cfg.get("rules", [])
    ]

def select_auto_k(ids, scores, etype: str, params: dict) -> tuple[list[str], list[float], str]:
    """
    Keep items above min_score, detect elbow by drop_ratio, clamp to [min_k, max_k] and per-type max.
    Returns (ids, scores, note).
    """
    if not ids:
        return ids, scores, "no-candidates"

    # Ensure sorted by score desc
    pairs = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    ids  = [p[0] for p in pairs]
    sc   = [float(p[1]) for p in pairs]

    min_k      = int(params.get("min_k", 1))
    max_k      = int(params.get("max_k", 5))
    min_score  = float(params.get("min_score", 0.15))
    drop_ratio = float(params.get("drop_ratio", 0.25))
    per_type   = params.get("per_type_max_k", {}) or {}
    type_cap   = int(per_type.get(etype, max_k))
    max_k      = min(max_k, type_cap)

    # 1) absolute threshold
    keep_idx = [i for i, s in enumerate(sc) if s >= min_score]
    if keep_idx:
        cutoff = max(keep_idx) + 1
    else:
        cutoff = min_k  # fallback

    # 2) elbow (score drop) within the first 'cutoff' items
    #    if big relative drop occurs earlier, prefer that earlier cutoff.
    best_cut = cutoff
    for i in range(1, min(cutoff, len(sc))):
        prev, cur = sc[i-1], sc[i]
        if prev <= 1e-9:
            continue
        drop = (prev - cur) / prev
        if drop >= drop_ratio:
            best_cut = max(i, min_k)
            break

    # 3) clamp final K
    k = max(min(best_cut, max_k), min_k)
    note = f"auto_k={k} (etype={etype}, cutoff={cutoff}, elbow={best_cut}, max_k={max_k})"
    return ids[:k], sc[:k], note

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="artifacts.csv")
    ap.add_argument("--out", required=True, help="predictions csv path")
    ap.add_argument("--split", default="test", help="train|dev|test or '' for all")
    ap.add_argument("--top_n", type=int, default=None, help="override stored length when auto_k disabled")
    ap.add_argument("--candidate_k", type=int, default=None, help="expand pool before rules")
    ap.add_argument("--min_score", type=float, default=None, help="legacy threshold when auto_k disabled")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_type = cfg.get("model", "tfidf").lower()
    model_dir  = Path(cfg["paths"]["models_dir"])
    rules_enabled = bool(cfg.get("rules", {}).get("enabled", False))
    rules = load_rules(cfg["paths"].get("rules")) if rules_enabled else []

    rec_cfg = cfg.get("recommend", {})
    candidate_k = args.candidate_k or rec_cfg.get("candidate_k", 10)
    top_n = args.top_n or rec_cfg.get("top_n", 10)
    min_score = rec_cfg.get("min_score", 0.0) if args.min_score is None else args.min_score

    auto_cfg = rec_cfg.get("auto_k", {})
    auto_enabled = bool(auto_cfg.get("enabled", False))

    # load recommender
    if model_type == "embeddings":
        rec = EmbeddingRecommender().load(model_dir / "emb_index.pkl")
    elif model_type == "hybrid":
        rec = HybridRecommender().load(model_dir / "hybrid_index.pkl")
    else:
        rec = TFIDFRecommender().load(model_dir / "tfidf_index.pkl")

    arts = load_artifacts(args.inp)
    if args.split:
        arts = arts[arts["split"] == args.split].copy()

    rows=[]
    for _, r in arts.iterrows():
        ids, scores = rec.predict_topk(r["text"], k=candidate_k)
        notes = []

        if rules:
            ids, scores, rule_notes = apply_post_rules(r["text"], ids, scores, rules)
            notes.extend(rule_notes)

        if auto_enabled:
            ids, scores, why = select_auto_k(ids, scores, str(r.get("evidence_type", "default")), auto_cfg)
            notes.append(why)
        else:
            # legacy fixed length: threshold then take top_n
            pairs = [(i, s) for i, s in zip(ids, scores) if s >= float(min_score)]
            pairs.sort(key=lambda x: x[1], reverse=True)
            ids = [p[0] for p in pairs[:top_n]]
            scores = [p[1] for p in pairs[:top_n]]
            notes.append(f"fixed_top_n={len(ids)} (min_score={min_score})")

        rows.append({
            "artifact_id": int(r["artifact_id"]),
            "text": r["text"],
            "gold_controls": r["gold_controls"],
            "predicted_topk": ";".join(ids),           # variable length now
            "scores_topk": ";".join(f"{s:.4f}" for s in scores),
            "postrule_notes": " | ".join(notes),
            "evidence_type": r.get("evidence_type", "default"),
        })
    save_predictions(rows, args.out)
    mode = "auto-k" if auto_enabled else "fixed"
    print(f"Wrote predictions to {args.out} (mode={mode}, candidate_k={candidate_k}, rules={bool(rules)})")

if __name__ == "__main__":
    main()
