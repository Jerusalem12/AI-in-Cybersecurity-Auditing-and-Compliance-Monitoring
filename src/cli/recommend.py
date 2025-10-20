from __future__ import annotations
import argparse, yaml
from pathlib import Path
import pandas as pd
from crs.dataio import load_artifacts, parse_gold, save_predictions
from crs.recommenders.tfidf import TFIDFRecommender
from crs.recommenders.embeddings import EmbeddingRecommender
from crs.postrules import apply_post_rules, Rule

def load_rules(rule_path: str | None) -> list[Rule]:
    if not rule_path or not Path(rule_path).exists():
        return []
    import yaml
    cfg = yaml.safe_load(Path(rule_path).read_text())
    rules = []
    for r in cfg.get("rules", []):
        rules.append(Rule(
            control_id=r["control_id"],
            any_keywords=r.get("any_keywords", []),
            all_keywords=r.get("all_keywords", []),
            negative_keywords=r.get("negative_keywords", []),
            boost=float(r.get("boost", 0.08)),
            dampen=float(r.get("dampen", 0.06)),
        ))
    return rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="artifacts.csv")
    ap.add_argument("--out", required=True, help="predictions csv path")
    ap.add_argument("--split", default="test", help="train|dev|test or '' for all")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    k = args.k or cfg.get("recommend", {}).get("k_pred") or cfg.get("k", 3)
    model_type = cfg.get("model", "tfidf").lower()
    model_dir = Path(cfg["paths"]["models_dir"])

    if model_type == "embeddings":
        rec = EmbeddingRecommender().load(model_dir / "emb_index.pkl")
    else:
        rec = TFIDFRecommender().load(model_dir / "tfidf_index.pkl")

    arts = load_artifacts(args.inp)
    if args.split:
        arts = arts[arts["split"] == args.split].copy()

    rules_enabled = bool(cfg.get("rules", {}).get("enabled", False))
    rules = load_rules(cfg["paths"].get("rules")) if rules_enabled else []

    print(f"Rules enabled in config: {rules_enabled}")
    print(f"Rules loaded: {len(rules)}")
    if rules:
        print(f"Sample rules: {[r.control_id for r in rules[:3]]}")

    rows=[]
    rules_applied_count = 0
    for _, r in arts.iterrows():
        ids, scores = rec.predict_topk(r["text"], k=k)
        notes = []
        if rules:
            ids, scores, notes = apply_post_rules(r["text"], ids, scores, rules)
            ids, scores = ids[:k], scores[:k]
            if notes:
                rules_applied_count += 1
        rows.append({
            "artifact_id": int(r["artifact_id"]),
            "text": r["text"],
            "gold_controls": r["gold_controls"],
            "predicted_topk": ";".join(ids),
            "scores_topk": ";".join(f"{s:.4f}" for s in scores),
            "postrule_notes": " | ".join(notes)
        })
    save_predictions(rows, args.out)
    print(f"Rules applied to {rules_applied_count}/{len(rows)} artifacts")
    print(f"Wrote predictions to {args.out} (k={k}, rows={len(rows)}, rules={'enabled' if rules else 'disabled'})")

if __name__ == "__main__":
    main()
