from __future__ import annotations
import argparse, yaml
from pathlib import Path
import pandas as pd
from crs.dataio import load_artifacts, parse_gold, save_predictions
from crs.recommenders import TFIDFRecommender

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="artifacts.csv")
    ap.add_argument("--out", required=True, help="predictions csv path")
    ap.add_argument("--split", default="test", help="train|dev|test or '' for all")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    k = args.k or cfg.get("k", 3)
    model_path = Path(cfg["paths"]["models_dir"]) / "tfidf_index.pkl"

    rec = TFIDFRecommender().load(model_path)
    arts = load_artifacts(args.inp)
    if args.split:
        arts = arts[arts["split"] == args.split].copy()

    rows=[]
    for _, r in arts.iterrows():
        ids, scores = rec.predict_topk(r["text"], k=k)
        rows.append({
            "artifact_id": int(r["artifact_id"]),
            "text": r["text"],
            "gold_controls": r["gold_controls"],
            "predicted_topk": ";".join(ids),
            "scores_topk": ";".join(f"{s:.4f}" for s in scores),
        })
    save_predictions(rows, args.out)
    print(f"Wrote predictions to {args.out} (k={k}, rows={len(rows)})")

if __name__ == "__main__":
    main()
