from __future__ import annotations
import argparse, yaml
from pathlib import Path
import pandas as pd
from crs.dataio import load_artifacts
from crs.feedback import learn_from_train, aggregate_auditor_feedback
from crs.recommenders import TFIDFRecommender
from crs.controls import build_index_text
from crs.dataio import load_controls
from crs.utils import write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--feedback_csv", required=False, help="auditor feedback.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    controls = load_controls(cfg["paths"]["controls"])
    rec = TFIDFRecommender().load(Path(cfg["paths"]["models_dir"]) / "tfidf_index.pkl")

    boosts, negatives = {}, {}
    if args.feedback_csv:
        boosts, negatives = aggregate_auditor_feedback(args.feedback_csv)
    else:
        arts = load_artifacts(cfg["paths"]["artifacts"])
        train = arts[arts["split"]=="train"].copy()
        boosts, negatives = learn_from_train(train, rec, rounds=1)

    out_dir = Path(cfg["paths"]["models_dir"])
    write_json(boosts, out_dir / "boosts.json")
    write_json(negatives, out_dir / "negatives.json")
    print(f"Saved boosts/negatives to {out_dir}")

if __name__ == "__main__":
    main()
