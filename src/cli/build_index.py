from __future__ import annotations
import argparse, yaml
from pathlib import Path
from crs.dataio import load_controls
from crs.controls import build_index_text
from crs.recommenders import TFIDFRecommender
from crs.utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    controls_path = cfg["paths"]["controls"]
    model_dir = Path(cfg["paths"]["models_dir"])
    ensure_dir(model_dir)

    controls = load_controls(controls_path)
    texts = build_index_text(controls)
    rec = TFIDFRecommender(
        ngram_range=tuple(cfg.get("tfidf", {}).get("ngram_range", [1,2])),
        min_df=cfg.get("tfidf", {}).get("min_df", 1),
    ).fit(texts, controls["control_id"].tolist())
    rec.save(model_dir / "tfidf_index.pkl")
    print(f"Saved TF-IDF index to {model_dir/'tfidf_index.pkl'}")

if __name__ == "__main__":
    main()
