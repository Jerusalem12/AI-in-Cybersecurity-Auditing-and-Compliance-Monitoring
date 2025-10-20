from __future__ import annotations
import argparse, yaml
from pathlib import Path
from crs.dataio import load_controls
from crs.controls import build_index_text
from crs.recommenders.tfidf import TFIDFRecommender
from crs.recommenders.embeddings import EmbeddingRecommender
from crs.utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    model_type = cfg.get("model", "tfidf").lower()
    controls_path = cfg["paths"]["controls"]
    model_dir = Path(cfg["paths"]["models_dir"])
    ensure_dir(model_dir)
    controls = load_controls(controls_path)
    texts = build_index_text(controls)
    cids = controls["control_id"].tolist()

    if model_type == "embeddings":
        rec = EmbeddingRecommender(model_name=cfg.get("embeddings", {}).get("model_name")).fit(texts, cids)
        out = model_dir / "emb_index.pkl"
    else:
        rec = TFIDFRecommender(
            ngram_range=tuple(cfg.get("tfidf", {}).get("ngram_range", [1,2])),
            min_df=cfg.get("tfidf", {}).get("min_df", 1),
        ).fit(texts, cids)
        out = model_dir / "tfidf_index.pkl"

    rec.save(out)
    print(f"Saved {model_type} index to {out}")

if __name__ == "__main__":
    main()
