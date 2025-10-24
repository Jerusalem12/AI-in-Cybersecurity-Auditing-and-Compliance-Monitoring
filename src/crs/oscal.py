from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from .utils import now_iso, ensure_dir

def preds_to_oscal(preds_csv: str, out_path: str, model_id: str = "CRS-TFIDF-0.1"):
    """
    Convert a predictions CSV (artifact_id,text,gold_controls,predicted_topk,scores_topk)
    into a tiny OSCAL-like assessment-results JSON.
    """
    df = pd.read_csv(preds_csv)
    ts = now_iso()
    results=[]
    for _, r in df.iterrows():
        preds = r["predicted_topk"].split(";")
        scores = [float(x) for x in r["scores_topk"].split(";")]
        obs=[]
        for cid, sc in zip(preds, scores):
            obs.append({
                "control_id": cid,
                "confidence": round(float(sc), 4),
                "artifact_ref": f"artifact:{int(r['artifact_id'])}",
                "evidence_types": ["log","config","ticket","policy"],
                "timestamp": ts,
                "model": {"id": model_id, "version": "0.1"},
                "explanation": "Cosine similarity over control title+summary."
            })
        results.append({
            "uuid": hashlib.md5(str(r["artifact_id"]).encode()).hexdigest(),
            "title": "Artifact-to-control mapping",
            "description": r["text"],
            "observations": obs
        })
    bundle={
        "uuid": hashlib.sha256(f"{ts}-{model_id}".encode()).hexdigest()[:32],
        "type": "assessment-results",
        "metadata": {"title":"CRS assessment results","last-modified": ts},
        "results": results
    }
    p = Path(out_path); ensure_dir(p.parent)
    p.write_text(json.dumps(bundle, indent=2))
    return out_path
