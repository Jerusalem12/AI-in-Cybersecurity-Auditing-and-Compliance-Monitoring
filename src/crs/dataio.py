from __future__ import annotations
import pandas as pd
from pathlib import Path

REQUIRED_ARTIFACT_COLS = [
    "artifact_id","text","evidence_type","timestamp","gold_controls","split","gold_rationale"
]
REQUIRED_CONTROL_COLS = ["control_id","family","title","summary"]

def load_artifacts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_ARTIFACT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"artifacts.csv missing columns: {missing}")
    df["artifact_id"] = df["artifact_id"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    df["gold_controls"] = df["gold_controls"].fillna("").astype(str)
    df["split"] = df["split"].fillna("").astype(str)
    return df

def load_controls(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CONTROL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"controls.csv missing columns: {missing}")
    return df

def get_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in {"train","dev","test",""}:
        raise ValueError("split must be train|dev|test|''")
    return df if not split else df[df["split"] == split].copy()

def parse_gold(s: str) -> list[str]:
    return [x.strip() for x in s.split(";") if x.strip()]

def save_predictions(rows: list[dict], out_path: str | Path):
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(p, index=False)
