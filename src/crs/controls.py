from __future__ import annotations
import pandas as pd

def build_index_text(controls: pd.DataFrame) -> list[str]:
    # Concatenate title + summary for retrieval
    return (controls["title"].fillna("") + ". " + controls["summary"].fillna("")).tolist()
