from __future__ import annotations
import pandas as pd
from .dataio import parse_gold

def precision_at_k(df_preds: pd.DataFrame, k=3) -> float:
    vals=[]
    for _,r in df_preds.iterrows():
        gold=set(parse_gold(r["gold_controls"]))
        pred=r["predicted_topk"].split(";")[:k]
        tp=len(gold.intersection(pred))
        vals.append(tp/k)
    return float(sum(vals)/len(vals)) if vals else 0.0

def recall_at_k(df_preds: pd.DataFrame, k=3) -> float:
    vals=[]
    for _,r in df_preds.iterrows():
        gold=set(parse_gold(r["gold_controls"]))
        pred=r["predicted_topk"].split(";")[:k]
        tp=len(gold.intersection(pred))
        vals.append(tp/len(gold) if gold else 0.0)
    return float(sum(vals)/len(vals)) if vals else 0.0

def top1_accuracy(df_preds: pd.DataFrame) -> float:
    vals=[]
    for _,r in df_preds.iterrows():
        gold=set(parse_gold(r["gold_controls"]))
        top1=r["predicted_topk"].split(";")[0:1]
        vals.append(1.0 if (set(top1) & gold) else 0.0)
    return float(sum(vals)/len(vals)) if vals else 0.0

def jaccard(df_preds: pd.DataFrame, k=3) -> float:
    vals=[]
    for _,r in df_preds.iterrows():
        gold=set(parse_gold(r["gold_controls"]))
        pred=set(r["predicted_topk"].split(";")[:k])
        denom=len(gold|pred)
        vals.append(len(gold&pred)/denom if denom>0 else 0.0)
    return float(sum(vals)/len(vals)) if vals else 0.0

def acceptance_rate(df_feedback: pd.DataFrame) -> float:
    if df_feedback.empty: return 0.0
    accepted=(df_feedback["auditor_action"].str.lower()=="accept").sum()
    total=len(df_feedback)
    return float(accepted/total) if total else 0.0

def time_to_evidence_reduction(manual_seconds: list[int], assisted_seconds: list[int]) -> float:
    if not manual_seconds or not assisted_seconds: return 0.0
    import numpy as np
    m=float(np.median(manual_seconds))
    a=float(np.median(assisted_seconds))
    return float((m-a)/m) if m>0 else 0.0
