from __future__ import annotations
import argparse, yaml
from pathlib import Path
import pandas as pd
from crs.metrics import precision_at_k, recall_at_k, top1_accuracy, jaccard, acceptance_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred", required=False, help="predictions csv; if omitted, uses outputs/predictions/test.csv")
    ap.add_argument("--feedback_csv", required=False)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pred_path = args.pred or (Path(cfg["paths"]["predictions_dir"]) / "test.csv")
    df = pd.read_csv(pred_path)

    k_list = cfg.get("k_list", [3])
    rows=[]
    rows.append({"metric": "top1_accuracy", "value": top1_accuracy(df)})
    for k in k_list:
        rows.append({"metric": f"precision@{k}", "value": precision_at_k(df, k=k)})
        rows.append({"metric": f"recall@{k}", "value": recall_at_k(df, k=k)})
        rows.append({"metric": f"jaccard@{k}", "value": jaccard(df, k=k)})

    if args.feedback_csv and Path(args.feedback_csv).exists():
        fb = pd.read_csv(args.feedback_csv)
        rows.append({"metric":"acceptance_rate", "value": acceptance_rate(fb)})

    out_dir = Path("eval/tables"); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote metrics to {out_csv}")

if __name__ == "__main__":
    main()
