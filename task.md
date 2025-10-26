

# Project context (read this first)

* **Goal:** Map short operational **artifacts** (logs, config diffs, tickets) to **NIST SP 800-53 Rev. 5** controls and produce **OSCAL assessment-results–ready** evidence. Controls come from the official NIST catalog; outputs should align with the OSCAL Assessment Results model to keep evidence portable and auditable. ([NIST Computer Security Resource Center][1])
* **Modeling approach (no rules):**

  1. **Hybrid retriever** = lexical **BM25** + semantic **bi-encoder** retrieval; fuse scores for top-K candidate controls. ([GitHub][2])
  2. **Cross-encoder reranker** adds precise, context-aware scoring over the top-K pairs. ([Hugging Face][3])
  3. **Auto-K cardinality + calibration** decides how many controls (1–3) to return per artifact using calibrated probabilities; no heuristic rules. ([NIST Pages][4])
* **No leakage:** Training uses **train/dev** only; evaluation is on **held-out test** split with duplicate-text checks across partitions to prevent memorization.

---

# Notebook suite (create under `notebooks/`, in this order)

> Each notebook lists **Purpose**, **Inputs/Outputs**, **Steps**, and **Acceptance checks** for the agent. No code is needed here.

## 01_data_prep_and_splits.ipynb

**Purpose:** Load raw CSVs, validate schema, create **train/dev/test** splits, enforce **no leakage**.
**Inputs:** `data/raw/controls.csv` (columns: `control_id,family,title,summary`), `data/raw/artifacts.csv` (columns: `artifact_id,text,evidence_type,timestamp,partition?,gold_controls,gold_rationale`).
**Outputs:** `data/processed/artifacts_with_split.csv` + inline tables.
**Steps:**

1. Validate required columns, coerce types.
2. If `partition` missing/empty → assign **60/20/20** split with fixed seed.
3. **Leakage guard:** create a lowercase/trimmed **text hash**; if identical text appears in multiple partitions, keep the first occurrence (prefer **train**) and remove/move duplicates so each unique text appears in **exactly one** partition. Drop the hash afterward.
4. Save processed CSV; print per-partition counts and `evidence_type × partition`.
   **Acceptance checks:** partitions limited to `train/dev/test`; cross-partition duplicate text count == **0**; file saved at the specified path.

---

## 02_build_pairs_hard_negatives.ipynb

**Purpose:** Build **pairwise training data** from artifacts for supervised learning: positives (gold controls) + **hard negatives** from a weak retriever.
**Inputs:** `data/raw/controls.csv`, `data/processed/artifacts_with_split.csv` (from 01).
**Outputs:** `data/processed/pairs/train.jsonl`, `.../dev.jsonl`, `.../test.jsonl`.
**Pair schema:** `artifact_id, artifact_text, evidence_type, control_id, control_text, family, label (1/0)`.
**Steps:**

1. Make control **index text**: `title + ". " + summary`.
2. Use **BM25 or TF-IDF** to fetch top-K candidates (e.g., 32) per artifact. ([GitHub][2])
3. **Positives:** all gold controls; **hard negatives:** top-K non-gold candidates.
4. Write pairs by artifact partition (train/dev/test).
   **Acceptance checks:** no **test** artifact IDs appear in **train/dev** pair files; each train artifact yields ≥1 positive pair.

---

## 03_train_bi_encoder.ipynb

**Purpose:** Fine-tune a **bi-encoder** (contrastive) for semantic retrieval over controls.
**Inputs:** `data/processed/pairs/train.jsonl`, `.../dev.jsonl`.
**Outputs:** `models/bi_encoder/` (model + tokenizer) and, optionally, `models/bi_encoder/controls_embeddings.npy`.
**Recipe:** base **`sentence-transformers/multi-qa-mpnet-base-dot-v1`**; MultipleNegativesRankingLoss; batch 64 (grad-acc if needed), epochs 3–5, LR 2e-5 (10% warmup). Track **dev MRR@10**. ([Hugging Face][5])
**Acceptance checks:** model saved; best checkpoint noted; dev MRR@10 printed.

---

## 04_train_cross_encoder_and_calibrate.ipynb

**Purpose:** Train a **cross-encoder reranker** on pair labels and **calibrate** its probabilities on dev (never test).
**Inputs:** `data/processed/pairs/train.jsonl`, `.../dev.jsonl`.
**Outputs:** `models/cross_encoder/` and `models/calibration/cross_iso.pkl`.
**Recipe:** base **`cross-encoder/ms-marco-MiniLM-L6-v2`** (or L-2-v2), BCE loss, batch 32, epochs 2–3, LR 1.5e-5; fit **isotonic/Platt** calibration on dev. ([Hugging Face][3])
**Acceptance checks:** calibration file saved; dev AUC/MAP printed; reliability curve or ECE summary reported.

---

## 05_train_cardinality_autok.ipynb

**Purpose:** Train an **Auto-K** classifier to predict how many controls (1/2/3) to return per artifact after reranking.
**Inputs:** pair files + rerank scores (derived in-notebook).
**Outputs:** `models/cardinality/model.pkl`, `models/cardinality/feature_spec.json`.
**Features per artifact:** top probabilities `[s1..s4]` (pad zeros), deltas `[s1−s2, s2−s3]`, score entropy, `evidence_type` one-hot.
**Label:** `min(3, |gold_controls|)` per artifact from the processed split file.
**Acceptance checks:** dev accuracy & confusion matrix printed; model and feature spec saved.

---

## 06_predict_unified_pipeline.ipynb

**Purpose:** End-to-end inference **without rules**: **Hybrid retrieval** → **Cross-encoder rerank (calibrated)** → **Auto-K** → write predictions.
**Inputs:** `data/raw/controls.csv`, `data/processed/artifacts_with_split.csv`, trained models from 03–05.
**Outputs:** `outputs/predictions/test.csv` (evaluate only on **test**); optional `.../dev.csv`.
**Prediction schema:** `artifact_id,text,gold_controls,predicted_topk,scores_topk,explanations`

* `predicted_topk`: semicolon list of control_ids (length decided by Auto-K)
* `scores_topk`: semicolon list of **calibrated probabilities**
* `explanations`: brief note on which model(s) contributed (e.g., “hybrid fuse + cross-enc prob”)—purely informational; no rules.
  **Pipeline policy:**

1. **Retrieve (K=64):** BM25 score + bi-encoder dot-product (controls pre-encoded by 03). Normalize and **fuse** (e.g., 0.6*bi + 0.4*bm25). ([GitHub][2])
2. **Rerank (K→32):** cross-encoder → calibrated probabilities `p(control|artifact)`; optionally blend with normalized fused retrieval (e.g., 0.7 * p_cross + 0.3 * fused). ([Hugging Face][3])
3. **Auto-K:** predict K ∈ {1,2,3}; return top-K by calibrated prob; enforce a **min calibrated prob** per evidence_type if desired (e.g., ≥0.35); **no rules** or manual nudges.
4. **Write CSVs** for **test** (and dev for sanity).
   **Guardrails:** This notebook **must not** read `gold_controls` while ranking; only attach golds for later evaluation.
   **Acceptance checks:** average predictions per artifact ≈ 1–2.x; lengths ∈ {1,2,3}; no use of golds in ranking (note this explicitly in markdown).

---

## 07_evaluate_and_ablations.ipynb

**Purpose:** Final metrics + ablations + integrity checks (prove there’s no overfit/leakage).
**Inputs:** `data/processed/artifacts_with_split.csv`; `outputs/predictions/test.csv`; alt predictions for ablations.
**Outputs:** `eval/tables/metrics.csv` + inline tables/plots.
**Metrics:** Top-1, P@k/R@k/Jaccard@k (k∈{1,3,5}), Set-Precision/Recall/F1 (Auto-K), MRR, MAP, **per-family** precision/recall (join controls to family). NIST families per 800-53 provide the taxonomy. ([NIST Computer Security Resource Center][1])
**Ablations:**

* BM25-only; Bi-encoder-only; Hybrid (no cross-enc); Hybrid + Cross (no Auto-K, fixed-k=3); **Unified** (Hybrid + Cross + Auto-K).
  **Leakage checks (must run and print):**

1. Partitions present in predictions (should be only **test**).
2. 5 random rows: show `artifact_id`, gold set vs predicted set.
3. Duplicate texts across partitions = **0** (recompute via hash).
4. **Adversarial shuffle:** shuffle gold labels and recompute metrics → numbers **collapse**.
5. **Calibration reliability:** bin predicted probs; plot expected vs observed accuracy; report ECE.
   **Acceptance checks:** `eval/tables/metrics.csv` exists with all metrics; ablation table summarizes each variant; leakage checks printed and pass.

---

## Minimal config notes for the agent (no rules file needed)

* **BM25 backend:** use a standard Python BM25 library (e.g., `rank_bm25`) or a fast variant (e.g., BM25S). Save index artifacts under `models/bm25/`. ([GitHub][2])
* **Bi-encoder:** base `sentence-transformers/multi-qa-mpnet-base-dot-v1`. Save under `models/bi_encoder/`. ([Hugging Face][5])
* **Cross-encoder:** base `cross-encoder/ms-marco-MiniLM-L6-v2` (or L-2). Save under `models/cross_encoder/`; calibrator under `models/calibration/cross_iso.pkl`. ([Hugging Face][3])
* **Auto-K classifier:** simple Logistic Regression or LightGBM; save under `models/cardinality/`.
* **OSCAL (optional export later):** when converting predictions to evidence, follow the **Assessment Results** model structure (subject, observation, risk, actors). ([NIST Pages][4])

---

## What to tell your agent (verbatim prompts)

1. **Create seven notebooks** exactly as specified above under `notebooks/`. Use the file paths and acceptance checks verbatim.
2. **Run in order**: 01 → 02 → 03 → 04 → 05 → 06 → 07.
3. **Stop on any failed acceptance check**, fix, and rerun.
4. During **prediction (06)**, do **not** read or use `gold_controls` for scoring—only attach them to outputs for evaluation.
5. Save all outputs to the exact paths listed; do **not** invent new file names or directories.

---

### Why this is robust (and publishable)

* NIST SP 800-53 Rev. 5 provides the control taxonomy you’re mapping to; OSCAL offers a standard for evidence outputs. ([NIST Computer Security Resource Center][1])
* Hybrid retrieval (BM25 + bi-encoder) maximizes recall, cross-encoder reranking maximizes precision, and Auto-K yields the **right number** of controls—**without** hand-written rules. ([GitHub][2])

If you want, I can also add a brief one-pager for **OSCAL evidence export** (field names and mapping from predictions) to slot in after Notebook 07.

[1]: https://csrc.nist.gov/pubs/sp/800/53/r5/final?utm_source=chatgpt.com "NIST SP 800-53 Rev. 5 Security Controls"
[2]: https://github.com/dorianbrown/rank_bm25?utm_source=chatgpt.com "dorianbrown/rank_bm25: A Collection of BM25 Algorithms ..."
[3]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2?utm_source=chatgpt.com "cross-encoder/ms-marco-MiniLM-L6-v2"
[4]: https://pages.nist.gov/OSCAL/learn/concepts/layer/assessment/assessment-results/?utm_source=chatgpt.com "OSCAL Assessment Layer: Assessment Results Model"
[5]: https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1?utm_source=chatgpt.com "sentence-transformers/multi-qa-mpnet-base-dot-v1"


# Repository structure (what to create)

```
crs/
├─ notebooks/
│  ├─ 01_data_prep_and_splits.ipynb
│  ├─ 02_build_pairs_hard_negatives.ipynb
│  ├─ 03_train_bi_encoder.ipynb
│  ├─ 04_train_cross_encoder_and_calibrate.ipynb
│  ├─ 05_train_cardinality_autok.ipynb
│  ├─ 06_predict_unified_pipeline.ipynb
│  └─ 07_evaluate_and_ablations.ipynb
│
├─ data/
│  ├─ raw/
│  │  ├─ controls.csv           # NIST controls: control_id,family,title,summary  :contentReference[oaicite:5]{index=5}
│  │  └─ artifacts.csv          # artifact_id,text,evidence_type,timestamp,partition?,gold_controls,gold_rationale
│  └─ processed/
│     └─ artifacts_with_split.csv
│
├─ models/
│  ├─ bm25/                     # optional cached BM25 artifacts (doc-term stats)  :contentReference[oaicite:6]{index=6}
│  ├─ bi_encoder/               # fine-tuned Sentence-Transformers bi-encoder      :contentReference[oaicite:7]{index=7}
│  ├─ cross_encoder/            # fine-tuned cross-encoder reranker                :contentReference[oaicite:8]{index=8}
│  ├─ calibration/              # Platt/Isotonic calibrator over dev scores        :contentReference[oaicite:9]{index=9}
│  └─ cardinality/              # Auto-K classifier (1/2/3) + feature spec
│
├─ outputs/
│  └─ predictions/
│     ├─ test.csv               # unified pipeline predictions (held-out test)
│     └─ dev.csv                # optional sanity run on dev
│
├─ eval/
│  └─ tables/
│     └─ metrics.csv            # Top-1, P@k/R@k/J@k, Set-F1, MAP/MRR, per-family
│
├─ configs/
│  └─ predict_hybrid.yaml       # paths, retriever K’s, fusion weights, thresholds
│
├─ README.md                    # quickstart + notebook order + data schema
└─ LICENSE
```

> If you later export **OSCAL Assessment Results**, add `oscal/` with a short README and JSON examples referencing the official model. ([NIST Pages][4])

---

# Notebook purposes (for the agent to implement)

1. **01_data_prep_and_splits.ipynb** — Load raw CSVs; create **train/dev/test** (60/20/20) if missing; enforce **no cross-partition duplicate texts** (hash on normalized text); write `data/processed/artifacts_with_split.csv`.
   **Accept:** partitions only {train,dev,test}; duplicate-across-partitions = 0.

2. **02_build_pairs_hard_negatives.ipynb** — Build pairwise data: positives (gold controls) + **hard negatives** from BM25/TF-IDF top-K. Save `data/processed/pairs/{train,dev,test}.jsonl`.
   **Accept:** no test artifact IDs in train/dev pairs.

3. **03_train_bi_encoder.ipynb** — Fine-tune **bi-encoder** (e.g., `sentence-transformers/multi-qa-mpnet-base-dot-v1`) with contrastive loss; pick best by **dev MRR@10**; pre-encode control vectors. ([Hugging Face][5])
   **Accept:** model saved; dev MRR printed.

4. **04_train_cross_encoder_and_calibrate.ipynb** — Train **cross-encoder** (e.g., `ms-marco-MiniLM-L6-v2`) on pairs (BCE), then fit **isotonic/Platt** on dev for calibrated probs. ([Hugging Face][3])
   **Accept:** calibration artifact saved; dev AUC/MAP + reliability/ECE reported.

5. **05_train_cardinality_autok.ipynb** — Train **Auto-K** classifier on features derived from reranked scores (top probs, deltas, entropy, evidence_type one-hot).
   **Accept:** dev accuracy/confusion matrix; model + feature spec saved.

6. **06_predict_unified_pipeline.ipynb** — Full inference (**no rules**): **BM25 + bi-encoder retrieval → cross-encoder rerank (calibrated) → Auto-K** → write `outputs/predictions/test.csv` (only partition=test).
   **Accept:** avg predictions ~1–2.x; lengths ∈ {1,2,3}; **do not** use `gold_controls` in ranking.

7. **07_evaluate_and_ablations.ipynb** — Compute Top-1, P@k/R@k/J@k (k=1/3/5), Set-F1, MAP/MRR, per-family; run **leak checks** (partition coverage, 5 random gold vs pred, duplicate text hash=0, **adversarial label shuffle** collapses metrics).
   **Accept:** `eval/tables/metrics.csv` written; ablation table included.

---

# Minimal config note (the only config file you need)

**`configs/predict_hybrid.yaml` (what the agent should create)**

* Paths to raw/processed data and models.
* Retriever K’s (e.g., retrieve 64; rerank 32), fusion weights (e.g., bi 0.6 / bm25 0.4).
* Calibrated probability threshold (e.g., 0.35) and **Auto-K** on/off.
* No rule settings at all.

---

# Why these choices (for the agent’s understanding)

* **NIST SP 800-53 Rev.5** is the canonical control catalog you map to; keep it as `controls.csv` (id, family, title, summary). ([NIST CSRC][1])
* **OSCAL Assessment Results** is the target evidence schema you can export later for audit-ready outputs. ([NIST Pages][4])
* **BM25** gives strong lexical recall for config/log tokens; **bi-encoder** adds semantic recall; **cross-encoder** boosts precision; **Auto-K + calibration** decides a trustworthy number of controls—cleanly, without hand-authored rules. ([GitHub][2])

If you want, I can also add a one-pager after evaluation describing how to format the predictions into OSCAL **assessment-results** JSON (field-to-field mapping) for your paper appendix.

[1]: https://csrc.nist.gov/pubs/sp/800/53/r5/final?utm_source=chatgpt.com "NIST SP 800-53 Rev. 5 Security Controls"
[2]: https://github.com/dorianbrown/rank_bm25?utm_source=chatgpt.com "dorianbrown/rank_bm25: A Collection of BM25 Algorithms ..."
[3]: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2?utm_source=chatgpt.com "cross-encoder/ms-marco-MiniLM-L6-v2"
[4]: https://pages.nist.gov/OSCAL-Reference/models/v1.1.3/assessment-results/?utm_source=chatgpt.com "Assessment Results Model v1.1.3 Reference - NIST Pages"
[5]: https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1?utm_source=chatgpt.com "sentence-transformers/multi-qa-mpnet-base-dot-v1"
