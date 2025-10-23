# Control Retrieval Suite (CRS)

This repository assembles an evaluation-ready pipeline for mapping short operational artifacts (logs, configuration diffs, tickets) to NIST SP 800-53 Rev. 5 controls. The workflow emphasizes leakage-free data preparation, hybrid retrieval, calibrated scoring, and audit-friendly outputs that can be promoted to OSCAL assessment-results artifacts.

## Quickstart

1. **Review the data contract** in `data/raw/controls.csv` and `data/raw/artifacts.csv`. Ensure columns and types match the expected schema (see below).
2. **Work through the notebooks in order** under `notebooks/`. Each notebook documents its purpose, inputs/outputs, detailed steps, and acceptance checks.
3. **Populate processed artifacts** by executing Notebook `01`. Subsequent notebooks consume its outputs.
4. **Train retrieval models** (Notebooks `03`–`05`) after generating supervised pairs in `02`.
5. **Run inference** with Notebook `06` to write predictions for the held-out test split.
6. **Evaluate and perform ablations** in Notebook `07`, which writes summary metrics to `eval/tables/metrics.csv` and validates leak checks.

All configurable paths and thresholds live in `configs/predict_hybrid.yaml`.

## Data Schema

### `data/raw/controls.csv`
- `control_id`: NIST control identifier (e.g., `AC-2`).
- `family`: Control family abbreviation.
- `title`: Human-readable control title.
- `summary`: Short description used to construct lexical/semantic indices.

### `data/raw/artifacts.csv`
- `artifact_id`: Unique ID for each operational artifact.
- `text`: Artifact text snippet.
- `evidence_type`: Category describing the artifact source (log, ticket, etc.).
- `timestamp`: ISO8601 timestamp or similar; coerced to datetime during preprocessing.
- `partition` (optional): Desired split label (`train`, `dev`, `test`). Missing/blank values are reassigned via 60/20/20 splits with a fixed seed.
- `gold_controls`: Pipe- or comma-delimited list of authoritative control IDs.
- `gold_rationale`: Free-text justification for the gold mapping.

## Notebook Workflow

1. `01_data_prep_and_splits.ipynb` – Validate raw data, enforce partitions, and remove duplicate texts across splits.
2. `02_build_pairs_hard_negatives.ipynb` – Generate positive/negative control pairs per artifact using lexical retrieval.
3. `03_train_bi_encoder.ipynb` – Fine-tune a bi-encoder retriever (`multi-qa-mpnet-base-dot-v1`).
4. `04_train_cross_encoder_and_calibrate.ipynb` – Train a cross-encoder reranker and calibrate probabilities on the dev split.
5. `05_train_cardinality_autok.ipynb` – Fit an Auto-K classifier that predicts how many controls to emit per artifact.
6. `06_predict_unified_pipeline.ipynb` – Execute the end-to-end retrieval, reranking, calibration, and cardinality logic for inference.
7. `07_evaluate_and_ablations.ipynb` – Compute metrics, rerun leakage checks, and document ablation findings.

## Directory Layout

```
crs/
├─ notebooks/                  # Planning notebooks (no executable code committed)
├─ data/
│  ├─ raw/                     # Source CSVs from NIST catalog and labeled artifacts
│  └─ processed/               # Leak-free artifact splits and pair data
├─ models/                     # Saved retrievers, rerankers, calibration, Auto-K assets
├─ outputs/predictions/        # Unified pipeline predictions (dev/test)
├─ eval/tables/                # Evaluation tables written by Notebook 07
├─ configs/                    # Runtime configuration (`predict_hybrid.yaml`)
├─ README.md                   # This guide
└─ LICENSE                     # Repository license
```

## Configuration Highlights (`configs/predict_hybrid.yaml`)
- Paths for raw/processed data, trained models, and evaluation outputs.
- Retrieval depths for BM25 and bi-encoder candidates, fusion weights, and rerank cutoffs.
- Cross-encoder calibration threshold and Auto-K enablement.

## OSCAL Alignment

The pipeline is designed so that `outputs/predictions/*.csv` can be transformed into OSCAL assessment-results assets. When ready, add an `oscal/` folder with a README describing the JSON serialization that maps predictions to the OSCAL assessment-results model.

## Next Steps

- Execute the notebooks to materialize processed data, trained models, and evaluation artifacts.
- Integrate automated tests or scripts (e.g., unit tests around feature extraction) once the core pipeline is coded.
- Extend the repository with scripts or CLIs mirroring the notebook logic for production deployment.
