# Compliance Recommendation System (CRS)

AI-assisted mapping of operational artifacts to **NIST SP 800-53 r5** controls, with **OSCAL** evidence output and **auditor feedback**.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -e .
# Put your CSVs here:
mkdir -p data/raw
# copy files: data/raw/controls.csv, data/raw/artifacts.csv
make index
make recommend
make eval
```

## Project Structure

```
.
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ Makefile
├─ .pre-commit-config.yaml
├─ configs/
│  ├─ defaults.yaml
│  └─ eval.yaml
├─ data/              # raw/ (inputs), interim/, processed/
├─ models/            # saved vectorizers/embeddings
├─ src/
│  ├─ crs/            # library code (dataio, recommenders, metrics, oscal)
│  └─ cli/            # scripts: build_index, recommend, evaluate, learn_feedback
├─ outputs/           # predictions, oscal bundles
└─ eval/              # tables and plots
```

## What it does

* **Recommender**: TF-IDF (and optional embeddings) + cosine similarity over control text.
* **Feedback**: accept/reject/add logs � learn boosts/penalties.
* **Metrics**: Top-1 accuracy, P@3/R@3, Jaccard; Acceptance Rate; Time-to-Evidence.
* **Evidence**: Exports OSCAL *assessment-results* JSON.

## Config

See `configs/defaults.yaml` for k, model type, and file paths.

## Citation

Based on class project guidance and *Towards Automated Continuous Security Compliance (ESEM '24)* for problem framing.
