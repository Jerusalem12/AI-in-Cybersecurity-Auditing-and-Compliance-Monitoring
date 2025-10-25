.PHONY: setup format test clean
.PHONY: index-tfidf recommend-tfidf eval-tfidf
.PHONY: index-embeddings recommend-embeddings eval-embeddings
.PHONY: index-hybrid recommend-hybrid eval-hybrid
.PHONY: all-tfidf all-embeddings all-hybrid compare

PYTHON_BIN:=$(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
ifeq ($(PYTHON_BIN),)
$(error Neither python nor python3 found on PATH. Please install Python 3.)
endif

VENV_DIR=.venv
PYTHON=$(VENV_DIR)/bin/python
CONFIG_TFIDF=configs/tfidf.yaml
CONFIG_EMBEDDINGS=configs/embeddings.yaml
CONFIG_HYBRID=configs/hybrid.yaml
CONFIG_EVAL=configs/eval.yaml

$(PYTHON):
	$(PYTHON_BIN) -m venv $(VENV_DIR)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

setup: $(PYTHON)
	@echo "Environment ready. Run: make all-tfidf or make all-embeddings or make compare"

format: $(PYTHON)
	$(PYTHON) -m pre_commit run --all-files || true

test: $(PYTHON)
	$(PYTHON) -m pytest -q

# TF-IDF pipeline
index-tfidf: $(PYTHON)
	$(PYTHON) -m src.cli.build_index --config ${CONFIG_TFIDF}

recommend-tfidf: $(PYTHON)
	$(PYTHON) -m src.cli.recommend --config ${CONFIG_TFIDF} \
	  --in data/raw/artifacts.csv --out outputs/predictions/tfidf/test.csv --split test

eval-tfidf: $(PYTHON)
	$(PYTHON) -m src.cli.evaluate --config ${CONFIG_EVAL} \
	  --pred outputs/predictions/tfidf/test.csv --set_metrics
	@if [ -f eval/tables/metrics.csv ]; then \
	  mv eval/tables/metrics.csv eval/tables/metrics_tfidf.csv; \
	  echo "✓ Saved metrics to eval/tables/metrics_tfidf.csv"; \
	fi

all-tfidf: index-tfidf recommend-tfidf eval-tfidf
	@echo "✓ TF-IDF pipeline complete"

# Embeddings pipeline
index-embeddings: $(PYTHON)
	$(PYTHON) -m src.cli.build_index --config ${CONFIG_EMBEDDINGS}

recommend-embeddings: $(PYTHON)
	$(PYTHON) -m src.cli.recommend --config ${CONFIG_EMBEDDINGS} \
	  --in data/raw/artifacts.csv --out outputs/predictions/embeddings/test.csv --split test

eval-embeddings: $(PYTHON)
	$(PYTHON) -m src.cli.evaluate --config ${CONFIG_EVAL} \
	  --pred outputs/predictions/embeddings/test.csv --set_metrics
	@if [ -f eval/tables/metrics.csv ]; then \
	  mv eval/tables/metrics.csv eval/tables/metrics_embeddings.csv; \
	  echo "✓ Saved metrics to eval/tables/metrics_embeddings.csv"; \
	fi

all-embeddings: index-embeddings recommend-embeddings eval-embeddings
	@echo "✓ Embeddings pipeline complete"

# Hybrid pipeline
index-hybrid: $(PYTHON)
	$(PYTHON) -m src.cli.build_index --config ${CONFIG_HYBRID}

recommend-hybrid: $(PYTHON)
	$(PYTHON) -m src.cli.recommend --config ${CONFIG_HYBRID} \
	  --in data/raw/artifacts.csv --out outputs/predictions/hybrid/test.csv --split test

eval-hybrid: $(PYTHON)
	$(PYTHON) -m src.cli.evaluate --config ${CONFIG_EVAL} \
	  --pred outputs/predictions/hybrid/test.csv --set_metrics
	@if [ -f eval/tables/metrics.csv ]; then \
	  mv eval/tables/metrics.csv eval/tables/metrics_hybrid.csv; \
	  echo "✓ Saved metrics to eval/tables/metrics_hybrid.csv"; \
	fi

all-hybrid: index-hybrid recommend-hybrid eval-hybrid
	@echo "✓ Hybrid pipeline complete"

# Compare all three models
compare: all-tfidf all-embeddings all-hybrid
	@echo ""
	@echo "======================================================================"
	@echo "Model Comparison Results"
	@echo "======================================================================"
	@echo ""
	@echo "TF-IDF Metrics:"
	@cat eval/tables/metrics_tfidf.csv | column -t -s,
	@echo ""
	@echo "Embeddings Metrics:"
	@cat eval/tables/metrics_embeddings.csv | column -t -s,
	@echo ""
	@echo "Hybrid (60% Emb + 40% TF-IDF) Metrics:"
	@cat eval/tables/metrics_hybrid.csv | column -t -s,
	@echo ""
	@echo "Files saved:"
	@echo "  - eval/tables/metrics_tfidf.csv"
	@echo "  - eval/tables/metrics_embeddings.csv"
	@echo "  - eval/tables/metrics_hybrid.csv"
	@echo ""

clean:
	rm -rf data/interim data/processed models outputs eval/plots
	mkdir -p outputs/oscal outputs/predictions/tfidf outputs/predictions/embeddings outputs/predictions/hybrid eval/tables eval/plots
