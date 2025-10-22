.PHONY: setup format test clean
.PHONY: index-tfidf recommend-tfidf eval-tfidf
.PHONY: index-embeddings recommend-embeddings eval-embeddings
.PHONY: all-tfidf all-embeddings compare

VENV=. .venv/bin/activate
CONFIG_TFIDF=configs/tfidf.yaml
CONFIG_EMBEDDINGS=configs/embeddings.yaml
CONFIG_EVAL=configs/eval.yaml

setup:
	python -m venv .venv && ${VENV} && pip install -U pip && pip install -e .
	@echo "Environment ready. Run: make all-tfidf or make all-embeddings or make compare"

format:
	pre-commit run --all-files || true

test:
	${VENV} && pytest -q

# TF-IDF pipeline
index-tfidf:
	${VENV} && python -m src.cli.build_index --config ${CONFIG_TFIDF}

recommend-tfidf:
	${VENV} && python -m src.cli.recommend --config ${CONFIG_TFIDF} \
	  --in data/raw/artifacts.csv --out outputs/predictions/tfidf/test.csv --split test

eval-tfidf:
	${VENV} && python -m src.cli.evaluate --config ${CONFIG_EVAL} \
	  --pred outputs/predictions/tfidf/test.csv --set_metrics
	@if [ -f eval/tables/metrics.csv ]; then \
	  mv eval/tables/metrics.csv eval/tables/metrics_tfidf.csv; \
	  echo "✓ Saved metrics to eval/tables/metrics_tfidf.csv"; \
	fi

all-tfidf: index-tfidf recommend-tfidf eval-tfidf
	@echo "✓ TF-IDF pipeline complete"

# Embeddings pipeline
index-embeddings:
	${VENV} && python -m src.cli.build_index --config ${CONFIG_EMBEDDINGS}

recommend-embeddings:
	${VENV} && python -m src.cli.recommend --config ${CONFIG_EMBEDDINGS} \
	  --in data/raw/artifacts.csv --out outputs/predictions/embeddings/test.csv --split test

eval-embeddings:
	${VENV} && python -m src.cli.evaluate --config ${CONFIG_EVAL} \
	  --pred outputs/predictions/embeddings/test.csv --set_metrics
	@if [ -f eval/tables/metrics.csv ]; then \
	  mv eval/tables/metrics.csv eval/tables/metrics_embeddings.csv; \
	  echo "✓ Saved metrics to eval/tables/metrics_embeddings.csv"; \
	fi

all-embeddings: index-embeddings recommend-embeddings eval-embeddings
	@echo "✓ Embeddings pipeline complete"

# Compare both models
compare: all-tfidf all-embeddings
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
	@echo "Files saved:"
	@echo "  - eval/tables/metrics_tfidf.csv"
	@echo "  - eval/tables/metrics_embeddings.csv"
	@echo ""

clean:
	rm -rf data/interim data/processed models outputs eval/plots
	mkdir -p outputs/oscal outputs/predictions/tfidf outputs/predictions/embeddings eval/tables eval/plots
