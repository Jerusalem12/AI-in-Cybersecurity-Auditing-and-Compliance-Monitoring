.PHONY: setup format test index recommend eval oscal clean

VENV=. .venv/bin/activate

setup:
	python -m venv .venv && ${VENV} && pip install -U pip && pip install -e .
	@echo " Environment ready. Put CSVs into data/raw and run: make index recommend eval"

format:
	pre-commit run --all-files || true

test:
	${VENV} && pytest -q

index:
	${VENV} && python -m src.cli.build_index --config configs/defaults.yaml

recommend:
	${VENV} && python -m src.cli.recommend --config configs/defaults.yaml \
	  --in data/raw/artifacts.csv --out outputs/predictions/test.csv

eval:
	${VENV} && python -m src.cli.evaluate --config configs/eval.yaml

oscal:
	${VENV} && python -m src.crs.oscal --pred outputs/predictions/test.csv \
	  --out outputs/oscal/assessment_results.json

clean:
	rm -rf data/interim data/processed models outputs eval/plots
	mkdir -p outputs/oscal outputs/predictions eval/tables eval/plots
