.PHONY: install lint format test smoke audit clean

install:
	pip install -e ".[dev,xgboost]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest --cov=oulad_risk --cov-report=term-missing

smoke:
	oulad-risk synth --out-dir data/synthetic --students 1500
	oulad-risk run --data-dir data/synthetic --out-dir reports/smoke --model rf --min-auc 0.7

audit:
	oulad-risk audit --data-dir data/raw

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml reports data/synthetic
