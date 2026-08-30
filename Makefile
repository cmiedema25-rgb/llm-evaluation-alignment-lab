PYTHON ?= python

.PHONY: install lint test benchmark verify

install:
	$(PYTHON) -m pip install -e '.[dev]'

lint:
	ruff check .
	ruff format --check .

test:
	pytest --cov=llm_eval_lab --cov-report=term-missing --cov-fail-under=85 -q

benchmark:
	llm-eval-lab benchmark data/evaluation_cases.jsonl \
		--preferences data/sample_preferences.jsonl \
		--report evidence/benchmark-report.json

verify: lint test benchmark
