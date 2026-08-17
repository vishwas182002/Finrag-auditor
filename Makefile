PYTHON ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: install install-llm data test lint typecheck evaluate evaluate-full audit-planner evaluate-bge calibrate-abstention app verify

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

install-llm:
	$(BIN)/pip install -e ".[dev,openai]"

data:
	$(BIN)/python scripts/download_finqa.py

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .

typecheck:
	$(BIN)/mypy src/finrag

evaluate:
	$(BIN)/finrag evaluate --config configs/quick_eval.yaml

evaluate-full:
	$(BIN)/finrag evaluate --config configs/full_eval.yaml

audit-planner:
	$(BIN)/finrag audit-planner --config configs/planner_audit.yaml

evaluate-bge:
	$(BIN)/finrag evaluate-retrieval --config configs/reranker_bge_dev.yaml

calibrate-abstention:
	$(BIN)/finrag calibrate-abstention --config configs/abstention_bge_dev.yaml

app:
	$(BIN)/streamlit run app.py

verify: test lint evaluate
