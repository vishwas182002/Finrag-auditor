# Contributing

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional: ruff + hygiene hooks on commit
python scripts/download_finqa.py   # only needed for real evaluation runs
```

## Checks that must pass

```bash
ruff check .
mypy src/finrag
pytest --cov=finrag         # fails under 85% coverage
```

The test suite is fully offline: neural backends are monkeypatched onto their named
fallbacks, so no model download or API key is required. Set `HF_HUB_OFFLINE=1` to be
certain nothing reaches the Hub.

## Ground rules

- **Never evaluate on `data/held_out/test_v2_manifest.json` while selecting models,
  thresholds, prompts, or providers.** It is the sealed final cohort.
- Gold `gold_inds` and `program` fields may be used for scoring only. Do not pass them
  to retrieval, reranking, planning, or generation.
- Credentials are read from the environment variable named by
  `generation.api_key_env`. Never put keys in YAML, traces, or artifacts.
- Every new config key must be consumed somewhere; `AppConfig` forbids unknown keys and
  the test suite validates every `configs/*.yaml`.
- Committed artifacts are provenance. Do not regenerate them silently; add a new
  `run_name` directory and describe the change in `CHANGELOG.md` and the README.
- New metrics need a unit test with a hand-computed expected value.

## Reporting numbers

State the split, question count, corpus scope, backend names (from
`evaluation_metadata.json`), and whether a fallback backend was active. Numbers produced
on `fallback:*` backends are not comparable with the committed results.
