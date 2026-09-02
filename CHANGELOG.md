# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- `safe_calculate` now converts `decimal.Overflow`/`Underflow` (e.g. deeply nested
  exponents) into `UnsafeExpressionError`, and bounds the Decimal exponent range. A
  provider-generated expression could previously crash the `calculate` graph node
  and abort an evaluation run instead of routing to abstention. `validate_answer_plan`
  applies the same guard.
- `load_config` raised `UnboundLocalError` when `data.path` was absolute but
  `data.question_manifest` was relative.
- The CLI located the project root relative to the installed package, which is wrong
  for non-editable installs (the Docker image wrote artifacts into `site-packages`).
  Root resolution now honours `FINRAG_PROJECT_ROOT`, then the config file's
  `configs/` parent, then the working directory, then the source checkout.
- The Streamlit app ignored `generation.request_interval_seconds`.
- `finrag evaluate-retrieval` defaulted to the TinyBERT planner-audit config instead of
  the documented BGE development config.
- `streamlit` lower bound raised to 1.49 (`st.dataframe(width="stretch")` did not exist
  before that release).

### Added
- On-disk corpus embedding cache under `artifacts/indexes/`, keyed by embedding model
  and chunk contents, controlled by `retrieval.use_model_cache` (previously an unused key).
- `finrag.pipeline` with `build_corpus` / `build_index` / `build_provider` /
  `build_workflow`, replacing five copies of the same assembly sequence.
- `python -m finrag`, `finrag --version`, and sub-command help strings.
- `artifacts/results/<run>/retrieval_rows.jsonl` per-question retrieval metrics, which
  were computed but never written; `latency.csv` now includes the `planning` stage.
- A warning when the evidence gate's cross-encoder threshold is applied to the
  token-overlap fallback reranker, and when any fallback backend is in use.
- End-to-end offline tests for the evaluation runner, retrieval ablation, planner audit,
  abstention calibration, CLI, checkpoint resume/fingerprint isolation, embedding cache,
  OpenAI-compatible provider plumbing (stubbed client), and every abstention route.
  Coverage rises from 63% to 93% with an enforced 85% floor.
- CI: Python 3.11/3.12 matrix, mypy, coverage gate, CLI smoke, frozen-manifest digest
  check, and a Docker build on `main`. Dependabot and pre-commit configurations.
- `.dockerignore`; the image now installs CPU-only torch, copies committed artifacts and
  the frozen manifest, and declares volumes for raw data, model cache, and artifacts.
- `CONTRIBUTING.md`, `CITATION.cff`, this changelog.

### Removed
- `evaluation.checkpoint_every`: it was never read; every evaluation case is already
  appended to the checkpoint as soon as it completes.

## [0.1.0] - 2026-08-17

Initial public version: hybrid BM25/BGE retrieval with RRF and cross-encoder
reranking, LangGraph workflow with plan validation and citation verification,
FinQA evaluation harness, planner audit, reranker ablation, abstention calibration,
Streamlit audit UI.
