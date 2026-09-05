"""Content identities for evaluation reuse, including dirty source trees."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from finrag.config import AppConfig
from finrag.data.schemas import DocumentChunk, FinQAExample
from finrag.evaluation.answer_metrics import ANSWER_METRIC_VERSION
from finrag.evaluation.retrieval_metrics import METRIC_VERSION


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_digest(source_root: Path | None = None) -> str:
    root = source_root or Path(__file__).resolve().parents[1]
    return digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
        }
    )


def evaluation_identity(
    config: AppConfig,
    selected: list[FinQAExample],
    chunks: list[DocumentChunk],
    provider: str,
    backends: dict[str, str],
) -> dict[str, Any]:
    configuration = config.model_dump(mode="json")
    # Paths and pacing do not affect predictions; their contents do.
    configuration["data"].pop("path")
    configuration["data"].pop("question_manifest")
    configuration["evaluation"]["resume"] = False
    configuration["generation"].pop("request_interval_seconds")
    dependencies = {}
    for name in (
        "langgraph",
        "langchain-core",
        "langchain-openai",
        "numpy",
        "pydantic",
        "rank-bm25",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    return {
        "config": configuration,
        "data_sha256": hashlib.sha256(config.data.path.read_bytes()).hexdigest(),
        "selected_sha256": digest([example.model_dump(mode="json") for example in selected]),
        "corpus_sha256": digest([chunk.model_dump(mode="json") for chunk in chunks]),
        "source_sha256": source_digest(),
        "dependencies": dependencies,
        "provider": provider,
        "backends": backends,
        "retrieval_metric_version": METRIC_VERSION,
        "answer_metric_version": ANSWER_METRIC_VERSION,
    }
