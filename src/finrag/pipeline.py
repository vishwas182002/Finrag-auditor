"""Shared assembly of corpus, index, provider, and workflow from one configuration.

Every entry point (CLI, evaluation runners, Streamlit UI) previously repeated the
same load -> select -> corpus -> chunk -> index sequence. Centralizing it keeps the
question cohort, corpus scope, and cache location consistent across all of them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import AppConfig
from finrag.data.finqa import load_finqa, select_configured, selected_corpus_examples
from finrag.data.schemas import DocumentChunk, FinQAExample
from finrag.generation.providers import GenerationProvider, create_provider
from finrag.indexing.chunking import build_chunks
from finrag.indexing.index import RetrievalIndex

PROJECT_ROOT_ENV = "FINRAG_PROJECT_ROOT"
_ROOT_MARKERS = ("pyproject.toml", "configs")


def resolve_project_root(config_path: str | Path | None = None) -> Path:
    """Locate the repository root that owns ``configs/``, ``data/`` and ``artifacts/``.

    Resolution order: ``FINRAG_PROJECT_ROOT``; the directory implied by the config
    file (parent of its ``configs/`` folder); the current working directory when it
    contains project markers; finally the source checkout that contains this package.
    The last option only works for editable installs, so containers and wheel installs
    should set the environment variable or run from the project directory.
    """
    override = os.getenv(PROJECT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if config_path is not None:
        resolved = Path(config_path).resolve()
        if resolved.parent.name == "configs":
            return resolved.parent.parent
    cwd = Path.cwd().resolve()
    if all((cwd / marker).exists() for marker in _ROOT_MARKERS):
        return cwd
    return Path(__file__).resolve().parents[2]


def embedding_cache_dir(project_root: Path) -> Path:
    return project_root / "artifacts" / "indexes"


@dataclass(frozen=True)
class Corpus:
    examples: list[FinQAExample]
    selected: list[FinQAExample]
    corpus_examples: list[FinQAExample]
    chunks: list[DocumentChunk]

    @property
    def report_ids(self) -> list[str]:
        return sorted({chunk.report_id for chunk in self.chunks})


def build_corpus(config: AppConfig) -> Corpus:
    examples = load_finqa(config.data.path)
    selected = select_configured(
        examples,
        config.data.sample_size,
        config.seed,
        config.data.question_manifest,
    )
    corpus_examples = selected_corpus_examples(examples, selected, config.data.corpus_scope)
    chunks = build_chunks(corpus_examples, config.chunking)
    return Corpus(
        examples=examples, selected=selected, corpus_examples=corpus_examples, chunks=chunks
    )


def build_index(
    chunks: list[DocumentChunk],
    config: AppConfig,
    project_root: Path,
    allow_model_fallback: bool = True,
) -> RetrievalIndex:
    return RetrievalIndex(
        chunks,
        config.retrieval,
        allow_model_fallback=allow_model_fallback,
        cache_dir=embedding_cache_dir(project_root),
    )


def build_provider(config: AppConfig) -> GenerationProvider:
    return create_provider(
        config.generation.provider,
        config.generation.model,
        config.generation.temperature,
        config.generation.base_url,
        config.generation.api_key_env,
        config.generation.max_retries,
        config.generation.request_interval_seconds,
    )


def build_workflow(
    config: AppConfig, project_root: Path, allow_model_fallback: bool = True
) -> tuple[FinRAGWorkflow, Corpus]:
    corpus = build_corpus(config)
    index = build_index(corpus.chunks, config, project_root, allow_model_fallback)
    workflow = FinRAGWorkflow(index, build_provider(config), config)
    return workflow, corpus
