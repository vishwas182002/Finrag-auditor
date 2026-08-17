"""Typed configuration loading with explicit validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    split: Literal["dev", "test"] = "dev"
    path: Path = Path("data/raw/dev.json")
    sample_size: int = Field(50, ge=1)
    unanswerable_size: int = Field(10, ge=0)
    corpus_scope: Literal["selected_reports", "full_split"] = "selected_reports"
    question_manifest: Path | None = None


class ChunkingConfig(StrictModel):
    text_chunk_chars: int = Field(900, ge=100)
    text_overlap_chars: int = Field(120, ge=0)
    table_rows_per_chunk: int = Field(1, ge=1)

    @model_validator(mode="after")
    def overlap_smaller_than_chunk(self) -> ChunkingConfig:
        if self.text_overlap_chars >= self.text_chunk_chars:
            raise ValueError("text_overlap_chars must be smaller than text_chunk_chars")
        return self


class RetrievalConfig(StrictModel):
    method: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(5, ge=1)
    candidate_k: int = Field(15, ge=1)
    rrf_k: int = Field(60, ge=1)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    device: str = "cpu"
    use_model_cache: bool = True

    @model_validator(mode="after")
    def candidate_pool_is_large_enough(self) -> RetrievalConfig:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        return self


class EvidenceConfig(StrictModel):
    min_chunks: int = Field(1, ge=1)
    min_token_overlap: float = Field(0.14, ge=0.0, le=1.0)
    min_content_tokens: int = Field(4, ge=1)
    min_reranker_score: float = 0.0


class GenerationConfig(StrictModel):
    provider: Literal["extractive", "openai", "openai_compatible"] = "extractive"
    model: str = "deterministic-extractive-v1"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_retries: int = Field(0, ge=0, le=3)
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


class PlanningConfig(StrictModel):
    max_selected_chunks: int = Field(3, ge=1, le=10)
    allowed_constants: tuple[str, ...] = (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "100",
        "1000",
        "10000",
        "100000",
        "1000000",
        "10000000",
        "1000000000",
    )


class EvaluationConfig(StrictModel):
    numerical_tolerance: float = Field(0.02, ge=0.0, le=1.0)
    bootstrap_samples: int = Field(1000, ge=0)
    checkpoint_every: int = Field(5, ge=1)
    resume: bool = True
    run_name: str = "default"
    min_answerable_acceptance: float = Field(0.8, ge=0.0, le=1.0)


class AppConfig(StrictModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)  # type: ignore[arg-type]
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)  # type: ignore[arg-type]
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)  # type: ignore[arg-type]
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)  # type: ignore[arg-type]
    planning: PlanningConfig = Field(default_factory=PlanningConfig)  # type: ignore[arg-type]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)  # type: ignore[arg-type]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)  # type: ignore[arg-type]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text()) or {}
    extends = raw.pop("extends", None)
    if extends:
        base_path = (config_path.parent / extends).resolve()
        base_raw = yaml.safe_load(base_path.read_text()) or {}
        base_raw.pop("extends", None)
        raw = _deep_merge(base_raw, raw)
    config = AppConfig.model_validate(raw)
    if not config.data.path.is_absolute():
        project_root = (
            config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        )
        config.data.path = (project_root / config.data.path).resolve()
    if config.data.question_manifest is not None and not config.data.question_manifest.is_absolute():
        config.data.question_manifest = (project_root / config.data.question_manifest).resolve()
    return config
