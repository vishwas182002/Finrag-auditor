from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from finrag.config import AppConfig, ChunkingConfig, RetrievalConfig, load_config
from finrag.evaluation.checkpoint import JsonlCheckpoint
from finrag.generation.providers import create_provider


def test_configuration_validation() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(text_chunk_chars=100, text_overlap_chars=100)
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=10, candidate_k=5)
    assert AppConfig().seed == 42


def test_configuration_extends(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text("seed: 7\ndata:\n  sample_size: 10\n")
    (tmp_path / "quick.yaml").write_text("extends: base.yaml\ndata:\n  sample_size: 2\n")
    config = load_config(tmp_path / "quick.yaml")
    assert config.seed == 7
    assert config.data.sample_size == 2
    assert config.data.path == tmp_path / "data/raw/dev.json"


def test_checkpoint_resume(tmp_path) -> None:
    checkpoint = JsonlCheckpoint(tmp_path / "checkpoint.jsonl")
    checkpoint.append({"evaluation_id": "q1", "prediction": "1"})
    checkpoint.append({"evaluation_id": "q2", "prediction": "2"})
    assert checkpoint.completed_ids() == {"q1", "q2"}
    assert json.loads((tmp_path / "checkpoint.jsonl").read_text().splitlines()[1])["prediction"] == "2"


def test_openai_compatible_provider_requires_named_environment_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        create_provider(
            "openai_compatible",
            "test-model",
            base_url="https://example.invalid/openai/v1",
            api_key_env="GROQ_API_KEY",
        )
