from __future__ import annotations

import json 
from pathlib import Path

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


@pytest.mark.parametrize(
    "config_path",
    sorted(Path("configs").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_repository_configs_parse(config_path: Path) -> None:
    load_config(config_path)



def test_configuration_extends_recursively(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text(
        "retrieval:\n  reranker_model: BAAI/bge-reranker-base\n"
    )
    (tmp_path / "provider.yaml").write_text(
        "extends: base.yaml\ngeneration:\n  provider: openai_compatible\n"
    )
    (tmp_path / "smoke.yaml").write_text(
        "extends: provider.yaml\ndata:\n  sample_size: 10\n"
    )
    config = load_config(tmp_path / "smoke.yaml")
    assert config.retrieval.reranker_model == "BAAI/bge-reranker-base"
    assert config.generation.provider == "openai_compatible"
    assert config.data.sample_size == 10


def test_configuration_rejects_circular_extends(tmp_path) -> None:
    (tmp_path / "first.yaml").write_text("extends: second.yaml\n")
    (tmp_path / "second.yaml").write_text("extends: first.yaml\n")
    with pytest.raises(ValueError, match="Circular config inheritance"):
        load_config(tmp_path / "first.yaml")


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
