"""Model-backed provider plumbing (stubbed client) and data-selection edge cases."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from finrag.config import ChunkingConfig
from finrag.data.finqa import load_finqa, select_from_manifest, select_held_out
from finrag.data.schemas import AnswerPlan, FinQAExample, RetrievalHit
from finrag.generation.prompts import render_evidence, render_planning_request
from finrag.generation.providers import OpenAICompatibleProvider, create_provider
from finrag.indexing.chunking import build_chunks


class StubStructured:
    def __init__(self, plan: AnswerPlan | dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return self.plan


class StubChatOpenAI:
    instances: list[StubChatOpenAI] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.structured = StubStructured(
            AnswerPlan(decision="answer", answer_type="extractive", selected_citation_ids=["a/b"], reason_code="stub")
        )
        self.generate_calls: list[Any] = []
        type(self).instances.append(self)

    def with_structured_output(self, schema: Any) -> StubStructured:
        return self.structured

    def invoke(self, messages: Any) -> Any:
        self.generate_calls.append(messages)
        return types.SimpleNamespace(content="Revenue was 120. [CITATION: a/b]")


@pytest.fixture
def stub_langchain(monkeypatch: pytest.MonkeyPatch) -> type[StubChatOpenAI]:
    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = StubChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    StubChatOpenAI.instances.clear()
    return StubChatOpenAI


def test_openai_compatible_provider_plans_and_generates(stub_langchain: type[StubChatOpenAI], hits: list[RetrievalHit], monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("finrag.generation.providers.time.sleep", lambda seconds: sleeps.append(seconds))
    provider = create_provider(
        "openai_compatible",
        "stub-model",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_PROVIDER_KEY",
        max_retries=1,
        request_interval_seconds=5.0,
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "openai-compatible:stub-model"
    client = stub_langchain.instances[-1]
    assert client.kwargs["base_url"] == "https://example.invalid/v1"
    assert client.kwargs["api_key"] == "secret-value"
    plan = provider.plan("What was the change in revenue?", hits)
    assert plan.selected_citation_ids == ["a/b"]
    assert "Evidence:" in client.structured.calls[0][1][1]
    answer = provider.generate("q", hits, plan, "20")
    assert answer.endswith("[CITATION: a/b]")
    assert "Calculator result: 20" in client.generate_calls[0][1][1]
    # The second request is paced relative to the first one finishing.
    assert sleeps and 0 < sleeps[0] <= 5.0
    assert create_provider("openai", "m", api_key_env="TEST_PROVIDER_KEY").name == "openai:m"


def test_openai_compatible_provider_validates_dict_plans(stub_langchain: type[StubChatOpenAI], hits: list[RetrievalHit]) -> None:
    provider = OpenAICompatibleProvider("stub-model", api_key_env="TEST_PROVIDER_KEY")
    provider.planner = StubStructured({"decision": "abstain", "answer_type": "none", "reason_code": "no_operands"})  # type: ignore[assignment]
    assert provider.plan("q", hits).decision == "abstain"


def test_missing_openai_extra_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "x")
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(RuntimeError, match="openai"):
        OpenAICompatibleProvider("m", api_key_env="TEST_PROVIDER_KEY")
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("mystery", "m")


def test_prompt_rendering_lists_every_citation_id(hits: list[RetrievalHit]) -> None:
    rendered = render_evidence(hits)
    assert f"[{hits[0].chunk.citation_id}]" in rendered
    request = render_planning_request("q", hits, "{schema}")
    assert request.startswith("Question: q") and request.endswith("{schema}")


# --- data selection and chunking -------------------------------------------------------


def test_long_paragraphs_are_split_with_overlap(example: FinQAExample) -> None:
    long_text = " ".join(f"sentence{i}" for i in range(120))
    padded = example.model_copy(update={"pre_text": (long_text,)})
    config = ChunkingConfig(text_chunk_chars=300, text_overlap_chars=50)
    parts = [chunk for chunk in build_chunks([padded], config) if chunk.source_ids == ("text_0",)]
    assert len(parts) > 1
    assert [chunk.chunk_id for chunk in parts][:2] == ["text_0_part_0", "text_0_part_1"]
    assert parts[0].content[-50:] == parts[1].content[:50]


def test_held_out_sampling_fills_from_remaining_questions(example: FinQAExample) -> None:
    second = example.model_copy(update={"question_id": "ACME/2023/page_1.pdf-2"})
    third = example.model_copy(update={"question_id": "other-1", "report_id": "other"})
    selected = select_held_out([example, second, third], 3, 42)
    assert len(selected) == 3
    assert len({item.report_id for item in selected}) == 2


def test_manifest_validation_failures(tmp_path: Path, example: FinQAExample) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"question_ids": []}))
    with pytest.raises(ValueError, match="no question_ids"):
        select_from_manifest([example], manifest)
    manifest.write_text(json.dumps({"question_ids": ["a", "a"]}))
    with pytest.raises(ValueError, match="duplicate"):
        select_from_manifest([example], manifest)
    manifest.write_text(json.dumps({"question_ids": [example.question_id], "question_ids_sha256": "bad"}))
    with pytest.raises(ValueError, match="sha256"):
        select_from_manifest([example], manifest)
    manifest.write_text(json.dumps({"question_ids": ["missing-1"]}))
    with pytest.raises(ValueError, match="absent from split"):
        select_from_manifest([example], manifest)
    digest = hashlib.sha256((example.question_id + "\n").encode()).hexdigest()
    manifest.write_text(json.dumps({"question_ids": [example.question_id], "question_ids_sha256": digest, "sample_size": 2}))
    with pytest.raises(ValueError, match="sample_size"):
        select_from_manifest([example], manifest)


def test_load_finqa_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_finqa"):
        load_finqa(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError, match="JSON array"):
        load_finqa(tmp_path / "bad.json")
