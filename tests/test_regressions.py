"""Regression tests for defects found in review, plus previously untested branches."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sentence_transformers

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import AppConfig, EvidenceConfig, load_config
from finrag.data.schemas import AnswerPlan, RetrievalHit
from finrag.evaluation.answer_metrics import exact_match, numerical_match, score_answer
from finrag.evaluation.bootstrap import bootstrap_mean_ci
from finrag.evaluation.latency import latency_summary
from finrag.evaluation.retrieval_metrics import mean_metrics
from finrag.generation.planning import validate_answer_plan
from finrag.generation.providers import ExtractiveProvider, GenerationProvider
from finrag.logging import JsonFormatter, configure_logging
from finrag.retrieval.dense import DenseRetriever, embedding_cache_key
from finrag.tools.calculator import UnsafeExpressionError, safe_calculate

NESTED_POWER = "((((((((((10**12)**12)**12)**12)**12)**12)**12)**12)**12)**12)**12"


# --- calculator -------------------------------------------------------------------


def test_decimal_overflow_is_reported_as_unsafe_expression() -> None:
    """Previously escaped as decimal.Overflow and crashed the graph's calculate node."""
    with pytest.raises(UnsafeExpressionError, match="out of range"):
        safe_calculate(NESTED_POWER)


def test_plan_validation_survives_overflowing_expression(hits: list[RetrievalHit]) -> None:
    plan = AnswerPlan(
        decision="answer",
        answer_type="calculation",
        selected_citation_ids=[hits[0].chunk.citation_id],
        calculator_expression=NESTED_POWER,
        reason_code="test",
    )
    validation, _ = validate_answer_plan(plan, hits, ("10", "12"), 3)
    assert validation.valid is False
    assert "unsafe_or_invalid_expression" in validation.reasons


@pytest.mark.parametrize("expression", ["1 / 0", "(", "1 +", "abs(-1)", "x + 1", ""])
def test_invalid_expressions_never_raise_foreign_exceptions(expression: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_calculate(expression)


# --- configuration ----------------------------------------------------------------


def test_absolute_data_path_with_relative_manifest(tmp_path: Path) -> None:
    """Previously raised UnboundLocalError because project_root was conditionally bound."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.yaml").write_text(
        f"data:\n  path: {tmp_path / 'dev.json'}\n  question_manifest: data/held_out/m.json\n"
    )
    config = load_config(tmp_path / "configs" / "c.yaml")
    assert config.data.path == tmp_path / "dev.json"
    assert config.data.question_manifest == tmp_path / "data" / "held_out" / "m.json"


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text("evaluation:\n  checkpoint_every: 5\n")
    with pytest.raises(Exception, match="checkpoint_every"):
        load_config(tmp_path / "c.yaml")


# --- embedding cache ----------------------------------------------------------------


class StubSentenceTransformer:
    corpus_encodes = 0

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        if len(texts) > 1:
            type(self).corpus_encodes += 1
        vectors = np.zeros((len(texts), 8), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, hash(text) % 8] = 1.0
        return vectors


def test_embedding_cache_skips_reencoding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunks) -> None:
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", StubSentenceTransformer)
    StubSentenceTransformer.corpus_encodes = 0
    first = DenseRetriever(chunks, "stub-model", cache_dir=tmp_path)
    assert first.embedding_cache_hit is False
    second = DenseRetriever(chunks, "stub-model", cache_dir=tmp_path)
    assert second.embedding_cache_hit is True
    assert StubSentenceTransformer.corpus_encodes == 1
    assert np.array_equal(first.embeddings, second.embeddings)
    assert second.search("revenue", top_k=1)
    # A different model or corpus must not reuse the file.
    assert embedding_cache_key("stub-model", ["a"]) != embedding_cache_key("other", ["a"])
    assert embedding_cache_key("m", ["a", "b"]) != embedding_cache_key("m", ["ab"])
    third = DenseRetriever(chunks[:2], "stub-model", cache_dir=tmp_path)
    assert third.embedding_cache_hit is False


def test_corrupt_embedding_cache_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunks) -> None:
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", StubSentenceTransformer)
    key = embedding_cache_key("stub-model", [chunk.content for chunk in chunks])
    (tmp_path / f"dense_{key}.npy").write_bytes(b"not a numpy file")
    retriever = DenseRetriever(chunks, "stub-model", cache_dir=tmp_path)
    assert retriever.embedding_cache_hit is False
    np.save(tmp_path / f"dense_{key}.npy", np.zeros((1, 8), dtype=np.float32))
    assert DenseRetriever(chunks, "stub-model", cache_dir=tmp_path).embedding_cache_hit is False


# --- workflow abstention paths --------------------------------------------------------


class StubIndex:
    backends = {"dense": "test", "reranker": "test"}

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.last_timing_ms = {"retrieval": 0.1}

    def search(self, *args: Any, **kwargs: Any) -> list[RetrievalHit]:
        return self.hits


class ScriptedProvider(GenerationProvider):
    name = "scripted"

    def __init__(self, plan: AnswerPlan, answer: str) -> None:
        self._plan = plan
        self._answer = answer

    def plan(self, question: str, hits: list[RetrievalHit]) -> AnswerPlan:
        return self._plan

    def generate(self, question: str, hits: list[RetrievalHit], plan: AnswerPlan, calculator_result: str | None = None) -> str:
        return self._answer


def _config() -> AppConfig:
    return AppConfig(evidence=EvidenceConfig(min_token_overlap=0.1))


def _abstain_reason(trace: list[dict[str, Any]]) -> str:
    return next(event["reason"] for event in trace if event["node"] == "abstain")


def test_provider_abstention_reason_is_propagated(hits: list[RetrievalHit]) -> None:
    plan = AnswerPlan(decision="abstain", answer_type="none", reason_code="operand_missing")
    result = FinRAGWorkflow(StubIndex(hits), ScriptedProvider(plan, "unused"), _config()).answer("What was the change in revenue?")  # type: ignore[arg-type]
    assert result.abstained is True
    assert _abstain_reason(result.trace) == "operand_missing"


def test_calculator_rejection_routes_to_abstain(hits: list[RetrievalHit], monkeypatch: pytest.MonkeyPatch) -> None:
    plan = AnswerPlan(
        decision="answer",
        answer_type="calculation",
        selected_citation_ids=[hits[0].chunk.citation_id],
        calculator_expression="120 / 100",
        reason_code="test",
    )
    # Validation accepts the expression; make execution fail afterwards to exercise the edge.
    import finrag.agents.graph as graph_module

    def failing(expression: str) -> str:
        raise UnsafeExpressionError("forced")

    monkeypatch.setattr(graph_module, "safe_calculate", failing)
    result = FinRAGWorkflow(StubIndex(hits), ScriptedProvider(plan, "unused"), _config()).answer("What was the change in revenue?")  # type: ignore[arg-type]
    assert result.abstained is True
    assert _abstain_reason(result.trace) == "calculator_rejected_expression"
    assert "generate_answer" not in [event["node"] for event in result.trace]


def test_invented_citation_in_generation_is_overridden(hits: list[RetrievalHit]) -> None:
    plan = AnswerPlan(decision="answer", answer_type="extractive", selected_citation_ids=[hits[0].chunk.citation_id], reason_code="test")
    provider = ScriptedProvider(plan, "Revenue was 120. [CITATION: INVENTED/2020/page_1.pdf/text_9]")
    result = FinRAGWorkflow(StubIndex(hits), provider, _config()).answer("What was the change in revenue?")  # type: ignore[arg-type]
    assert result.abstained is True
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.citation_verification.invalid_ids == ["INVENTED/2020/page_1.pdf/text_9"]


def test_extractive_provider_generation_branches(hits: list[RetrievalHit], chunks) -> None:
    provider = ExtractiveProvider()
    plan = AnswerPlan(decision="answer", answer_type="extractive", reason_code="test")
    assert provider.generate("q", [], plan) == "INSUFFICIENT_EVIDENCE"
    assert "the calculated result is 20%" in provider.generate("What percentage changed?", hits, plan, "20")
    assert "retrieved value is 100" in provider.generate("q", hits, plan)
    text_only = RetrievalHit(chunk=chunks[2], score=1.0, rank=1, method="test")
    assert provider.generate("q", [text_only], plan).startswith("Other company discussed emissions targets.")


# --- metrics and utilities ------------------------------------------------------------


def test_text_answers_fall_back_to_normalized_exact_match() -> None:
    assert exact_match("The Company Reported A Loss", "company reported loss") is True
    assert numerical_match("no number here", "12", 0.02) is None
    assert score_answer("none", "12", 0.02) == {"exact_match": 0.0, "numerical_accuracy": 0.0}
    assert numerical_match("0", "0", 0.02) is True


def test_bootstrap_and_latency_edge_cases() -> None:
    assert bootstrap_mean_ci([]) == {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    point = bootstrap_mean_ci([1.0, 3.0], samples=0)
    assert point["ci_low"] == point["ci_high"] == 2.0
    summary = latency_summary([{"total": 10.0, "retrieval": 4.0}, {"total": 30.0}])
    assert summary["total"]["median_ms"] == 20.0
    assert summary["retrieval"]["mean_ms"] == 4.0
    assert mean_metrics([]) == {}


def test_json_logging_emits_parseable_records(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(logging.INFO)
    logging.getLogger("finrag.test").info("hello %s", "world")
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["message"] == "hello world"
    assert record["level"] == "INFO"
    assert isinstance(JsonFormatter().format(logging.makeLogRecord({"msg": "x"})), str)
