"""Offline end-to-end coverage of every runner, the CLI, and the pipeline helpers.

The neural backends are forced onto their named fallbacks so the tests run in CI
without model downloads. They check artifact contracts, resume behaviour, and
routing—not retrieval quality.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import sentence_transformers

from finrag.cli import main
from finrag.config import AppConfig, EvaluationConfig, EvidenceConfig, load_config
from finrag.evaluation.abstention_calibration import run_abstention_calibration
from finrag.evaluation.planner_audit import run_planner_audit
from finrag.evaluation.retrieval_ablation import run_retrieval_ablation
from finrag.evaluation.runner import run_evaluation
from finrag.pipeline import build_corpus, build_workflow, resolve_project_root


def _record(report: str, index: int, question: str, answer: str, program: str, gold: dict[str, str]) -> dict[str, Any]:
    return {
        "id": f"{report}-{index}",
        "filename": report,
        "pre_text": [
            f"{report.split('/')[0]} reported annual results for the fiscal year.",
            "Operating costs of 80 were recorded, while interest expense was 12.",
        ],
        "post_text": ["Revenue increased during the year as volumes recovered."],
        "table": [["Metric", "2023", "2022"], ["Revenue", "120", "100"], ["Net income", "30", "25"]],
        "qa": {"question": question, "answer": answer, "program": program, "gold_inds": gold},
    }


SYNTHETIC_RECORDS = [
    _record("ACME/2023/page_1.pdf", 1, "What was the change in revenue?", "20", "subtract(120, 100)", {"table_1": "row"}),
    _record("ACME/2023/page_1.pdf", 2, "What was the total of revenue and net income in 2023?", "150", "add(120, 30)", {"table_1": "row", "table_2": "row"}),
    _record("BETA/2022/page_4.pdf", 1, "What was the percentage change in net income?", "20%", "subtract(30, 25), divide(#0, 25)", {"table_2": "row"}),
    _record("GAMMA/2021/page_9.pdf", 1, "What were the operating costs recorded?", "80", "", {"text_1": "text"}),
    _record("DELTA/2020/page_2.pdf", 1, "What was the average of revenue in 2023 and 2022?", "110", "add(120, 100), divide(#0, const_2)", {"table_1": "row"}),
]


@pytest.fixture
def offline_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("offline test environment")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", unavailable)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", unavailable)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project root with a synthetic FinQA split and a config file."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "raw" / "dev.json").write_text(json.dumps(SYNTHETIC_RECORDS))
    (tmp_path / "configs" / "synthetic.yaml").write_text(
        "seed: 1\n"
        "data:\n  path: data/raw/dev.json\n  sample_size: 4\n  unanswerable_size: 2\n"
        "  corpus_scope: full_split\n"
        "evidence:\n  min_token_overlap: 0.05\n  min_reranker_score: 0.0\n"
        "evaluation:\n  bootstrap_samples: 20\n  run_name: synthetic\n"
        "  min_answerable_acceptance: 0.0\n"
    )
    monkeypatch.setenv("FINRAG_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _config(project: Path) -> AppConfig:
    return load_config(project / "configs" / "synthetic.yaml")


def test_resolve_project_root_prefers_env_then_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINRAG_PROJECT_ROOT", str(tmp_path))
    assert resolve_project_root("anything.yaml") == tmp_path.resolve()
    monkeypatch.delenv("FINRAG_PROJECT_ROOT")
    (tmp_path / "configs").mkdir()
    assert resolve_project_root(tmp_path / "configs" / "x.yaml") == tmp_path.resolve()
    # A config outside a configs/ folder falls back to the working directory when it
    # looks like a project, and otherwise to the source checkout.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("")
    assert resolve_project_root(tmp_path / "loose.yaml") == tmp_path.resolve()
    assert (resolve_project_root() / "configs").exists()


def test_build_corpus_scopes_chunks(project: Path) -> None:
    corpus = build_corpus(_config(project))
    assert len(corpus.selected) == 4
    assert len(corpus.report_ids) == 4
    assert any(chunk.chunk_id == "table_0" for chunk in corpus.chunks)


def test_run_evaluation_writes_contract_and_resumes(project: Path, offline_backends: None) -> None:
    config = _config(project)
    report = run_evaluation(config, project)
    results = project / "artifacts" / "results" / "synthetic"
    for name in (
        "retrieval_metrics.json",
        "answer_metrics.json",
        "predictions.jsonl",
        "retrieval_rows.jsonl",
        "latency.csv",
        "ablation.csv",
        "evaluation_metadata.json",
    ):
        assert (results / name).exists(), name
    assert report["metadata"]["retrieval_backends"]["dense"].startswith("fallback:")
    assert report["answer"]["answerable_questions"] == 4
    assert report["answer"]["constructed_unanswerable_questions"] == 2
    assert set(report["retrieval"]) == {"bm25", "dense", "hybrid", "hybrid_rerank"}
    predictions = [json.loads(line) for line in (results / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 6
    assert all(row["provider"] == "deterministic-extractive-v1" for row in predictions)
    with (results / "latency.csv").open() as handle:
        header = next(csv.reader(handle))
    assert "planning" in header
    retrieval_rows = (results / "retrieval_rows.jsonl").read_text().splitlines()
    assert len(retrieval_rows) == 4 * 4  # four methods x four questions
    assert len(list((results / "ablation.csv").open())) == 1 + 4 + 1

    checkpoints = list((project / "artifacts" / "checkpoints").glob("workflow_synthetic_*.jsonl"))
    assert len(checkpoints) == 1
    before = checkpoints[0].read_text()
    resumed = run_evaluation(config, project)
    assert checkpoints[0].read_text() == before, "resume must not re-append completed rows"
    assert resumed["answer"]["numerical_accuracy"] == report["answer"]["numerical_accuracy"]


def test_changed_configuration_creates_a_separate_checkpoint(project: Path, offline_backends: None) -> None:
    config = _config(project)
    run_evaluation(config, project)
    altered = config.model_copy(
        update={"evaluation": EvaluationConfig(run_name="synthetic", bootstrap_samples=20, min_answerable_acceptance=0.0), "seed": 2},
        deep=True,
    )
    run_evaluation(altered, project)
    checkpoints = list((project / "artifacts" / "checkpoints").glob("workflow_synthetic_*.jsonl"))
    assert len(checkpoints) == 2


def test_retrieval_ablation_and_planner_audit(project: Path, offline_backends: None) -> None:
    config = _config(project)
    ablation = run_retrieval_ablation(config, project)
    assert "latency_note" in ablation["methods"]["hybrid_rerank"]
    rows = (project / "artifacts" / "retrieval_ablation" / "synthetic" / "retrieval_rows.jsonl").read_text().splitlines()
    assert len(rows) == 16
    audit = run_planner_audit(config, project)
    assert audit["metadata"]["generation_model_called"] is False
    assert audit["oracle_gold_evidence"]["gold_evidence_available"] == 1.0
    assert (project / "artifacts" / "planner_audit" / "synthetic" / "planner_audit_rows.jsonl").exists()


def test_abstention_calibration_reports_sweep(project: Path, offline_backends: None) -> None:
    config = _config(project)
    report = run_abstention_calibration(config, project)
    assert report["metadata"]["constructed_unanswerable_questions"] == 4
    assert 0.0 <= report["selected"]["balanced_gate_accuracy"] <= 1.0
    saved = json.loads((project / "artifacts" / "abstention_calibration" / "synthetic" / "abstention_calibration.json").read_text())
    assert len(saved["threshold_sweep"]) >= 2


def test_cli_ask_and_evaluate(project: Path, offline_backends: None, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = str(project / "configs" / "synthetic.yaml")
    main(["ask", "What was the change in revenue?", "--config", config_path, "--report-id", "ACME/2023/page_1.pdf"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["provider"] == "deterministic-extractive-v1"
    assert all(hit["chunk"]["report_id"] == "ACME/2023/page_1.pdf" for hit in payload["retrieved"])
    main(["evaluate", "--config", config_path])
    summary = json.loads(capsys.readouterr().out)
    assert summary["metadata"]["run_name"] == "synthetic"
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "finrag" in capsys.readouterr().out


def test_workflow_warns_when_gate_threshold_meets_fallback_reranker(project: Path, offline_backends: None, caplog: pytest.LogCaptureFixture) -> None:
    config = _config(project).model_copy(update={"evidence": EvidenceConfig(min_reranker_score=0.6461)}, deep=True)
    with caplog.at_level("WARNING"):
        workflow, _ = build_workflow(config, project)
    assert workflow.index.uses_fallback
    assert any("calibrated for a cross-encoder" in message for message in caplog.messages)
