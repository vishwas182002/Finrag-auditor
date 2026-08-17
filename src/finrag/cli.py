"""Command-line entrypoint for evaluation and one-off questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import load_config
from finrag.data.finqa import load_finqa, select_configured, selected_corpus_examples
from finrag.evaluation.abstention_calibration import run_abstention_calibration
from finrag.evaluation.planner_audit import run_planner_audit
from finrag.evaluation.retrieval_ablation import run_retrieval_ablation
from finrag.evaluation.retrieval_comparison import compare_retrieval_runs
from finrag.evaluation.runner import run_evaluation
from finrag.generation.providers import create_provider
from finrag.indexing.chunking import build_chunks
from finrag.indexing.index import RetrievalIndex
from finrag.logging import configure_logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evaluate(config_path: str) -> None:
    root = _project_root()
    config = load_config(config_path)
    result = run_evaluation(config, root)
    print(json.dumps(result, indent=2))


def audit_planner(config_path: str) -> None:
    root = _project_root()
    config = load_config(config_path)
    result = run_planner_audit(config, root)
    print(json.dumps(result, indent=2))


def evaluate_retrieval(config_path: str) -> None:
    root = _project_root()
    config = load_config(config_path)
    result = run_retrieval_ablation(config, root)
    print(json.dumps(result, indent=2))


def compare_retrieval(baseline: str, candidate: str, output: str) -> None:
    result = compare_retrieval_runs(Path(baseline), Path(candidate), Path(output))
    print(json.dumps(result, indent=2))


def calibrate_abstention(config_path: str) -> None:
    root = _project_root()
    config = load_config(config_path)
    result = run_abstention_calibration(config, root)
    print(json.dumps(result, indent=2))


def ask(config_path: str, question: str, report_id: str | None) -> None:
    config = load_config(config_path)
    examples = load_finqa(config.data.path)
    selected = select_configured(
        examples,
        config.data.sample_size,
        config.seed,
        config.data.question_manifest,
    )
    corpus = selected_corpus_examples(examples, selected, config.data.corpus_scope)
    chunks = build_chunks(corpus, config.chunking)
    index = RetrievalIndex(chunks, config.retrieval)
    provider = create_provider(
        config.generation.provider,
        config.generation.model,
        config.generation.temperature,
        config.generation.base_url,
        config.generation.api_key_env,
        config.generation.max_retries,
        config.generation.request_interval_seconds,
    )
    result = FinRAGWorkflow(index, provider, config).answer(
        question, allowed_report_ids=[report_id] if report_id else None
    )
    print(result.model_dump_json(indent=2))


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="finrag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", default="configs/quick_eval.yaml")
    planner_parser = subparsers.add_parser("audit-planner")
    planner_parser.add_argument("--config", default="configs/planner_audit.yaml")
    retrieval_parser = subparsers.add_parser("evaluate-retrieval")
    retrieval_parser.add_argument("--config", default="configs/planner_audit.yaml")
    comparison_parser = subparsers.add_parser("compare-retrieval")
    comparison_parser.add_argument("--baseline", required=True)
    comparison_parser.add_argument("--candidate", required=True)
    comparison_parser.add_argument("--output", required=True)
    calibration_parser = subparsers.add_parser("calibrate-abstention")
    calibration_parser.add_argument("--config", default="configs/abstention_bge_dev.yaml")
    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--config", default="configs/quick_eval.yaml")
    ask_parser.add_argument("--report-id")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.config)
    elif args.command == "audit-planner":
        audit_planner(args.config)
    elif args.command == "evaluate-retrieval":
        evaluate_retrieval(args.config)
    elif args.command == "compare-retrieval":
        compare_retrieval(args.baseline, args.candidate, args.output)
    elif args.command == "calibrate-abstention":
        calibrate_abstention(args.config)
    else:
        ask(args.config, args.question, args.report_id)


if __name__ == "__main__":
    main()
