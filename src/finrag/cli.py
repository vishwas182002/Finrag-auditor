"""Command-line entrypoint for evaluation and one-off questions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from finrag import __version__
from finrag.config import load_config
from finrag.evaluation.abstention_calibration import run_abstention_calibration
from finrag.evaluation.planner_audit import run_planner_audit
from finrag.evaluation.retrieval_ablation import run_retrieval_ablation
from finrag.evaluation.retrieval_comparison import compare_retrieval_runs
from finrag.evaluation.runner import run_evaluation
from finrag.logging import configure_logging
from finrag.pipeline import build_workflow, resolve_project_root


def evaluate(config_path: str) -> None:
    root = resolve_project_root(config_path)
    config = load_config(config_path)
    result = run_evaluation(config, root)
    print(json.dumps(result, indent=2))


def audit_planner(config_path: str) -> None:
    root = resolve_project_root(config_path)
    config = load_config(config_path)
    result = run_planner_audit(config, root)
    print(json.dumps(result, indent=2))


def evaluate_retrieval(config_path: str) -> None:
    root = resolve_project_root(config_path)
    config = load_config(config_path)
    result = run_retrieval_ablation(config, root)
    print(json.dumps(result, indent=2))


def compare_retrieval(baseline: str, candidate: str, output: str) -> None:
    result = compare_retrieval_runs(Path(baseline), Path(candidate), Path(output))
    print(json.dumps(result, indent=2))


def calibrate_abstention(config_path: str) -> None:
    root = resolve_project_root(config_path)
    config = load_config(config_path)
    result = run_abstention_calibration(config, root)
    print(json.dumps(result, indent=2))


def ask(config_path: str, question: str, report_id: str | None) -> None:
    root = resolve_project_root(config_path)
    config = load_config(config_path)
    workflow, _ = build_workflow(config, root)
    result = workflow.answer(question, allowed_report_ids=[report_id] if report_id else None)
    print(result.model_dump_json(indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finrag",
        description="Grounded financial-document RAG evaluation harness.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="Run the full workflow evaluation")
    evaluate_parser.add_argument("--config", default="configs/quick_eval.yaml")
    planner_parser = subparsers.add_parser("audit-planner", help="Audit the legacy planner")
    planner_parser.add_argument("--config", default="configs/planner_audit.yaml")
    retrieval_parser = subparsers.add_parser(
        "evaluate-retrieval", help="Retrieval-only ablation; no generation calls"
    )
    retrieval_parser.add_argument("--config", default="configs/reranker_bge_dev.yaml")
    comparison_parser = subparsers.add_parser(
        "compare-retrieval", help="Paired bootstrap comparison of two retrieval runs"
    )
    comparison_parser.add_argument("--baseline", required=True)
    comparison_parser.add_argument("--candidate", required=True)
    comparison_parser.add_argument("--output", required=True)
    calibration_parser = subparsers.add_parser(
        "calibrate-abstention", help="Calibrate the evidence-sufficiency gate on dev"
    )
    calibration_parser.add_argument("--config", default="configs/abstention_bge_dev.yaml")
    ask_parser = subparsers.add_parser("ask", help="Answer one question with a full trace")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--config", default="configs/quick_eval.yaml")
    ask_parser.add_argument("--report-id")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
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
