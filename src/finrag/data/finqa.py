"""Official FinQA parser and deterministic held-out sampling."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from finrag.data.schemas import FinQAExample


def parse_record(record: dict[str, Any]) -> FinQAExample:
    qa = record["qa"]
    gold_ids = tuple(sorted(qa.get("gold_inds", {}).keys()))
    table = tuple(tuple(str(cell) for cell in row) for row in record.get("table", []))
    return FinQAExample(
        question_id=str(record["id"]),
        report_id=str(record["filename"]),
        question=str(qa["question"]),
        answer=str(qa["answer"]),
        program=str(qa.get("program", "")),
        gold_source_ids=gold_ids,
        pre_text=tuple(str(x) for x in record.get("pre_text", [])),
        post_text=tuple(str(x) for x in record.get("post_text", [])),
        table=table,
    )


def load_finqa(path: str | Path) -> list[FinQAExample]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"FinQA split not found at {source}. Run `python scripts/download_finqa.py`."
        )
    raw = json.loads(source.read_text())
    if not isinstance(raw, list):
        raise ValueError("FinQA split must be a JSON array")
    return [parse_record(record) for record in raw]


def select_held_out(
    examples: list[FinQAExample], sample_size: int, seed: int
) -> list[FinQAExample]:
    """Select across reports and evidence types without reading answers for tuning."""
    rng = random.Random(seed)
    by_report: dict[str, list[FinQAExample]] = {}
    for example in examples:
        by_report.setdefault(example.report_id, []).append(example)
    reports = sorted(by_report)
    rng.shuffle(reports)
    selected: list[FinQAExample] = []
    # One question/report first maximizes report diversity.
    for report_id in reports:
        bucket = sorted(by_report[report_id], key=lambda x: x.question_id)
        selected.append(rng.choice(bucket))
        if len(selected) == sample_size:
            return sorted(selected, key=lambda x: x.question_id)
    remaining = [x for x in examples if x not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return sorted(selected, key=lambda x: x.question_id)


def select_from_manifest(
    examples: list[FinQAExample], manifest_path: str | Path
) -> list[FinQAExample]:
    """Load an immutable question-ID cohort and fail on drift or missing IDs."""
    manifest = json.loads(Path(manifest_path).read_text())
    requested = [str(question_id) for question_id in manifest.get("question_ids", [])]
    if not requested:
        raise ValueError("Question manifest contains no question_ids")
    if len(requested) != len(set(requested)):
        raise ValueError("Question manifest contains duplicate question_ids")
    expected_digest = manifest.get("question_ids_sha256")
    actual_digest = hashlib.sha256(("\n".join(requested) + "\n").encode()).hexdigest()
    if expected_digest is not None and str(expected_digest) != actual_digest:
        raise ValueError("Question manifest question_ids_sha256 does not match question_ids")
    by_id = {example.question_id: example for example in examples}
    missing = [question_id for question_id in requested if question_id not in by_id]
    if missing:
        raise ValueError(f"Question manifest contains IDs absent from split: {missing[:3]}")
    expected_size = manifest.get("sample_size")
    if expected_size is not None and int(expected_size) != len(requested):
        raise ValueError("Question manifest sample_size does not match question_ids")
    return [by_id[question_id] for question_id in requested]


def select_configured(
    examples: list[FinQAExample],
    sample_size: int,
    seed: int,
    manifest_path: str | Path | None = None,
) -> list[FinQAExample]:
    if manifest_path is not None:
        return select_from_manifest(examples, manifest_path)
    return select_held_out(examples, sample_size, seed)


def selected_corpus_examples(
    all_examples: list[FinQAExample], selected: list[FinQAExample], scope: str
) -> list[FinQAExample]:
    if scope == "full_split":
        return all_examples
    report_ids = {example.report_id for example in selected}
    return [example for example in all_examples if example.report_id in report_ids]
