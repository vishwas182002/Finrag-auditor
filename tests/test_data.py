from __future__ import annotations

import hashlib
import json

from finrag.config import ChunkingConfig
from finrag.data.finqa import load_finqa, parse_record, select_from_manifest, select_held_out
from finrag.data.schemas import FinQAExample
from finrag.indexing.chunking import build_chunks, serialize_table_row, serialize_table_zero


def test_finqa_parsing() -> None:
    record = {
        "id": "A/2020/p.pdf-1",
        "filename": "A/2020/p.pdf",
        "pre_text": ["Before"],
        "post_text": ["After"],
        "table": [["Metric", "Value"], ["Revenue", "10"]],
        "qa": {
            "question": "Revenue?",
            "answer": "10",
            "program": "subtract(11, 1)",
            "gold_inds": {"table_1": "row"},
        },
    }
    parsed = parse_record(record)
    assert parsed.question_id == record["id"]
    assert parsed.gold_source_ids == ("table_1",)
    assert parsed.table[1][0] == "Revenue"


def test_load_finqa(tmp_path) -> None:
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "x-1",
                    "filename": "x",
                    "pre_text": [],
                    "post_text": [],
                    "table": [],
                    "qa": {"question": "q", "answer": "1", "gold_inds": {}},
                }
            ]
        )
    )
    assert len(load_finqa(path)) == 1


def test_table_serialization_preserves_headers() -> None:
    rendered = serialize_table_row(("Metric", "2023", "2022"), ("Revenue", "120", "100"))
    assert "Metric = Revenue" in rendered
    assert "2023 = 120" in rendered
    assert "2022 = 100" in rendered
    assert "column_1 = 2023" in serialize_table_zero(("Metric", "2023", "2022"))


def test_chunk_metadata_preservation(example: FinQAExample) -> None:
    chunks = build_chunks([example], ChunkingConfig())
    table_zero = next(chunk for chunk in chunks if chunk.source_ids == ("table_0",))
    table = next(chunk for chunk in chunks if chunk.source_ids == ("table_1",))
    post = next(chunk for chunk in chunks if chunk.source_section == "post_text")
    assert table.report_id == example.report_id
    assert table.source_ids == ("table_1",)
    assert table.metadata["header"] == ["Metric", "2023", "2022"]
    assert table_zero.metadata["role"] == "header_or_first_row"
    assert post.source_ids == ("text_1",)


def test_held_out_sampling_is_deterministic(example: FinQAExample) -> None:
    examples = [
        example,
        example.model_copy(update={"question_id": "other-1", "report_id": "other"}),
    ]
    assert select_held_out(examples, 2, 42) == select_held_out(examples, 2, 42)


def test_manifest_selection_preserves_frozen_order(tmp_path, example: FinQAExample) -> None:
    other = example.model_copy(update={"question_id": "other-1", "report_id": "other"})
    manifest = tmp_path / "manifest.json"
    ids = ["other-1", example.question_id]
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    manifest.write_text(
        json.dumps(
            {"sample_size": 2, "question_ids": ids, "question_ids_sha256": digest}
        )
    )
    selected = select_from_manifest([example, other], manifest)
    assert [item.question_id for item in selected] == ["other-1", example.question_id]
