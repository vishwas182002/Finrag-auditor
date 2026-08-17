"""FinQA chunking that preserves paragraph and table-row identities."""

from __future__ import annotations

from collections.abc import Iterable

from finrag.config import ChunkingConfig
from finrag.data.schemas import DocumentChunk, FinQAExample


def serialize_table_row(header: tuple[str, ...], row: tuple[str, ...]) -> str:
    if not row:
        return ""
    label = row[0]
    pairs = [f"{header[i]} = {value}" for i, value in enumerate(row) if i < len(header)]
    return f"Row {label}: " + " | ".join(pairs)


def serialize_table_zero(row: tuple[str, ...]) -> str:
    """Preserve row zero, which may be a header or data in headerless FinQA tables."""
    pairs = [f"column_{index} = {value}" for index, value in enumerate(row)]
    return "Table row 0: " + " | ".join(pairs)


def _text_chunks(
    report_id: str,
    paragraphs: tuple[str, ...],
    section: str,
    offset: int,
    config: ChunkingConfig,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for local_idx, paragraph in enumerate(paragraphs):
        source_idx = offset + local_idx
        source_id = f"text_{source_idx}"
        cleaned = " ".join(paragraph.split())
        if not cleaned or cleaned == ".":
            continue
        if len(cleaned) <= config.text_chunk_chars:
            spans = [cleaned]
        else:
            spans = []
            start = 0
            while start < len(cleaned):
                spans.append(cleaned[start : start + config.text_chunk_chars])
                if start + config.text_chunk_chars >= len(cleaned):
                    break
                start += config.text_chunk_chars - config.text_overlap_chars
        for part, content in enumerate(spans):
            chunk_id = source_id if len(spans) == 1 else f"{source_id}_part_{part}"
            chunks.append(
                DocumentChunk(
                    report_id=report_id,
                    chunk_id=chunk_id,
                    source_section=section,  # type: ignore[arg-type]
                    source_type="text",
                    source_ids=(source_id,),
                    content=content,
                    raw_content=paragraph,
                    metadata={"paragraph_index": source_idx, "part": part},
                )
            )
    return chunks


def chunk_report(example: FinQAExample, config: ChunkingConfig) -> list[DocumentChunk]:
    chunks = _text_chunks(example.report_id, example.pre_text, "pre_text", 0, config)
    chunks.extend(
        _text_chunks(
            example.report_id,
            example.post_text,
            "post_text",
            len(example.pre_text),
            config,
        )
    )
    if example.table:
        header = example.table[0]
        chunks.append(
            DocumentChunk(
                report_id=example.report_id,
                chunk_id="table_0",
                source_section="table",
                source_type="table",
                source_ids=("table_0",),
                content=serialize_table_zero(header),
                raw_content=str([list(header)]),
                metadata={
                    "row_start": 0,
                    "role": "header_or_first_row",
                },
            )
        )
        rows = example.table[1:]
        step = config.table_rows_per_chunk
        for start in range(0, len(rows), step):
            group = rows[start : start + step]
            source_ids = tuple(f"table_{i}" for i in range(start + 1, start + 1 + len(group)))
            content = "\n".join(serialize_table_row(header, row) for row in group)
            chunks.append(
                DocumentChunk(
                    report_id=example.report_id,
                    chunk_id=source_ids[0] if len(source_ids) == 1 else f"{source_ids[0]}_{source_ids[-1]}",
                    source_section="table",
                    source_type="table",
                    source_ids=source_ids,
                    content=content,
                    raw_content=str([list(row) for row in group]),
                    metadata={"header": list(header), "row_start": start + 1},
                )
            )
    return chunks


def build_chunks(examples: Iterable[FinQAExample], config: ChunkingConfig) -> list[DocumentChunk]:
    by_report: dict[str, FinQAExample] = {}
    for example in examples:
        by_report.setdefault(example.report_id, example)
    chunks: list[DocumentChunk] = []
    for report_id in sorted(by_report):
        chunks.extend(chunk_report(by_report[report_id], config))
    return chunks
