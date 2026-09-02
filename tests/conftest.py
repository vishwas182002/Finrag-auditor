from __future__ import annotations

import pytest

from finrag.data.schemas import DocumentChunk, FinQAExample, RetrievalHit


@pytest.fixture
def example() -> FinQAExample:
    return FinQAExample(
        question_id="ACME/2023/page_1.pdf-1",
        report_id="ACME/2023/page_1.pdf",
        question="What was the change in revenue?",
        answer="20",
        program="subtract(120, 100)",
        gold_source_ids=("table_1",),
        pre_text=("Acme reported annual results.",),
        post_text=("Revenue increased during the year.",),
        table=(("Metric", "2023", "2022"), ("Revenue", "120", "100")),
    )


@pytest.fixture
def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            report_id="ACME/2023/page_1.pdf",
            chunk_id="table_1",
            source_section="table",
            source_type="table",
            source_ids=("table_1",),
            content="Row Revenue: Metric = Revenue | 2023 = 120 | 2022 = 100",
            raw_content="[['Revenue', '120', '100']]",
        ),
        DocumentChunk(
            report_id="ACME/2023/page_1.pdf",
            chunk_id="text_0",
            source_section="pre_text",
            source_type="text",
            source_ids=("text_0",),
            content="Acme reported annual results and costs of 80.",
            raw_content="Acme reported annual results and costs of 80.",
        ),
        DocumentChunk(
            report_id="OTHER/2023/page_2.pdf",
            chunk_id="text_0",
            source_section="pre_text",
            source_type="text",
            source_ids=("text_0",),
            content="Other company discussed emissions targets.",
            raw_content="Other company discussed emissions targets.",
        ),
    ]


@pytest.fixture
def hits(chunks: list[DocumentChunk]) -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk=chunks[0],
            score=1.0,
            rank=1,
            method="test",
            component_scores={"test": 1.0},
        )
    ]

