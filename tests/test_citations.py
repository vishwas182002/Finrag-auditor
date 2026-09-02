from __future__ import annotations

from finrag.tools.citations import parse_citations, verify_citations


def test_citation_parsing(hits) -> None:
    citation = hits[0].chunk.citation_id
    text = f"Revenue increased. [CITATION: {citation}]"
    assert parse_citations(text) == [citation]
    assert verify_citations(text, hits).valid is True


def test_invalid_invented_citation(hits) -> None:
    result = verify_citations("Revenue increased. [CITATION: invented/chunk]", hits)
    assert result.valid is False
    assert result.invalid_ids == ["invented/chunk"]


def test_malformed_citation(hits) -> None:
    result = verify_citations("Revenue increased. [CITATION missing-colon]", hits)
    assert result.valid is False
    assert result.malformed == ["[CITATION missing-colon]"]

