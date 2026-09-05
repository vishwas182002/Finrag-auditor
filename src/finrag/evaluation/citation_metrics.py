"""Citation reference integrity and gold-evidence quality metrics."""

from __future__ import annotations

from finrag.data.schemas import AnswerResult


def citation_metrics(
    result: AnswerResult, gold_source_ids: set[str], gold_report_id: str
) -> dict[str, float]:
    lookup = {hit.chunk.citation_id: hit.chunk for hit in result.retrieved}
    cited = list(dict.fromkeys(result.citations))
    relevant_citations = [
        citation
        for citation in cited
        if citation in lookup
        and lookup[citation].report_id == gold_report_id
        and bool(set(lookup[citation].source_ids) & gold_source_ids)
    ]
    covered_sources = {
        source_id
        for citation in cited
        if citation in lookup and lookup[citation].report_id == gold_report_id
        for source_id in lookup[citation].source_ids
        if source_id in gold_source_ids
    }
    return {
        # This is an enforced output invariant, not a semantic-quality metric.
        "citation_reference_integrity": float(result.citation_verification.valid),
        "citation_precision": len(relevant_citations) / len(cited) if cited else 0.0,
        "citation_recall": len(covered_sources) / max(len(gold_source_ids), 1),
        "has_valid_citation": float(bool(result.citation_verification.valid_ids)),
    }
