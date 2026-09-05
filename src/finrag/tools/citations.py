"""Citation syntax parsing and retrieved-evidence verification."""

from __future__ import annotations

import re

from finrag.data.schemas import AnswerPlan, CitationVerification, RetrievalHit
from finrag.retrieval.bm25 import tokenize

CITATION_RE = re.compile(r"\[CITATION:\s*([^\]\n]+?)\s*\]")
CITATION_LIKE_RE = re.compile(r"\[CITATION[^\]]*\]")


def parse_citations(text: str) -> list[str]:
    return [match.strip() for match in CITATION_RE.findall(text)]


def verify_citations(answer: str, retrieved: list[RetrievalHit]) -> CitationVerification:
    cited_ids = parse_citations(answer)
    valid_lookup = {hit.chunk.citation_id: hit.chunk for hit in retrieved}
    valid_ids = [citation for citation in cited_ids if citation in valid_lookup]
    invalid_ids = [citation for citation in cited_ids if citation not in valid_lookup]
    well_formed = {match.group(0) for match in CITATION_RE.finditer(answer)}
    malformed = [token for token in CITATION_LIKE_RE.findall(answer) if token not in well_formed]
    reasons: list[str] = []
    if not cited_ids:
        reasons.append("answer_has_no_citations")
    if invalid_ids:
        reasons.append("citation_not_in_retrieved_evidence")
    if malformed:
        reasons.append("malformed_citation")
    # Lightweight support audit: a cited claim must share a non-trivial content token
    # with at least one cited chunk. Calculator-derived numbers are supported by the
    # cited operands even when the final result is not verbatim in the source.
    answer_without_citations = CITATION_RE.sub("", answer)
    answer_tokens = {
        token
        for token in tokenize(answer_without_citations)
        if len(token) > 2 and not token.isdigit()
    }
    evidence_tokens: set[str] = set()
    for citation in valid_ids:
        evidence_tokens.update(tokenize(valid_lookup[citation].content))
    if valid_ids and answer_tokens and not (answer_tokens & evidence_tokens):
        reasons.append("cited_evidence_has_no_lexical_support")
    return CitationVerification(
        valid=not reasons,
        cited_ids=cited_ids,
        valid_ids=valid_ids,
        invalid_ids=invalid_ids,
        malformed=malformed,
        reasons=reasons,
    )


def render_calculation_answer(result: str, plan: AnswerPlan) -> str:
    """Render the trusted Decimal result; no model may rewrite this value."""
    prefix = {"USD": "$", "EUR": "€", "GBP": "£"}.get(plan.result_unit, "")
    suffix = "%" if plan.result_unit == "percent" else ""
    citations = " ".join(f"[CITATION: {citation}]" for citation in plan.selected_citation_ids)
    return f"{prefix}{result}{suffix} {citations}"


def verify_answer(
    answer: str,
    retrieved: list[RetrievalHit],
    plan: AnswerPlan,
    calculator_result: str | None = None,
) -> CitationVerification:
    """Enforce a bounded output contract, not an entailment/QA-accuracy claim.

    Calculations are the canonical tool result plus the plan's unit and citations.
    Extractions must be a contiguous, verbatim passage in every cited chunk.
    Matching a passage does not prove it answers the question or uses the right year.
    """
    verification = verify_citations(answer, retrieved)
    reasons = list(verification.reasons)
    # Lexical overlap is unnecessary for a canonical calculation or exact quote.
    reasons = [reason for reason in reasons if reason != "cited_evidence_has_no_lexical_support"]
    if plan.answer_type == "calculation":
        if calculator_result is None or answer != render_calculation_answer(
            calculator_result, plan
        ):
            reasons.append("answer_does_not_match_calculator_result")
    elif plan.answer_type == "extractive":
        passage = " ".join(CITATION_RE.sub("", answer).strip().split())
        lookup = {hit.chunk.citation_id: hit.chunk for hit in retrieved}
        if not passage or any(
            not re.search(
                r"(?<![\w.])" + re.escape(passage) + r"(?![\w.])",
                " ".join(lookup[citation].content.split()),
            )
            for citation in verification.valid_ids
        ):
            reasons.append("answer_is_not_a_verbatim_cited_passage")
    else:
        reasons.append("unsupported_answer_type")
    verification.reasons = reasons
    verification.valid = not reasons
    return verification
