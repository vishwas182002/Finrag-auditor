"""Stable grounding prompt shared by generation providers."""

from collections.abc import Sequence

from finrag.data.schemas import RetrievalHit

PLANNING_SYSTEM_PROMPT = """You are the planning stage of a financial-document QA system.
Return only the requested structured object; do not provide chain-of-thought.
Select at most three citation IDs, and only IDs present in the supplied evidence.
If arithmetic is required, provide one expression containing only numbers copied
from the selected evidence plus ordinary scaling constants such as 1, 100, 1000,
or 1000000. Preserve operand order. Never estimate or invent a value. If the
evidence does not identify every required operand, choose abstain.
Set result_unit explicitly: number, percent, USD, EUR, or GBP. Expressions must
produce base units (expand millions/billions explicitly), or percentage points
for percent. Do not assign a currency absent from the evidence.
"""

SYSTEM_PROMPT = """Extract an answer from the supplied financial-report evidence.
Return one contiguous VERBATIM passage copied from a cited chunk, followed by
[CITATION: report_id/chunk_id]. Preserve its words, numbers, units, signs and dates.
Do not paraphrase, add explanations, or combine passages. If no passage answers
the question, return exactly INSUFFICIENT_EVIDENCE. Calculated answers are rendered
separately by the trusted calculator; do not compute or rewrite them.
"""


def render_planning_request(question: str, hits: Sequence[RetrievalHit], schema: str) -> str:
    return (
        f"Question: {question}\n\nEvidence:\n{render_evidence(hits)}\n\n"
        "Return a structured answer plan matching this JSON schema:\n"
        f"{schema}"
    )


def render_evidence(hits: Sequence[RetrievalHit]) -> str:
    blocks: list[str] = []
    for hit in hits:
        chunk = hit.chunk
        blocks.append(f"[{chunk.citation_id}]\n{chunk.content}")
    return "\n\n".join(blocks)
