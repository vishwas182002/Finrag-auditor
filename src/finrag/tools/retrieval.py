"""Graph-facing retrieval and transparent evidence-sufficiency tools."""

from __future__ import annotations

from collections.abc import Collection

from finrag.config import EvidenceConfig
from finrag.data.schemas import RetrievalHit
from finrag.indexing.index import RetrievalIndex
from finrag.retrieval.bm25 import tokenize

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "was",
    "were",
    "what",
}


def retrieve_evidence(
    index: RetrievalIndex,
    question: str,
    method: str | None = None,
    allowed_report_ids: Collection[str] | None = None,
) -> list[RetrievalHit]:
    return index.search(question, method=method, allowed_report_ids=allowed_report_ids)


def evidence_sufficiency(
    question: str, hits: list[RetrievalHit], config: EvidenceConfig
) -> tuple[bool, dict[str, float | int | str]]:
    query_tokens = {
        token for token in tokenize(question) if token not in STOPWORDS and len(token) > 2
    }
    evidence_tokens = set()
    for hit in hits:
        evidence_tokens.update(tokenize(hit.chunk.content))
    overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
    content_tokens = sum(len(tokenize(hit.chunk.content)) for hit in hits)
    reranker_applies = bool(hits and hits[0].method == "hybrid_rerank")
    reranker_score = hits[0].score if reranker_applies else config.min_reranker_score
    sufficient = (
        len(hits) >= config.min_chunks
        and content_tokens >= config.min_content_tokens
        and overlap >= config.min_token_overlap
        and reranker_score >= config.min_reranker_score
    )
    return sufficient, {
        "decision": "answer" if sufficient else "abstain",
        "chunk_count": len(hits),
        "content_tokens": content_tokens,
        "query_token_overlap": round(overlap, 6),
        "required_overlap": config.min_token_overlap,
        "top_reranker_score": round(reranker_score, 6),
        "required_reranker_score": config.min_reranker_score,
        "reranker_threshold_applied": str(reranker_applies).lower(),
    }
