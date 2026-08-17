"""Provider abstraction with structured planning and a deterministic fallback."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

from finrag.data.schemas import AnswerPlan, RetrievalHit
from finrag.generation.planning import legacy_answer_plan
from finrag.generation.prompts import (
    PLANNING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    render_evidence,
    render_planning_request,
)


class GenerationProvider(ABC):
    name: str

    @abstractmethod
    def plan(self, question: str, hits: list[RetrievalHit]) -> AnswerPlan:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        plan: AnswerPlan,
        calculator_result: str | None = None,
    ) -> str:
        raise NotImplementedError


class ExtractiveProvider(GenerationProvider):
    """Deterministic infrastructure fallback; it is not presented as an LLM."""

    name = "deterministic-extractive-v1"

    def plan(self, question: str, hits: list[RetrievalHit]) -> AnswerPlan:
        return legacy_answer_plan(question, hits)

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        plan: AnswerPlan,
        calculator_result: str | None = None,
    ) -> str:
        if not hits:
            return "INSUFFICIENT_EVIDENCE"
        citation = f"[CITATION: {hits[0].chunk.citation_id}]"
        context_words = re.findall(r"[A-Za-z]{3,}", hits[0].chunk.content)[:5]
        context_label = " ".join(context_words) or "the retrieved evidence"
        if calculator_result is not None:
            suffix = "%" if any(
                word in question.lower() for word in ("percent", "percentage")
            ) else ""
            return (
                f"Using {context_label}, the calculated result is "
                f"{calculator_result}{suffix}. {citation}"
            )
        numbers = re.findall(
            r"(?<![\w])[-$€£]?\(?\d[\d,]*(?:\.\d+)?\)?%?", hits[0].chunk.content
        )
        if numbers:
            return f"According to {context_label}, the retrieved value is {numbers[-1]}. {citation}"
        sentence = hits[0].chunk.content.strip().split(". ")[0].rstrip(".")
        return f"{sentence}. {citation}"


class OpenAICompatibleProvider(GenerationProvider):
    """LangChain chat provider supporting OpenAI and compatible free-tier endpoints."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 0,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the `openai` project extra") from exc
        self.model_name = model
        self.name = f"openai-compatible:{model}" if base_url else f"openai:{model}"
        client_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = ChatOpenAI(**client_kwargs)
        self.planner = self.client.with_structured_output(AnswerPlan)

    def plan(self, question: str, hits: list[RetrievalHit]) -> AnswerPlan:
        response = self.planner.invoke(
            [
                ("system", PLANNING_SYSTEM_PROMPT),
                (
                    "human",
                    render_planning_request(
                        question, hits, str(AnswerPlan.model_json_schema())
                    ),
                ),
            ]
        )
        return response if isinstance(response, AnswerPlan) else AnswerPlan.model_validate(response)

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        plan: AnswerPlan,
        calculator_result: str | None = None,
    ) -> str:
        evidence = render_evidence(hits)
        user = (
            f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
            f"Validated plan: {plan.model_dump_json()}\n"
            f"Calculator result: {calculator_result or 'not used'}"
        )
        response = self.client.invoke([("system", SYSTEM_PROMPT), ("human", user)])
        return str(response.content)


def create_provider(
    kind: str,
    model: str,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    max_retries: int = 0,
) -> GenerationProvider:
    if kind == "extractive":
        return ExtractiveProvider()
    if kind in {"openai", "openai_compatible"}:
        return OpenAICompatibleProvider(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key_env=api_key_env,
            max_retries=max_retries,
        )
    raise ValueError(f"Unknown provider: {kind}")
