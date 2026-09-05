"""Streamlit interface for questions, evidence traces, and saved evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from finrag.agents.graph import FinRAGWorkflow
from finrag.config import load_config
from finrag.pipeline import build_workflow

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "artifacts" / "results" / "dev_quick_v2"

st.set_page_config(page_title="FinRAG Auditor", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Building lexical, dense, and reranking indexes…")
def build_system(config_path: str) -> tuple[FinRAGWorkflow, list[str], dict[str, str]]:
    config = load_config(config_path)
    workflow, corpus = build_workflow(config, ROOT, allow_model_fallback=True)
    return workflow, corpus.report_ids, workflow.index.backends


def read_json(filename: str) -> dict[str, Any] | None:
    path = RESULTS / filename
    return json.loads(path.read_text()) if path.exists() else None


def read_project_json(relative_path: str) -> dict[str, Any] | None:
    path = ROOT / relative_path
    return json.loads(path.read_text()) if path.exists() else None


st.title("FinRAG Auditor")
st.caption("Grounded financial-document RAG with inspectable retrieval, calculation, citations, and abstention")

ask_tab, eval_tab, about_tab = st.tabs(["Ask & audit", "Evaluation", "System card"])

with ask_tab:
    if not (ROOT / "data" / "raw" / "dev.json").exists():
        st.warning(
            "Interactive retrieval is unavailable until FinQA is downloaded. "
            "Run `python scripts/download_finqa.py`. Saved evaluation results remain available."
        )
    else:
        try:
            workflow, report_ids, backends = build_system(str(ROOT / "configs" / "quick_eval.yaml"))
            with st.sidebar:
                st.subheader("Runtime")
                st.markdown(
                    "**Dense**  \n"
                    f"`{backends['dense'].removeprefix('sentence-transformers:')}`  \n\n"
                    "**Reranker**  \n"
                    f"`{backends['reranker'].removeprefix('cross-encoder:')}`"
                )
            report_choice = st.selectbox("Report scope", ["All indexed reports", *report_ids])
            question = st.text_input(
                "Financial question",
                placeholder="What was the percentage change in operating income?",
            )
            if st.button("Run grounded workflow", type="primary", disabled=not question.strip()):
                allowed = None if report_choice == "All indexed reports" else [report_choice]
                with st.spinner("Retrieving and auditing evidence…"):
                    result = workflow.answer(question.strip(), allowed_report_ids=allowed)
                status_col, latency_col, provider_col = st.columns(3)
                status_col.metric("Decision", "ABSTAIN" if result.abstained else "ANSWER")
                latency_col.metric("Total latency", f"{result.latency_ms['total']:.1f} ms")
                provider_col.metric("Provider", result.provider.removeprefix("deterministic-"))
                if result.abstained:
                    st.warning(result.answer)
                else:
                    st.success(result.answer)
                if result.calculator_expression:
                    st.subheader("Calculator")
                    st.code(
                        f"{result.calculator_expression} = {result.calculator_result}",
                        language="text",
                    )
                if result.plan is not None:
                    st.subheader("Validated answer plan")
                    st.json(
                        {
                            "plan": result.plan.model_dump(),
                            "validation": (
                                result.plan_validation.model_dump()
                                if result.plan_validation is not None
                                else None
                            ),
                        },
                        expanded=False,
                    )
                st.subheader("Cited and retrieved evidence")
                for hit in result.retrieved:
                    cited = hit.chunk.citation_id in result.citations
                    label = (
                        f"{'✓ CITED' if cited else 'Retrieved'} · rank {hit.rank} · "
                        f"{hit.chunk.citation_id} · score {hit.score:.4f}"
                    )
                    with st.expander(label, expanded=cited or hit.rank == 1):
                        if cited:
                            st.markdown("**Citation verified against this retrieved chunk.**")
                        st.write(hit.chunk.content)
                        score_frame = pd.DataFrame(
                            [{"component": key, "score": value} for key, value in hit.component_scores.items()]
                        )
                        if not score_frame.empty:
                            st.dataframe(score_frame, hide_index=True, width="stretch")
                st.subheader("Operational trace")
                st.json(result.trace, expanded=False)
                st.subheader("Latency breakdown")
                st.dataframe(
                    pd.DataFrame(
                        [{"stage": key, "milliseconds": value} for key, value in result.latency_ms.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"The interactive system could not start: {exc}")
            st.info("The Evaluation tab can still display previously generated artifacts.")

with eval_tab:
    st.subheader("Corrected historical scoring · no new inference")
    st.caption("V1 retrieval claims are superseded. These regrades retain the original answers and do not evaluate the current workflow.")
    corrected_rows = []
    for label in ("historical_workflow", "dev_quick"):
        regrade = read_project_json(f"artifacts/regraded_v2/{label}/summary.json")
        if regrade:
            corrected_rows.append({"cohort": label, "questions": regrade["answerable_questions"], **regrade["retrieval"], **regrade["answer"]})
    if corrected_rows:
        st.dataframe(pd.DataFrame(corrected_rows), hide_index=True, width="stretch")
    retrieval = read_json("retrieval_metrics.json")
    answers = read_json("answer_metrics.json")
    ablation_path = RESULTS / "ablation.csv"
    latency_path = RESULTS / "latency.csv"
    if retrieval is None or answers is None or not ablation_path.exists() or retrieval.get("metadata", {}).get("retrieval_metric_version") != "report-scoped-v2":
        st.info("No saved evaluation run found. Run `finrag evaluate --config configs/quick_eval.yaml`.")
    else:
        metadata = retrieval["metadata"]
        st.subheader("Development workflow · current scoring")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", metadata["sample_size"])
        c2.metric("Reports", metadata["report_count"])
        c3.metric("Corpus chunks", metadata["corpus_chunks"])
        c4.metric("Provider", metadata["provider"].removeprefix("deterministic-"))
        st.subheader("Retrieval ablation")
        ablation = pd.read_csv(ablation_path)
        st.dataframe(ablation, hide_index=True, width="stretch")
        retrieval_chart = ablation[ablation["configuration"] != "full_workflow"].set_index("configuration")
        st.bar_chart(retrieval_chart[["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]])
        st.subheader("Answer, grounding, and abstention")
        st.json({"answer": answers["answer"], "citations": answers["citations"]}, expanded=True)
        if latency_path.exists():
            latency = pd.read_csv(latency_path)
            st.subheader("Latency samples")
            st.dataframe(latency, hide_index=True, width="stretch")

    planner_audit = read_project_json(
        "artifacts/planner_audit/legacy_planner_dev_v2/planner_audit.json"
    )
    bge_ablation = read_project_json(
        "artifacts/retrieval_ablation/bge_reranker_dev_v2/retrieval_metrics.json"
    )
    gate_calibration = read_project_json(
        "artifacts/abstention_calibration/bge_gate_dev/abstention_calibration.json"
    )
    if planner_audit is not None and planner_audit.get("metadata", {}).get("retrieval_metric_version") == "report-scoped-v2":
        st.subheader("Legacy planner diagnosis · development split")
        st.json(
            {
                "retrieved_evidence": planner_audit["retrieved_evidence"],
                "conditional_on_gold_available": planner_audit[
                    "retrieved_evidence_conditional_on_gold_available"
                ],
                "oracle_gold_evidence": planner_audit["oracle_gold_evidence"],
            },
            expanded=False,
        )
    if bge_ablation is not None and bge_ablation.get("metadata", {}).get("retrieval_metric_version") == "report-scoped-v2":
        st.subheader("BGE reranker selection · development split")
        method_rows = [
            {"configuration": method, **details["metrics"]}
            for method, details in bge_ablation["methods"].items()
        ]
        st.dataframe(pd.DataFrame(method_rows), hide_index=True, width="stretch")
    if gate_calibration is not None:
        st.subheader("Evidence-gate calibration · development split")
        st.json(
            {
                "zero_threshold": gate_calibration["zero_threshold"],
                "selection_constraint": gate_calibration["selection_constraint"],
                "selected": gate_calibration["selected"],
                "unconstrained_best": gate_calibration["unconstrained_best"],
            },
            expanded=False,
        )

with about_tab:
    st.markdown(
        """
### What is audited

- BM25 lexical and BGE dense retrieval are fused with Reciprocal Rank Fusion.
- The BGE cross-encoder reranks only the strongest candidates.
- A configured overlap gate can abstain before planning.
- The provider selects evidence and emits a typed calculation/extraction plan.
- Plan validation blocks invented citations and operands absent from selected evidence.
- Arithmetic is restricted to a Decimal-backed AST whitelist.
- Calculations are rendered directly from the Decimal result.
- Extractions must match a verbatim passage in their cited evidence.
- Output consistency does not prove the plan answers the financial question.
- The trace contains operational decisions and timings, never hidden chain-of-thought.

The default deterministic extractive provider makes retrieval, routing, and citation tests
fully reproducible without credentials. It is deliberately weak as an answer generator and
is labeled as such in every result artifact.
"""
    )
