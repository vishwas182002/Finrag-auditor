from __future__ import annotations

from finrag.evaluation.abstention_calibration import select_threshold


def test_threshold_selection_uses_balanced_gate_accuracy() -> None:
    rows = [
        {"answerable": True, "base_rule_passes": True, "top_reranker_score": 0.9},
        {"answerable": True, "base_rule_passes": True, "top_reranker_score": 0.8},
        {"answerable": False, "base_rule_passes": True, "top_reranker_score": 0.2},
        {"answerable": False, "base_rule_passes": True, "top_reranker_score": 0.1},
    ]
    selected, _ = select_threshold(rows)
    assert 0.2 < selected["threshold"] < 0.8
    assert selected["balanced_gate_accuracy"] == 1.0
