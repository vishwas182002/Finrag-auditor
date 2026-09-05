# Superseded v1 evaluation artifacts

These files are preserved byte-for-byte from commit
`8addb6bdde7fdf7ccfd50aa6168564509102a0f5` for provenance. They are not current
performance claims. The original evaluator matched report-local source IDs
without report identity; its retrieval, citation, and evidence-availability
statistics can credit the wrong report. Its answer grader ignored explicit
financial scales and currency conflicts. Its nDCG could repeat evidence credit.

The original 883-question comparison's **+10.92 MRR points (CI 8.65–13.25)** is
superseded. Complete retrieval hits were not saved for that comparison, so it
cannot be repaired from those per-question scalar scores alone.

`../regraded_v2/` corrects scoring on the available historical workflow hits and
answers. It does not rerun inference or apply the current strict answer verifier.
New model runs use distinct v2 directories outside this archive. Never mix their
scores, confidence intervals, model versions, or latency measurements with v1.
