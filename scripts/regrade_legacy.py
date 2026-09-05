"""Reproduce the v2 regrade of the two archived workflow runs; no models required."""

from pathlib import Path

from finrag.evaluation.regrade import regrade_predictions


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for label, source in [
        ("historical_workflow", "results/predictions.jsonl"),
        ("dev_quick", "results/dev_quick/predictions.jsonl"),
    ]:
        report = regrade_predictions(
            root / "artifacts/legacy_v1" / source, root / "artifacts/regraded_v2" / label
        )
        print(label, report)


if __name__ == "__main__":
    main()
