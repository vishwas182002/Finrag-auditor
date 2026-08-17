"""Append-only JSONL checkpointing for resumable evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlCheckpoint:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def completed_ids(self) -> set[str]:
        return {str(row["evaluation_id"]) for row in self.rows()}

    def append(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

