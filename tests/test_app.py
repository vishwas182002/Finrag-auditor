"""Execute the Streamlit script headlessly to catch import and API breakage."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_app_renders_saved_artifacts_without_data() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "FinRAG Auditor"
    # Without data/raw/dev.json the interactive tab must degrade gracefully...
    if not (APP.parent / "data" / "raw" / "dev.json").exists():
        assert any("download_finqa" in warning.value for warning in at.warning)
    # ...while the committed evaluation artifacts still render.
    subheaders = [block.value for block in at.subheader]
    assert "Retrieval ablation" in subheaders
    assert "Legacy planner diagnosis · development split" in subheaders
