import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_runs_without_exception():
    at = AppTest.from_file(str(SRC / "app.py")).run()
    assert len(at.exception) == 0
    assert at.title
