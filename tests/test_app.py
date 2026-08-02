from pathlib import Path

from streamlit.testing.v1 import AppTest

import llm_client

SRC = Path(__file__).resolve().parent.parent / "src"


class FakeLLMClient:
    """Stand-in for LLMClient that satisfies the surface area app.py uses."""

    def __init__(self, reply="hello", error: BaseException | None = None):
        self.model = "fake/model"
        self._reply = reply
        self._error = error
        self.history = [{"role": "system", "content": "system"}]

    def reset(self) -> None:
        self.history = [{"role": "system", "content": "system"}]

    def estimated_tokens(self) -> int:
        return sum(len(m.get("content", "")) // 4 + 1 for m in self.history)

    def ask(self, prompt: str) -> str:
        if self._error is not None:
            raise self._error
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": self._reply})
        return self._reply


def _app(fake: FakeLLMClient | None = None) -> AppTest:
    at = AppTest.from_file(str(SRC / "app.py"))
    if fake is not None:
        at.session_state["client"] = fake
    return at


def _markdown_text(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown]


def test_missing_key_shows_setup_error(monkeypatch):
    monkeypatch.setattr(llm_client, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = _app()
    at.run()
    assert len(at.exception) == 0
    assert any("OPENROUTER_API_KEY" in e.value for e in at.error)


def test_chat_flow_appends_transcript():
    at = _app(FakeLLMClient())
    at.run()
    at.chat_input[0].set_value("hi").run()
    assert len(at.exception) == 0
    text = _markdown_text(at)
    assert "hi" in text
    assert "hello" in text


def test_failed_ask_shows_persistent_error_banner():
    at = _app(FakeLLMClient(error=llm_client.LLMError("boom")))
    at.run()
    at.chat_input[0].set_value("hi").run()
    assert len(at.exception) == 0
    assert any("boom" in e.value for e in at.error)


def test_clear_conversation_empties_transcript():
    at = _app(FakeLLMClient())
    at.run()
    at.chat_input[0].set_value("hi").run()
    assert "hello" in _markdown_text(at)
    at.button[0].click().run()
    assert len(at.exception) == 0
    assert "hello" not in _markdown_text(at)


def test_sidebar_offers_three_selectable_models():
    at = _app(FakeLLMClient())
    at.run()
    assert len(at.exception) == 0
    model_box = at.selectbox[0]
    assert len(model_box.options) == 3
    assert model_box.value == "NVIDIA Nemotron 3 Ultra 550B"
