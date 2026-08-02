from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from unittest import mock

import httpx
import openai
import pytest

import llm_client
from llm_client import LLMClient, LLMError


class FakeCompletions:
    def __init__(self, side_effect):
        self.side_effect = side_effect
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        effect = self.side_effect.pop(0) if isinstance(self.side_effect, list) else self.side_effect
        if isinstance(effect, BaseException):
            raise effect
        return effect


class FakeClient:
    def __init__(self, side_effect):
        self.chat = SimpleNamespace(completions=FakeCompletions(side_effect))


def _ok_response(reply="hello"):
    choice = SimpleNamespace(message=SimpleNamespace(content=reply))
    return SimpleNamespace(choices=[choice])


def _api_error(status_code, headers=None):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code, headers=headers or {}, request=request)
    return openai.APIStatusError(f"err {status_code}", response=response, body=None)


@pytest.fixture
def client():
    fake = FakeClient(_ok_response())
    return LLMClient(api_key="test-key", client=fake, max_retries=3)


def test_ask_returns_reply_and_grows_history(client):
    assert client.ask("hi") == "hello"
    roles = [m["role"] for m in client.history]
    assert roles == ["system", "user", "assistant"]


def test_reset_clears_history(client):
    client.ask("hi")
    client.reset()
    assert client.history == [{"role": "system", "content": client.system_prompt}]


def test_empty_prompt_raises(client):
    with pytest.raises(LLMError):
        client.ask("   ")


def test_rate_limit_retries_then_succeeds(monkeypatch):
    fake = FakeClient([_api_error(429, headers={"retry-after": "0.1"}), _ok_response()])
    sleeper = mock.Mock()
    monkeypatch.setattr(llm_client.time, "sleep", sleeper)
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)

    assert client.ask("hi") == "hello"
    assert len(fake.chat.completions.calls) == 2
    sleeper.assert_called_once()


def test_retries_exhausted_raises(monkeypatch):
    fake = FakeClient(_api_error(429))
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)

    with pytest.raises(LLMError):
        client.ask("hi")
    assert len(fake.chat.completions.calls) == 3


def test_non_retryable_fails_fast(client):
    fake = FakeClient(_api_error(400))
    bad = LLMClient(api_key="test-key", client=fake, max_retries=3)
    with pytest.raises(LLMError):
        bad.ask("hi")
    assert len(fake.chat.completions.calls) == 1


def test_timeout_is_retried(monkeypatch):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    fake = FakeClient([openai.APITimeoutError(request), _ok_response()])
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)

    assert client.ask("hi") == "hello"
    assert len(fake.chat.completions.calls) == 2


def test_connection_error_is_retried(monkeypatch):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    fake = FakeClient(
        [openai.APIConnectionError(message="conn", request=request), _ok_response()]
    )
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)

    assert client.ask("hi") == "hello"
    assert len(fake.chat.completions.calls) == 2


def test_history_is_trimmed():
    fake = FakeClient(_ok_response(reply="ok"))
    client = LLMClient(
        api_key="test-key", client=fake, max_retries=1, max_history_tokens=100
    )
    for i in range(10):
        client.ask("x" * 200 + str(i))
    assert client.estimated_tokens() <= 250
    assert len(client.history) < 10


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm_client, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError):
        LLMClient(api_key=None, client=FakeClient(_ok_response()))


def test_failed_ask_rolls_back_user_message(monkeypatch):
    fake = FakeClient(_api_error(400))
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)
    with pytest.raises(LLMError):
        client.ask("first")
    assert client.history == [{"role": "system", "content": client.system_prompt}]

    client.client = FakeClient(_ok_response("hello"))
    assert client.ask("second") == "hello"
    assert [m["role"] for m in client.history] == ["system", "user", "assistant"]


def test_retry_exhaustion_rolls_back_user_message(monkeypatch):
    fake = FakeClient(_api_error(429))
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=2)
    with pytest.raises(LLMError):
        client.ask("hi")
    assert client.history == [{"role": "system", "content": client.system_prompt}]


def test_empty_reply_is_retried(monkeypatch):
    fake = FakeClient([_ok_response(reply=""), _ok_response(reply="hello")])
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)

    assert client.ask("hi") == "hello"
    assert len(fake.chat.completions.calls) == 2


def test_persistently_empty_reply_raises(monkeypatch):
    fake = FakeClient(_ok_response(reply=""))
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=2)

    with pytest.raises(LLMError):
        client.ask("hi")
    assert len(fake.chat.completions.calls) == 2
    assert client.history == [{"role": "system", "content": client.system_prompt}]


def test_empty_choices_is_retried(monkeypatch):
    fake = FakeClient([SimpleNamespace(choices=[]), _ok_response("hello")])
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=3)

    assert client.ask("hi") == "hello"
    assert len(fake.chat.completions.calls) == 2


def test_zero_max_retries_fails_immediately(monkeypatch):
    fake = FakeClient(_api_error(429))
    monkeypatch.setattr(llm_client.time, "sleep", mock.Mock())
    client = LLMClient(api_key="test-key", client=fake, max_retries=0)

    with pytest.raises(LLMError):
        client.ask("hi")
    assert len(fake.chat.completions.calls) == 0
    assert client.history == [{"role": "system", "content": client.system_prompt}]


def test_retry_after_is_capped():
    exc = _api_error(429, headers={"retry-after": "999999"})
    assert llm_client._retry_delay(1, exc) <= 30.0


def test_malformed_retry_after_falls_back_to_backoff():
    exc = _api_error(429, headers={"retry-after": "not-a-number"})
    delay = llm_client._retry_delay(1, exc)
    assert 0.5 <= delay <= 30.0


def test_sdk_builtin_retries_disabled(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return FakeClient(_ok_response("hello"))

    monkeypatch.setattr(llm_client.openai, "OpenAI", fake_openai)
    monkeypatch.setattr(llm_client, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    client = LLMClient()
    assert captured["max_retries"] == 0
    assert captured["api_key"] == "test-key"
    assert client.ask("hi") == "hello"


def test_retry_after_http_date_is_parsed():
    future = datetime.now(UTC) + timedelta(seconds=5)
    exc = _api_error(429, headers={"retry-after": format_datetime(future)})
    delay = llm_client._retry_delay(1, exc)
    assert 4.0 <= delay <= 5.0


def test_retry_after_invalid_falls_back_to_backoff():
    exc = _api_error(429, headers={"retry-after": "garbage"})
    delay = llm_client._retry_delay(1, exc)
    assert 0.5 <= delay <= 30.0


def test_default_timeout_read_from_env(monkeypatch):
    fake = FakeClient(_ok_response())
    monkeypatch.setenv("LLM_TIMEOUT", "45")
    client = LLMClient(api_key="test-key", client=fake, max_retries=1)
    client.ask("hi")
    assert fake.chat.completions.calls[0]["timeout"] == 45


def test_ask_override_timeout():
    fake = FakeClient(_ok_response())
    client = LLMClient(api_key="test-key", client=fake, max_retries=1)
    client.ask("hi", timeout=10)
    assert fake.chat.completions.calls[0]["timeout"] == 10
