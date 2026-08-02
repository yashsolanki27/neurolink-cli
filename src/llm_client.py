import logging
import os
import random
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
SYSTEM_PROMPT = (
    "You are NeuroLink, a helpful and honest assistant. "
    "Answer clearly and directly. If you are unsure or lack information, say so "
    "instead of guessing."
)

MAX_HISTORY_TOKENS = 3000
CHARS_PER_TOKEN = 4
MAX_BACKOFF = 30.0

RETRY_STATUS_CODES = (408, 425, 429, 500, 502, 503, 504)


class LLMError(RuntimeError):
    """Raised when the model cannot produce a reply."""


class EmptyReplyError(LLMError):
    """Internal signal that the model returned no usable content; retryable."""


def _retry_delay(
    attempt: int, exc: BaseException, max_backoff: float = MAX_BACKOFF
) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), max_backoff)
            except (TypeError, ValueError):
                pass
    return min(2**attempt, max_backoff) * (0.5 + random.random() * 0.5)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        client: openai.OpenAI | None = None,
        max_retries: int | None = None,
        max_history_tokens: int | None = None,
    ) -> None:
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")

        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
                "your OpenRouter API key (https://openrouter.ai/settings/keys)."
            )

        if client is None:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=resolved_key,
                max_retries=0,
            )
        else:
            self.client = client

        resolved_model = model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        assert resolved_model is not None
        self.model = resolved_model
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_retries = (
            int(os.getenv("LLM_MAX_RETRIES", "4")) if max_retries is None else max_retries
        )
        self.max_history_tokens = (
            int(os.getenv("LLM_MAX_HISTORY_TOKENS", str(MAX_HISTORY_TOKENS)))
            if max_history_tokens is None
            else max_history_tokens
        )
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset(self) -> None:
        self.history = [{"role": "system", "content": self.system_prompt}]

    def estimated_tokens(self) -> int:
        return sum(
            len(str(m.get("content", ""))) // CHARS_PER_TOKEN + 1 for m in self.history
        )

    def _trim_history(self) -> None:
        while len(self.history) > 2 and self.estimated_tokens() > self.max_history_tokens:
            del self.history[1:3]
        while len(self.history) > 2 and self.estimated_tokens() > self.max_history_tokens:
            del self.history[1]

    def ask(self, prompt: str, timeout: float = 30.0) -> str:
        if not prompt or not prompt.strip():
            raise LLMError("Empty prompt.")

        self.history.append({"role": "user", "content": prompt})
        self._trim_history()

        last_error: BaseException | None = None
        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.history,  # type: ignore[arg-type]  # dict is runtime-compatible with the SDK param types
                        timeout=timeout,
                    )
                    if not response.choices:
                        raise EmptyReplyError("Model returned no choices.")
                    reply = response.choices[0].message.content
                    if not reply:
                        raise EmptyReplyError("Model returned an empty response.")
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                except EmptyReplyError as exc:
                    last_error = self._backoff(attempt, exc)
                except LLMError:
                    raise
                except (
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.RateLimitError,
                ) as exc:
                    last_error = self._backoff(attempt, exc)
                except openai.APIStatusError as exc:
                    if exc.status_code in RETRY_STATUS_CODES:
                        last_error = self._backoff(attempt, exc)
                    else:
                        raise LLMError(f"API error {exc.status_code}: {exc}") from exc
        finally:
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()

        if last_error is None:
            detail = "No attempts were made."
        else:
            detail = f"Last error: {type(last_error).__name__}: {last_error}"
        raise LLMError(f"All {self.max_retries} attempts failed. {detail}")

    def _backoff(self, attempt: int, exc: BaseException) -> BaseException:
        delay = _retry_delay(attempt, exc)
        if attempt < self.max_retries:
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                self.max_retries,
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
        else:
            logger.warning(
                "Attempt %d/%d failed (%s); giving up",
                attempt,
                self.max_retries,
                type(exc).__name__,
            )
        return exc

    def __str__(self) -> str:
        return f"LLMClient(model={self.model})"
