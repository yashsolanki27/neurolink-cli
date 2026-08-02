# NeuroLink

Free AI chat you can trust — bring your own OpenRouter API key. Web GUI only, no CLI.
Requests go straight from your app to the model provider; NeuroLink stores nothing.

**Scope:** current build is a hardened chat core (Streamlit). The next milestone turns
NeuroLink into a **document-grounded knowledge assistant**: upload PDF/TXT, get answers
that cite the exact chunks they came from.

## Architecture

```
src/
  app.py          Streamlit GUI — renders chat, owns session state, surfaces errors
  llm_client.py   LLMClient — talks to OpenRouter, manages history + retries
tests/
  test_llm_client.py, test_app.py
```

- **`LLMClient`** is the single conversation owner. It keeps the message list
  (system prompt + user/assistant turns), trims old turns when the rough token budget
  (`LLM_MAX_HISTORY_TOKENS`) is exceeded, and calls the model.
- **Error handling is typed, not blanket.** `ask()` catches `openai.APITimeoutError`,
  `openai.APIConnectionError`, `openai.RateLimitError`, and   `openai.APIStatusError`
  separately. Timeouts, connection failures, rate limits (429), transient 5xx, and
  empty replies are retried with exponential backoff (honoring the server's
  `Retry-After` header, capped at 30s); everything else (4xx, bad keys) fails fast as
  `LLMError`. A failed `ask()` rolls back its pending user message, so history never
  gets a dangling turn. All failures surface in the GUI as a readable message, never a
  raw traceback.
- **`app.py`** keeps the LLM client in `st.session_state` (one per browser session, never
  cached globally), so each user gets an isolated conversation.
- **The API key never enters the codebase.** It is read from environment
  (`OPENROUTER_API_KEY`) or Streamlit Cloud secrets. Only `.env.example` is committed.

## Quickstart (local)

```bash
uv sync
cp .env.example .env        # add your OPENROUTER_API_KEY
uv run streamlit run src/app.py
```

Get a free key at <https://openrouter.ai/settings/keys>. The default model is a free-tier
OpenRouter model, so this costs nothing to run.

### Configuration (`.env`)
| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required. API key from OpenRouter |
| `OPENROUTER_MODEL` | `google/gemma-4-26b-a4b-it:free` | Model ID served by OpenRouter |
| `LLM_MAX_RETRIES` | `4` | Retry attempts before surfacing an error |
| `LLM_MAX_HISTORY_TOKENS` | `3000` | Rough context cap; oldest messages are trimmed |

## Deploy to Streamlit Cloud (free)
1. Push this repo to GitHub.
2. On <https://streamlit.io/cloud> → "New app" → pick the repo.
3. Main file: `src/app.py`.
4. In **Advanced settings**, set **Python version** to `3.13`. Cloud ignores
   `.python-version` and the default drifts (3.14+ in 2026), which can stall
   `uv sync` mid-deploy. Then click **Save**.
5. Settings → Secrets: add `OPENROUTER_API_KEY=...` (same line format).
6. Deploy. Your live URL is generated automatically.

Dependencies are managed by `uv` (`pyproject.toml` + `uv.lock`); Streamlit Cloud
installs them via `uv sync` from `uv.lock`, so keep it in sync with
`uv lock`/`uv add` changes.

## Development
```bash
uv run ruff check src tests
uv run mypy
uv run pytest
```

## License
TBD
