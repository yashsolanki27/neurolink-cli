import os
from typing import cast

import streamlit as st
from streamlit.errors import StreamlitAPIException

from llm_client import LLMClient, LLMError

st.set_page_config(page_title="NeuroLink", page_icon="🧠", layout="centered")

MODELS = {
    "NVIDIA Nemotron 3 Ultra 550B": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "NVIDIA Nemotron 3 Super 120B": "nvidia/nemotron-3-super-120b-a12b:free",
    "Google Gemma 4 26B": "google/gemma-4-26b-a4b-it:free",
}


def default_model_label() -> str:
    env_model = os.getenv("OPENROUTER_MODEL")
    for label, model_id in MODELS.items():
        if model_id == env_model:
            return label
    return next(iter(MODELS))


def resolve_api_key() -> str | None:
    try:
        key = st.secrets["OPENROUTER_API_KEY"]
    except (KeyError, StreamlitAPIException):
        return os.getenv("OPENROUTER_API_KEY")
    return key if isinstance(key, str) else None


def get_client() -> LLMClient | None:
    if "client" not in st.session_state:
        try:
            st.session_state.client = LLMClient(api_key=resolve_api_key())
        except LLMError as e:
            st.session_state.client = None
            st.session_state.setup_error = str(e)
    return cast(LLMClient | None, st.session_state.client)


st.title("NeuroLink")
st.caption("Free AI chat — your API key, your conversation, nothing saved by us.")

client = get_client()

if client is None:
    setup_error = getattr(
        st.session_state, "setup_error", "Failed to initialize the model client."
    )
    st.error(setup_error)
    st.info(
        "**Setup for local dev:** `cp .env.example .env` → add your key, then "
        "restart. **On Streamlit Cloud:** set `OPENROUTER_API_KEY` in app settings → Secrets."
    )
    st.stop()

history = client.history

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

options = list(MODELS)
model_label = st.sidebar.selectbox(
    "Model",
    options,
    index=options.index(default_model_label()),
    key="model_select",
)
client.model = MODELS[model_label]

st.sidebar.markdown(f"**Active:** `{client.model}`")
st.sidebar.markdown(f"**Context:** ~{client.estimated_tokens()} tokens")

if st.sidebar.button("Clear conversation", use_container_width=True):
    client.reset()
    st.session_state.pop("last_error", None)
    st.rerun()

for msg in history[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = client.ask(prompt)
                st.session_state.pop("last_error", None)
                st.write(reply)
            except LLMError as e:
                st.session_state["last_error"] = str(e)
    st.rerun()
