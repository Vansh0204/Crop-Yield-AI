"""Groq LLM client for optional crop-yield advisory text (Streamlit + free-tier Groq)."""

from typing import Optional

import streamlit as st

GROQ_MODEL = "llama3-8b-8192"

FALLBACK_ADVISORY_UNAVAILABLE = (
    "The AI advisory is unavailable right now. "
    "This usually means the Groq API key is not set in Streamlit secrets (GROQ_API_KEY), "
    "or the advisory service could not be reached. "
    "Your numeric yield prediction from the model is still valid."
)


def _load_groq_api_key() -> Optional[str]:
    """Return a non-empty API key from st.secrets, or None if missing or unreadable."""
    try:
        raw = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None
    if raw is None:
        return None
    key = str(raw).strip()
    return key if key else None


def get_llm_response(prompt: str) -> str:
    """
    Call Groq's chat API and return the assistant's text, or a fallback message on failure.

    The API key is read from ``st.secrets["GROQ_API_KEY"]`` for Streamlit deployment.
    """
    api_key = _load_groq_api_key()
    if not api_key:
        return FALLBACK_ADVISORY_UNAVAILABLE

    try:
        from groq import Groq
    except ImportError:
        return FALLBACK_ADVISORY_UNAVAILABLE

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = completion.choices[0].message
        text = choice.content if choice and choice.content is not None else ""
        text = text.strip()
        return text if text else FALLBACK_ADVISORY_UNAVAILABLE
    except Exception:
        return FALLBACK_ADVISORY_UNAVAILABLE
