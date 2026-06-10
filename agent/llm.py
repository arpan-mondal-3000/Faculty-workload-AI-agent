from __future__ import annotations

import logging
from functools import lru_cache

from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GROQ_MODEL = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
_TEMPERATURE   = getattr(settings, "LLM_TEMPERATURE",   0.0) 
_MAX_TOKENS    = getattr(settings, "LLM_MAX_TOKENS",     4096)
_REQUEST_TIMEOUT = getattr(settings, "LLM_TIMEOUT",      60)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    api_key = getattr(settings, "GROQ_API_KEY", None)
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )

    logger.info(
        "Loading Groq LLM — model: %s | temperature: %s | max_tokens: %s",
        _GROQ_MODEL, _TEMPERATURE, _MAX_TOKENS,
    )

    llm = ChatGroq(
        model=_GROQ_MODEL,
        api_key=api_key,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        timeout=_REQUEST_TIMEOUT,
        stop_sequences=["Observation:"],
    )

    logger.info("Groq LLM loaded successfully.")
    return llm


def get_llm_for_streaming() -> ChatGroq:
    api_key = getattr(settings, "GROQ_API_KEY", None)
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model=_GROQ_MODEL,
        api_key=api_key,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
        timeout=_REQUEST_TIMEOUT,
        streaming=True,
    )