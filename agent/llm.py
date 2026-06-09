"""
agent/llm.py
------------
LLM loader for the Faculty Timetable Agent.

Loads ``llama-3.1-8b-instant`` from Groq via LangChain's ChatGroq wrapper.
The model is instantiated once per process (``@lru_cache``) and shared across
all agent invocations to avoid redundant API round-trips.

Configuration is read from ``config.settings`` so that API keys and model
parameters stay out of source code.

Usage
-----
>>> from agent.llm import get_llm
>>> llm = get_llm()
>>> llm.invoke("Hello!")
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Groq model identifier — llama-3.1-8b-instant gives the best speed/quality
# balance for structured academic-scheduling tasks at Groq's free tier.
# _GROQ_MODEL = getattr(settings, "GROQ_MODEL", "llama-3.1-8b-instant")
_GROQ_MODEL = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")

# Safe defaults — override via settings if needed
_TEMPERATURE   = getattr(settings, "LLM_TEMPERATURE",   0.0)   # deterministic for scheduling
_MAX_TOKENS    = getattr(settings, "LLM_MAX_TOKENS",     4096)
_REQUEST_TIMEOUT = getattr(settings, "LLM_TIMEOUT",      60)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Return a singleton ``ChatGroq`` instance configured for
    ``llama-3.1-8b-instant``.

    The model is loaded once and cached for the lifetime of the process.
    Temperature is set to 0 for deterministic, fact-grounded responses —
    critical for scheduling and workload queries where hallucinated numbers
    are harmful.

    Returns
    -------
    ChatGroq
        Ready-to-use LangChain chat model.

    Raises
    ------
    ValueError
        If ``settings.GROQ_API_KEY`` is missing or empty.

    Example
    -------
    >>> llm = get_llm()
    >>> response = llm.invoke("What is 2 + 2?")
    >>> print(response.content)
    """
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
        # Stop sequences help the ReAct agent cleanly end its Thought/Action loop
        stop_sequences=["Observation:"],
    )

    logger.info("Groq LLM loaded successfully.")
    return llm


def get_llm_for_streaming() -> ChatGroq:
    """
    Return a **non-cached** ``ChatGroq`` instance with streaming enabled.

    Use this variant in the Streamlit frontend when you want to display
    tokens as they arrive (``st.write_stream``).  A fresh object is returned
    each time because ``streaming=True`` is stateful during a generation.

    Returns
    -------
    ChatGroq
        Streaming-enabled chat model (not cached).

    Example
    -------
    >>> llm = get_llm_for_streaming()
    >>> for chunk in llm.stream("Tell me about Prof. Sharma"):
    ...     print(chunk.content, end="", flush=True)
    """
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