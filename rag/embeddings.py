"""
rag/embeddings.py
-----------------
Centralised embedding model wrapper for the Faculty Timetable Agent.

Responsibilities
----------------
* Load the sentence-transformers model once and expose a reusable instance.
* Wrap the model in LangChain's HuggingFaceEmbeddings so it integrates
  seamlessly with ChromaDB and any LangChain retrieval chain.
* Provide a lightweight utility to embed arbitrary text on demand (useful for
  query-time embedding outside of a retrieval chain).

Model
-----
sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional dense vectors
  - Fast inference, low memory footprint
  - Strong performance on semantic similarity tasks
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*HuggingFaceEmbeddings.*deprecated.*",
    category=DeprecationWarning,
)
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory – cached so the heavy model is loaded only once per process
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return a singleton HuggingFaceEmbeddings instance.

    The model is loaded from the local HuggingFace cache on first call and
    reused on every subsequent call thanks to ``lru_cache``.

    Returns
    -------
    HuggingFaceEmbeddings
        Ready-to-use LangChain embedding object.

    Example
    -------
    >>> embedder = get_embedding_model()
    >>> vec = embedder.embed_query("Prof. Sharma teaches Data Structures")
    >>> len(vec)
    384
    """
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)

    model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},  # "cpu" or "cuda"
        encode_kwargs={
            "normalize_embeddings": True,   # cosine similarity works best normalised
            "batch_size": 64,               # tune up/down based on available RAM
        },
    )

    logger.info("Embedding model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings and return their dense vectors.

    Parameters
    ----------
    texts : list[str]
        Raw text strings to embed.

    Returns
    -------
    list[list[float]]
        One 384-dim vector per input string.

    Example
    -------
    >>> vecs = embed_texts(["Data Structures", "Fluid Mechanics"])
    >>> len(vecs), len(vecs[0])
    (2, 384)
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Slightly different from ``embed_texts`` — uses the model's
    query-specific encoding path which can improve retrieval quality.

    Parameters
    ----------
    query : str
        The user's natural-language question.

    Returns
    -------
    list[float]
        A single 384-dim vector.

    Example
    -------
    >>> vec = embed_query("Which faculty is free on Tuesday at 2 PM?")
    >>> len(vec)
    384
    """
    model = get_embedding_model()
    return model.embed_query(query)