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
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)

    model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        },
    )

    logger.info("Embedding model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    model = get_embedding_model()
    return model.embed_query(query)