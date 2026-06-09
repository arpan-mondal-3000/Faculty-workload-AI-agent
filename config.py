"""
config.py
---------
Centralised configuration for the Faculty Timetable Agent.

All values are read from environment variables (via a .env file loaded by
python-dotenv) with sensible defaults so the project runs out of the box for
development.

Usage
-----
    from config import settings

    print(settings.GROQ_API_KEY)
    print(settings.TIMETABLE_PATH)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (works whether you run from root or a subdir)
load_dotenv(Path(__file__).parent / ".env")

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent


# ── Data paths ───────────────────────────────────────────────────────────────
FACULTY_WORKLOAD_PATH = str(ROOT / "data" / "faculty_workload.csv")
TIMETABLE_PATH        = str(ROOT / "data" / "timetable.csv")
POLICIES_PATH         = str(ROOT / "data" / "policies.txt")


# ── Vector store ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(ROOT / "vectorstore" / "chroma_db")


# ── Embedding model (HuggingFace sentence-transformers) ──────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")   # "cuda" if GPU available


# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL",    "llama-3.1-8b-instant")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS",   "4096"))
LLM_TIMEOUT     = int(os.getenv("LLM_TIMEOUT",      "60"))


# ── HuggingFace (optional, for gated models) ─────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ── Settings proxy object ─────────────────────────────────────────────────────
# All other modules do `from config import settings` and access settings.X
# This avoids scattering `import config; config.GROQ_API_KEY` everywhere.

class _Settings:
    """Lightweight namespace that exposes every config value as an attribute."""

    # Data
    FACULTY_WORKLOAD_PATH: str = FACULTY_WORKLOAD_PATH
    TIMETABLE_PATH:        str = TIMETABLE_PATH
    POLICIES_PATH:         str = POLICIES_PATH

    # Vector store
    CHROMA_PERSIST_DIR: str = CHROMA_PERSIST_DIR

    # Embeddings
    EMBEDDING_MODEL:  str = EMBEDDING_MODEL
    EMBEDDING_DEVICE: str = EMBEDDING_DEVICE

    # LLM
    GROQ_API_KEY:    str   = GROQ_API_KEY
    GROQ_MODEL:      str   = GROQ_MODEL
    LLM_TEMPERATURE: float = LLM_TEMPERATURE
    LLM_MAX_TOKENS:  int   = LLM_MAX_TOKENS
    LLM_TIMEOUT:     int   = LLM_TIMEOUT

    # HuggingFace
    HF_TOKEN: str = HF_TOKEN

    # Logging
    LOG_LEVEL: str = LOG_LEVEL


settings = _Settings()


# ── Startup validation ────────────────────────────────────────────────────────

def validate() -> list[str]:
    """
    Check that all required configuration is present.

    Returns a list of error strings.  Empty list means config is valid.
    Called automatically by main.py before launching any subcommand.

    Example
    -------
    >>> from config import validate
    >>> errors = validate()
    >>> if errors:
    ...     for e in errors: print("CONFIG ERROR:", e)
    """
    errors: list[str] = []

    if not settings.GROQ_API_KEY:
        errors.append(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
        )

    for label, path in [
        ("FACULTY_WORKLOAD_PATH", settings.FACULTY_WORKLOAD_PATH),
        ("TIMETABLE_PATH",        settings.TIMETABLE_PATH),
        ("POLICIES_PATH",         settings.POLICIES_PATH),
    ]:
        if not Path(path).exists():
            errors.append(f"{label} not found: {path}")

    return errors