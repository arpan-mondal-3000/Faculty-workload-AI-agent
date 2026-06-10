"""
Retrieval interface for the Faculty Timetable Agent.

This module exposes three retrievers — one per ChromaDB collection — plus a
unified ``MultiSourceRetriever`` that fans out across all three and merges
results by relevance score.

Retriever types
---------------
``get_policy_retriever()``
    Semantic search over ``policies.txt`` chunks.
    Use when the query is about rules, limits, or institutional guidelines.

``get_workload_retriever()``
    Semantic search over faculty workload rows.
    Use when the query asks about a specific professor's courses or load.

``get_timetable_retriever()``
    Semantic search over timetable slots.
    Use when the query asks about availability, room, day, or time.

``get_multi_source_retriever()``
    Searches all three collections and merges the top-k results.
    Use as the default RAG tool for open-ended queries.

All retrievers support optional ChromaDB metadata ``filter`` dicts so the
LangChain agent tools can narrow searches (e.g. filter by department or day).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag.embeddings import get_embedding_model, embed_query
from rag.ingest import ingest_all

logger = logging.getLogger(__name__)

# Module-level singletons — populated on first call to any getter
_policies_store:  Optional[Chroma] = None
_workload_store:  Optional[Chroma] = None
_timetable_store: Optional[Chroma] = None


# ---------------------------------------------------------------------------
# Internal: ensure stores are initialised
# ---------------------------------------------------------------------------

def _ensure_stores() -> None:
    """
    Lazily initialise the three ChromaDB vector stores.

    Calls ``ingest_all()`` which either loads existing on-disk collections or
    runs the full ingestion pipeline if they don't exist yet.
    """
    global _policies_store, _workload_store, _timetable_store

    if _policies_store is None:
        logger.info("Initialising ChromaDB vector stores …")
        _policies_store, _workload_store, _timetable_store = ingest_all()
        logger.info("Vector stores ready.")


# ---------------------------------------------------------------------------
# Per-collection retriever factories
# ---------------------------------------------------------------------------

def get_policy_retriever(
    k: int = 5,
    search_type: str = "similarity",
    filter: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    """
    Return a retriever scoped to the **policies** collection.

    Best for queries like:
    - "What is the maximum workload for a Professor?"
    - "Can a faculty teach more than 3 hours consecutively?"
    - "What are the rules for substitution?"

    Parameters
    ----------
    k : int
        Number of top documents to return. Default 5.
    search_type : str
        ``"similarity"`` (cosine) or ``"mmr"`` (maximal marginal relevance,
        reduces redundancy). Default ``"similarity"``.
    filter : dict, optional
        ChromaDB metadata filter, e.g. ``{"source": "policies"}``.

    Returns
    -------
    BaseRetriever
        LangChain-compatible retriever.

    Example
    -------
    >>> r = get_policy_retriever(k=3)
    >>> docs = r.invoke("max hours per week for professor")
    """
    _ensure_stores()
    search_kwargs: Dict[str, Any] = {"k": k}
    if filter:
        search_kwargs["filter"] = filter

    return _policies_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def get_workload_retriever(
    k: int = 5,
    search_type: str = "similarity",
    filter: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    """
    Return a retriever scoped to the **workload** collection.

    Best for queries like:
    - "What is Prof. Sharma's workload this week?"
    - "Which CSE faculty are teaching more than 10 hours?"
    - "List all HODs and their load."

    Parameters
    ----------
    k : int
        Number of top documents to return. Default 5.
    search_type : str
        ``"similarity"`` or ``"mmr"``. Default ``"similarity"``.
    filter : dict, optional
        Metadata filter, e.g. ``{"department": "CSE"}`` to restrict to one
        department, or ``{"faculty_id": "F101"}`` for a specific professor.

    Returns
    -------
    BaseRetriever

    Example
    -------
    >>> r = get_workload_retriever(filter={"department": "EEE"})
    >>> docs = r.invoke("faculty with highest load")
    """
    _ensure_stores()
    search_kwargs: Dict[str, Any] = {"k": k}
    if filter:
        search_kwargs["filter"] = filter

    return _workload_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def get_timetable_retriever(
    k: int = 5,
    search_type: str = "similarity",
    filter: Optional[Dict[str, Any]] = None,
) -> BaseRetriever:
    """
    Return a retriever scoped to the **timetable** collection.

    Best for queries like:
    - "Which faculty is free on Tuesday at 2 PM?"
    - "What is scheduled in Room 201 on Monday?"
    - "Show all lab sessions for CSE this week."

    Parameters
    ----------
    k : int
        Number of top documents to return. Default 5.
    search_type : str
        ``"similarity"`` or ``"mmr"``. Default ``"similarity"``.
    filter : dict, optional
        Metadata filter, e.g. ``{"day": "Tuesday"}`` or
        ``{"faculty_id": "F102", "day": "Wednesday"}``.

    Returns
    -------
    BaseRetriever

    Example
    -------
    >>> r = get_timetable_retriever(filter={"day": "Monday"})
    >>> docs = r.invoke("rooms occupied at 10 AM")
    """
    _ensure_stores()
    search_kwargs: Dict[str, Any] = {"k": k}
    if filter:
        search_kwargs["filter"] = filter

    return _timetable_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


# ---------------------------------------------------------------------------
# Multi-source retriever
# ---------------------------------------------------------------------------

class MultiSourceRetriever(BaseRetriever):
    """
    Fan-out retriever that queries all three collections in parallel and
    merges results, deduplicating by content.

    Parameters
    ----------
    k_per_source : int
        Number of results to fetch from *each* collection. The merged result
        set can have up to ``3 * k_per_source`` documents before dedup.
    search_type : str
        Passed through to each individual retriever.
    sources : list[str]
        Which collections to query. Defaults to all three:
        ``["policies", "workload", "timetable"]``.

    Usage
    -----
    >>> r = MultiSourceRetriever(k_per_source=3)
    >>> docs = r.invoke("Prof. Iyer schedule and workload")
    """

    k_per_source: int = 3
    search_type: str = "similarity"
    sources: List[str] = ["policies", "workload", "timetable"]

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        """
        Retrieve from each enabled collection and merge.

        Deduplication is content-based (exact string match). Results are
        ordered: workload first, timetable second, policies third — matching
        how the agent most commonly uses them.
        """
        _ensure_stores()

        source_map = {
            "policies":  _policies_store,
            "workload":  _workload_store,
            "timetable": _timetable_store,
        }

        all_docs: List[Document] = []
        seen_contents: set[str] = set()

        # Define retrieval order: workload → timetable → policies
        ordered_sources = [s for s in ["workload", "timetable", "policies"] if s in self.sources]

        for source in ordered_sources:
            store = source_map.get(source)
            if store is None:
                continue

            try:
                retriever = store.as_retriever(
                    search_type=self.search_type,
                    search_kwargs={"k": self.k_per_source},
                )
                docs = retriever.invoke(query)
                for doc in docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        all_docs.append(doc)
            except Exception as exc:
                logger.warning("Error retrieving from '%s': %s", source, exc)

        logger.debug(
            "MultiSourceRetriever: query=%r → %d merged docs", query, len(all_docs)
        )
        return all_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        """Async passthrough — delegates to the sync version."""
        return self._get_relevant_documents(query, run_manager=run_manager)


def get_multi_source_retriever(
    k_per_source: int = 4,
    search_type: str = "similarity",
    sources: Optional[List[str]] = None,
) -> MultiSourceRetriever:
    """
    Return a ``MultiSourceRetriever`` configured for all three collections.

    Parameters
    ----------
    k_per_source : int
        Documents to fetch from each collection. Default 4 (up to 12 total).
    search_type : str
        ``"similarity"`` or ``"mmr"``. Default ``"similarity"``.
    sources : list[str], optional
        Subset of ``["policies", "workload", "timetable"]`` to query.
        Defaults to all three.

    Returns
    -------
    MultiSourceRetriever

    Example
    -------
    >>> r = get_multi_source_retriever(k_per_source=3)
    >>> docs = r.invoke("Summarise CSE department workload")
    """
    _ensure_stores()

    return MultiSourceRetriever(
        k_per_source=k_per_source,
        search_type=search_type,
        sources=sources or ["policies", "workload", "timetable"],
    )


# ---------------------------------------------------------------------------
# Utility: similarity search with scores
# ---------------------------------------------------------------------------

def similarity_search_with_scores(
    query: str,
    collection: str = "timetable",
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[tuple[Document, float]]:
    """
    Run a raw similarity search and return ``(Document, score)`` pairs.

    Useful for debugging retrieval quality or building evaluation harnesses.

    Parameters
    ----------
    query : str
        The search query.
    collection : str
        One of ``"policies"``, ``"workload"``, ``"timetable"``.
    k : int
        Number of results to return.
    filter : dict, optional
        Optional ChromaDB metadata filter.

    Returns
    -------
    list[tuple[Document, float]]
        Sorted by descending similarity score.

    Example
    -------
    >>> results = similarity_search_with_scores(
    ...     "free slot Tuesday 2 PM", collection="timetable", k=5
    ... )
    >>> for doc, score in results:
    ...     print(f"{score:.3f}  {doc.page_content[:80]}")
    """
    _ensure_stores()

    store_map = {
        "policies":  _policies_store,
        "workload":  _workload_store,
        "timetable": _timetable_store,
    }
    store = store_map.get(collection)
    if store is None:
        raise ValueError(f"Unknown collection '{collection}'. Choose from {list(store_map)}")

    kwargs: Dict[str, Any] = {"k": k}
    if filter:
        kwargs["filter"] = filter

    return store.similarity_search_with_score(query, **kwargs)