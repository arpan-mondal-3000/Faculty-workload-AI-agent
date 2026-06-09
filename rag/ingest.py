"""
rag/ingest.py
-------------
Data ingestion pipeline for the Faculty Timetable Agent.

What this module does
---------------------
1. Loads ``policies.txt`` (unstructured text) and splits it into overlapping
   chunks suitable for semantic retrieval.
2. Loads ``faculty_workload.csv`` and ``timetable.csv``, converts each row into
   a human-readable natural-language sentence, and treats every sentence as a
   retrievable document.
3. Embeds all documents using the shared embedding model (all-MiniLM-L6-v2).
4. Stores the embedded documents in ChromaDB with metadata so that the
   retriever can filter by source (policies / workload / timetable).

Run this script once before starting the app, or whenever the source data
changes::

    python -m rag.ingest

Collections created in ChromaDB
---------------------------------
- ``policies``   : chunks from policies.txt
- ``workload``   : one document per faculty row
- ``timetable``  : one document per timetable slot
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import settings
from rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_policies(path: str | Path) -> List[Document]:
    """
    Read ``policies.txt``, split into overlapping chunks, and wrap each chunk
    in a LangChain ``Document`` with ``source`` metadata.

    Chunk strategy
    --------------
    * chunk_size  = 500 chars  → keeps policy items intact
    * chunk_overlap = 100 chars → preserves context across boundaries
    * Splits on section headers first (``\\n---``), then paragraphs, then
      sentences, so semantically related rules stay together.

    Parameters
    ----------
    path : str | Path
        Path to ``policies.txt``.

    Returns
    -------
    list[Document]
        Chunked policy documents ready for embedding.
    """
    logger.info("Loading policies from %s", path)
    raw_text = Path(path).read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n---", "\n\n", "\n", ". ", " "],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": "policies", "file": str(path)},
        )
        for chunk in chunks
    ]
    logger.info("Policies split into %d chunks.", len(docs))
    return docs


def _load_faculty_workload(path: str | Path) -> List[Document]:
    """
    Convert each row of ``faculty_workload.csv`` into a natural-language
    sentence so the LLM can reason about it in plain English.

    Each document is tagged with rich metadata (FacultyID, Department, etc.)
    so the retriever can apply metadata filters.

    Parameters
    ----------
    path : str | Path
        Path to ``faculty_workload.csv``.

    Returns
    -------
    list[Document]
        One document per faculty row.

    Example generated sentence
    --------------------------
    "Prof. Sharma (F101) is a Professor in the CSE department specialising in
    Algorithms & Data Structures. They teach Data Structures (CS201) for
    6 hours/week. Max allowed: 12 hrs. Total load (teaching + research +
    admin): 9 hrs. Status: Active."
    """
    logger.info("Loading faculty workload from %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    docs: List[Document] = []
    for _, row in df.iterrows():
        text = (
            f"{row['Name']} ({row['FacultyID']}) is a {row['Designation']} "
            f"in the {row['Department']} department specialising in "
            f"{row['Specialization']}. "
            f"They teach {row['Course']} ({row['CourseCode']}) for "
            f"{row['HoursPerWeek']} hours/week. "
            f"Max allowed: {row['MaxHoursAllowed']} hrs. "
            f"Research load: {row['ResearchLoad']} hrs, "
            f"Admin load: {row['AdminLoad']} hrs. "
            f"Total load: {row['TotalLoad']} hrs. "
            f"HOD: {row['IsHOD']}. "
            f"Years of experience: {row['YearsOfExperience']}. "
            f"Room preference: {row['RoomPreference']}. "
            f"Status: {row['Status']}."
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "workload",
                    "faculty_id": str(row["FacultyID"]),
                    "name": str(row["Name"]),
                    "department": str(row["Department"]),
                    "designation": str(row["Designation"]),
                    "course_code": str(row["CourseCode"]),
                    "hours_per_week": int(row["HoursPerWeek"]),
                    "is_hod": str(row["IsHOD"]),
                    "status": str(row["Status"]),
                },
            )
        )

    logger.info("Faculty workload: %d documents created.", len(docs))
    return docs


def _load_timetable(path: str | Path) -> List[Document]:
    """
    Convert each row of ``timetable.csv`` into a natural-language sentence.

    Each document captures: day, time window, course, faculty, room, section,
    class type, and enrollment — everything the agent needs to answer
    availability or clash queries.

    Parameters
    ----------
    path : str | Path
        Path to ``timetable.csv``.

    Returns
    -------
    list[Document]
        One document per timetable slot.

    Example generated sentence
    --------------------------
    "On Monday from 10:00 to 11:00, Prof. Nair (F105) teaches Operating
    Systems (CS301) to CSE Semester 5 Section A in Room 203. Class type:
    Lecture. Enrolled: 57 students. Status: Scheduled."
    """
    logger.info("Loading timetable from %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    docs: List[Document] = []
    for _, row in df.iterrows():
        text = (
            f"On {row['Day']} from {row['StartTime']} to {row['EndTime']}, "
            f"{row['Faculty']} ({row['FacultyID']}) teaches "
            f"{row['CourseName']} ({row['CourseCode']}) to "
            f"{row['Department']} Semester {row['Semester']} "
            f"Section {row['Section']} in {row['Room']}. "
            f"Class type: {row['ClassType']}. "
            f"Room capacity: {row['RoomCapacity']}. "
            f"Enrolled: {row['EnrolledStudents']} students. "
            f"Status: {row['Status']}."
        )
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "timetable",
                    "slot_id": str(row["SlotID"]),
                    "day": str(row["Day"]),
                    "start_time": str(row["StartTime"]),
                    "end_time": str(row["EndTime"]),
                    "faculty_id": str(row["FacultyID"]),
                    "faculty_name": str(row["Faculty"]),
                    "department": str(row["Department"]),
                    "course_code": str(row["CourseCode"]),
                    "room": str(row["Room"]),
                    "section": str(row["Section"]),
                    "semester": int(row["Semester"]),
                    "class_type": str(row["ClassType"]),
                    "status": str(row["Status"]),
                },
            )
        )

    logger.info("Timetable: %d documents created.", len(docs))
    return docs


def _build_or_load_collection(
    docs: List[Document],
    collection_name: str,
    persist_dir: str,
) -> Chroma:
    """
    Create a ChromaDB collection from ``docs`` or load it if it already exists.

    Parameters
    ----------
    docs : list[Document]
        Documents to embed and store.
    collection_name : str
        Name of the ChromaDB collection.
    persist_dir : str
        Directory where ChromaDB persists its on-disk store.

    Returns
    -------
    Chroma
        The populated (or loaded) vector store.
    """
    embedder = get_embedding_model()

    collection_path = Path(persist_dir) / collection_name

    if collection_path.exists() and any(collection_path.iterdir()):
        logger.info(
            "Collection '%s' already exists at %s — loading from disk.",
            collection_name,
            collection_path,
        )
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=str(collection_path),
        )

    logger.info(
        "Creating collection '%s' with %d documents …",
        collection_name,
        len(docs),
    )
    collection_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name=collection_name,
        persist_directory=str(collection_path),
    )

    logger.info("Collection '%s' created and persisted.", collection_name)
    return vectorstore


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_all(
    force_reingest: bool = False,
) -> Tuple[Chroma, Chroma, Chroma]:
    """
    Run the full ingestion pipeline and return the three ChromaDB collections.

    Parameters
    ----------
    force_reingest : bool
        If ``True``, delete existing collections and re-embed from scratch.
        Useful when source files change.

    Returns
    -------
    tuple[Chroma, Chroma, Chroma]
        ``(policies_store, workload_store, timetable_store)``

    Usage
    -----
    >>> policies_db, workload_db, timetable_db = ingest_all()
    """
    persist_dir = settings.CHROMA_PERSIST_DIR

    if force_reingest:
        import shutil
        logger.warning("force_reingest=True — wiping existing ChromaDB data at %s", persist_dir)
        shutil.rmtree(persist_dir, ignore_errors=True)

    # Load source data
    policy_docs   = _load_policies(settings.POLICIES_PATH)
    workload_docs = _load_faculty_workload(settings.FACULTY_WORKLOAD_PATH)
    timetable_docs = _load_timetable(settings.TIMETABLE_PATH)

    # Build / load collections
    policies_store  = _build_or_load_collection(policy_docs,   "policies",  persist_dir)
    workload_store  = _build_or_load_collection(workload_docs,  "workload",  persist_dir)
    timetable_store = _build_or_load_collection(timetable_docs, "timetable", persist_dir)

    logger.info("Ingestion complete.")
    return policies_store, workload_store, timetable_store


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest data into ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing collections and re-embed from scratch.",
    )
    args = parser.parse_args()

    ingest_all(force_reingest=args.force)