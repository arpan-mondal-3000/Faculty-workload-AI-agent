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