"""
utils/csv_loader.py
-------------------
Centralised, cached CSV loading utilities for the Faculty Timetable Agent.

Responsibilities
----------------
* Load ``faculty_workload.csv`` and ``timetable.csv`` into pandas DataFrames
  exactly once per process (via ``@lru_cache`` on the underlying read).
* Normalise column names, data types, and time fields so every downstream
  utility receives clean, consistent data.
* Expose lightweight re-load helpers so tests or the Streamlit app can
  invalidate the cache when source files change.

All public functions return copies of the cached DataFrame so callers can
mutate freely without corrupting the shared cache.
"""

from __future__ import annotations

import logging
from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal cached readers  (private — use public wrappers below)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _read_faculty_workload(path: str) -> pd.DataFrame:
    """
    Read and normalise ``faculty_workload.csv``.

    Normalisation steps
    -------------------
    * Strip whitespace from all column names and string values.
    * Coerce numeric columns to the correct dtypes.
    * Add a ``IsHOD_bool`` boolean column for easy filtering.
    * Add ``LoadUtilisationPct`` = TotalLoad / MaxHoursAllowed * 100.

    Parameters
    ----------
    path : str
        Absolute path to ``faculty_workload.csv``.

    Returns
    -------
    pd.DataFrame
        Normalised workload DataFrame.
    """
    logger.info("Reading faculty workload CSV from %s", path)
    df = pd.read_csv(path)

    # --- Clean column names & string values ---
    df.columns = df.columns.str.strip()
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # --- Coerce numeric columns ---
    int_cols = [
        "HoursPerWeek", "MaxHoursAllowed", "YearsOfExperience",
        "ResearchLoad", "AdminLoad", "TotalLoad",
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # --- Derived columns ---
    df["IsHOD_bool"] = df["IsHOD"].str.upper() == "YES"
    df["LoadUtilisationPct"] = (
        df["TotalLoad"] / df["MaxHoursAllowed"] * 100
    ).round(1)
    df["RemainingCapacity"] = df["MaxHoursAllowed"] - df["TotalLoad"]

    logger.info("Faculty workload loaded: %d rows.", len(df))
    return df


@lru_cache(maxsize=1)
def _read_timetable(path: str) -> pd.DataFrame:
    """
    Read and normalise ``timetable.csv``.

    Normalisation steps
    -------------------
    * Strip whitespace from all column names and string values.
    * Parse ``StartTime`` / ``EndTime`` strings into ``datetime.time`` objects
      and also keep them as ``timedelta`` minutes-since-midnight for arithmetic.
    * Add ``DurationMinutes`` column.
    * Add a canonical ``DayOrder`` column (0 = Monday … 5 = Saturday) so
      DataFrames can be sorted chronologically.

    Parameters
    ----------
    path : str
        Absolute path to ``timetable.csv``.

    Returns
    -------
    pd.DataFrame
        Normalised timetable DataFrame.
    """
    logger.info("Reading timetable CSV from %s", path)
    df = pd.read_csv(path)

    # --- Clean column names & string values ---
    df.columns = df.columns.str.strip()
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # --- Numeric coercions ---
    df["Semester"]        = pd.to_numeric(df["Semester"],        errors="coerce").astype("Int64")
    df["RoomCapacity"]    = pd.to_numeric(df["RoomCapacity"],    errors="coerce").astype("Int64")
    df["EnrolledStudents"]= pd.to_numeric(df["EnrolledStudents"],errors="coerce").astype("Int64")

    # --- Time parsing ---
    def _to_time(t_str: str) -> time:
        """Parse 'HH:MM' → datetime.time."""
        h, m = map(int, str(t_str).strip().split(":"))
        return time(h, m)

    def _to_minutes(t_str: str) -> int:
        """Parse 'HH:MM' → minutes since midnight (for arithmetic)."""
        h, m = map(int, str(t_str).strip().split(":"))
        return h * 60 + m

    df["StartTimeParsed"]   = df["StartTime"].apply(_to_time)
    df["EndTimeParsed"]     = df["EndTime"].apply(_to_time)
    df["StartMinutes"]      = df["StartTime"].apply(_to_minutes)
    df["EndMinutes"]        = df["EndTime"].apply(_to_minutes)
    df["DurationMinutes"]   = df["EndMinutes"] - df["StartMinutes"]

    # --- Day ordering ---
    day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                 "Thursday": 3, "Friday": 4, "Saturday": 5}
    df["DayOrder"] = df["Day"].map(day_order).fillna(99).astype(int)

    logger.info("Timetable loaded: %d rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Public getters  (always return a copy so callers can mutate freely)
# ---------------------------------------------------------------------------

def load_faculty_workload(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return a clean copy of the faculty workload DataFrame.

    Parameters
    ----------
    path : str, optional
        Override the default path from ``settings.FACULTY_WORKLOAD_PATH``.
        Useful in tests.

    Returns
    -------
    pd.DataFrame
        Columns include: FacultyID, Name, Department, Designation, Course,
        CourseCode, HoursPerWeek, MaxHoursAllowed, TotalLoad,
        LoadUtilisationPct, RemainingCapacity, IsHOD_bool, …

    Example
    -------
    >>> df = load_faculty_workload()
    >>> df[df["Department"] == "CSE"][["Name", "TotalLoad"]]
    """
    resolved = str(path or settings.FACULTY_WORKLOAD_PATH)
    return _read_faculty_workload(resolved).copy()


def load_timetable(path: Optional[str] = None) -> pd.DataFrame:
    """
    Return a clean copy of the timetable DataFrame.

    Parameters
    ----------
    path : str, optional
        Override the default path from ``settings.TIMETABLE_PATH``.

    Returns
    -------
    pd.DataFrame
        Columns include: SlotID, Day, StartTime, EndTime, StartMinutes,
        EndMinutes, DurationMinutes, DayOrder, CourseCode, CourseName,
        FacultyID, Faculty, Department, Semester, Section, Room,
        RoomCapacity, ClassType, EnrolledStudents, Status, …

    Example
    -------
    >>> df = load_timetable()
    >>> df[df["Day"] == "Monday"].sort_values("StartMinutes")
    """
    resolved = str(path or settings.TIMETABLE_PATH)
    return _read_timetable(resolved).copy()


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def reload_all() -> None:
    """
    Invalidate both caches and force a fresh read on the next access.

    Call this from the Streamlit app whenever the admin uploads new CSV files.

    Example
    -------
    >>> reload_all()
    >>> df = load_timetable()   # reads fresh from disk
    """
    _read_faculty_workload.cache_clear()
    _read_timetable.cache_clear()
    logger.info("CSV caches cleared — next load will read from disk.")


# ---------------------------------------------------------------------------
# Validation helpers (called at startup or after re-upload)
# ---------------------------------------------------------------------------

def validate_csvs() -> dict[str, list[str]]:
    """
    Check that both CSVs have the required columns and return any issues found.

    Returns
    -------
    dict[str, list[str]]
        ``{"faculty_workload": [...errors], "timetable": [...errors]}``
        Empty lists mean the file is valid.

    Example
    -------
    >>> issues = validate_csvs()
    >>> if any(issues.values()):
    ...     print(issues)
    """
    required_workload_cols = {
        "FacultyID", "Name", "Department", "Designation", "Course",
        "CourseCode", "HoursPerWeek", "MaxHoursAllowed", "TotalLoad",
        "IsHOD", "Status",
    }
    required_timetable_cols = {
        "SlotID", "Day", "StartTime", "EndTime", "CourseCode", "CourseName",
        "FacultyID", "Faculty", "Department", "Semester", "Section",
        "Room", "ClassType", "Status",
    }

    errors: dict[str, list[str]] = {"faculty_workload": [], "timetable": []}

    try:
        wdf = load_faculty_workload()
        missing = required_workload_cols - set(wdf.columns)
        if missing:
            errors["faculty_workload"].append(f"Missing columns: {sorted(missing)}")
        if wdf.empty:
            errors["faculty_workload"].append("File is empty.")
    except Exception as exc:
        errors["faculty_workload"].append(f"Load error: {exc}")

    try:
        tdf = load_timetable()
        missing = required_timetable_cols - set(tdf.columns)
        if missing:
            errors["timetable"].append(f"Missing columns: {sorted(missing)}")
        if tdf.empty:
            errors["timetable"].append("File is empty.")
    except Exception as exc:
        errors["timetable"].append(f"Load error: {exc}")

    return errors