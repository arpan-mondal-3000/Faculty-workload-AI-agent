"""
utils/timetable_utils.py
------------------------
Timetable query and analysis utilities for the Faculty Timetable Agent.

All functions operate on the normalised timetable DataFrame returned by
``csv_loader.load_timetable()``.  They are pure query functions — they never
modify the source DataFrame or the CSV files.

Public API
----------
``get_faculty_schedule(faculty_name_or_id, day)``
    Return all sessions for a given faculty, optionally filtered by day.

``get_free_faculty_at(day, time_str)``
    Return faculty members with no class at the given day + time slot.

``get_room_schedule(room, day)``
    Return all sessions scheduled in a specific room, optionally on one day.

``detect_faculty_clashes()``
    Scan for any faculty assigned to two sessions at the same time.

``detect_room_clashes()``
    Scan for any room double-booked at the same time.

``detect_consecutive_overloads()``
    Flag faculty with more than 3 back-to-back teaching hours in one day.

``get_daily_schedule(day)``
    Full sorted schedule for a given day across all departments.

``get_section_schedule(department, semester, section)``
    Weekly schedule for a specific student section.

``suggest_free_slots(faculty_name_or_id)``
    Return day + time slots where a faculty member has no class.
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Optional

import pandas as pd

from utils.csv_loader import load_faculty_workload, load_timetable

logger = logging.getLogger(__name__)

# Standard teaching slots used for free-slot analysis
_ALL_SLOTS = [
    ("08:00", "09:00"), ("09:00", "10:00"), ("10:00", "11:00"),
    ("11:00", "12:00"), ("14:00", "15:00"), ("15:00", "16:00"),
    ("16:00", "17:00"),
]
_ALL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_faculty_id(query: str, tt: pd.DataFrame) -> list[str]:
    """
    Resolve a name fragment or FacultyID to a list of matching FacultyIDs.

    Matching is case-insensitive and supports partial name matches so
    "sharma" matches "Prof. Sharma".

    Parameters
    ----------
    query : str
        Faculty name (partial or full) or FacultyID (e.g. "F101").
    tt : pd.DataFrame
        Timetable DataFrame.

    Returns
    -------
    list[str]
        Matching FacultyID values. Empty list if nothing found.
    """
    q = query.strip().upper()

    # Exact FacultyID match first
    if q in tt["FacultyID"].str.upper().values:
        return [q]

    # Partial name match (case-insensitive)
    mask = tt["Faculty"].str.upper().str.contains(q, na=False)
    return tt.loc[mask, "FacultyID"].unique().tolist()


def _parse_time_str(t_str: str) -> int:
    """Convert 'HH:MM' to minutes-since-midnight."""
    h, m = map(int, t_str.strip().split(":"))
    return h * 60 + m


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """True when two [start, end) intervals in minutes-since-midnight overlap."""
    return start_a < end_b and end_a > start_b


# ---------------------------------------------------------------------------
# Schedule queries
# ---------------------------------------------------------------------------

def get_faculty_schedule(
    faculty: str,
    day: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return all timetable slots for a given faculty member.

    Parameters
    ----------
    faculty : str
        Faculty name (partial or full) or FacultyID.
    day : str, optional
        Day filter, e.g. ``"Monday"``. Case-insensitive. If omitted, all days
        are returned sorted by day order then start time.

    Returns
    -------
    pd.DataFrame
        Subset of timetable rows for that faculty, with columns:
        Day, StartTime, EndTime, CourseName, CourseCode, Section, Room,
        ClassType, EnrolledStudents.

    Raises
    ------
    ValueError
        If no faculty matching the query is found.

    Example
    -------
    >>> df = get_faculty_schedule("Prof. Sharma")
    >>> df = get_faculty_schedule("F101", day="Monday")
    """
    tt = load_timetable()
    ids = _resolve_faculty_id(faculty, tt)
    if not ids:
        raise ValueError(f"No faculty found matching '{faculty}'.")

    mask = tt["FacultyID"].isin(ids)

    if day:
        mask &= tt["Day"].str.upper() == day.upper()

    cols = [
        "Day", "DayOrder", "StartTime", "EndTime", "DurationMinutes",
        "CourseName", "CourseCode", "Section",
        "Room", "ClassType", "Faculty",
    ]
    return (
        tt.loc[mask, cols]
        .sort_values(["DayOrder", "StartTime"])
        .reset_index(drop=True)
    )


def get_daily_schedule(day: str) -> pd.DataFrame:
    """
    Return the full sorted schedule for a given day across all departments.

    Parameters
    ----------
    day : str
        Day name, e.g. ``"Tuesday"``. Case-insensitive.

    Returns
    -------
    pd.DataFrame
        All sessions on that day, sorted by start time then department.

    Example
    -------
    >>> df = get_daily_schedule("Wednesday")
    """
    tt = load_timetable()
    mask = tt["Day"].str.upper() == day.upper()
    cols = [
        "StartTime", "EndTime", "Faculty",
        "CourseName", "Section", "Room", "Department",
    ]
    return (
        tt.loc[mask, cols]
        .sort_values(["StartMinutes", "Department"])
        .reset_index(drop=True)
    )


def get_room_schedule(room: str, day: Optional[str] = None) -> pd.DataFrame:
    """
    Return all sessions scheduled in a specific room.

    Parameters
    ----------
    room : str
        Room name, e.g. ``"Room 201"`` or ``"Lab 101"``. Case-insensitive
        partial match supported.
    day : str, optional
        Filter by day. If omitted, returns the full week.

    Returns
    -------
    pd.DataFrame
        Sessions in that room sorted by day then start time.

    Example
    -------
    >>> df = get_room_schedule("Room 201")
    >>> df = get_room_schedule("Lab 101", day="Thursday")
    """
    tt = load_timetable()
    mask = tt["Room"].str.upper().str.contains(room.upper(), na=False)
    if day:
        mask &= tt["Day"].str.upper() == day.upper()

    cols = [
        "Day", "DayOrder", "StartTime", "EndTime",
        "Faculty", "CourseName", "Section",
    ]
    return (
        tt.loc[mask, cols]
        .sort_values(["DayOrder", "StartMinutes"])
        .reset_index(drop=True)
    )


def get_section_schedule(
    department: str,
    semester: int,
    section: str,
) -> pd.DataFrame:
    """
    Return the full weekly schedule for a specific student section.

    Parameters
    ----------
    department : str
        Department code, e.g. ``"CSE"``. Case-insensitive.
    semester : int
        Semester number, e.g. ``5``.
    section : str
        Section label, e.g. ``"A"``. Case-insensitive.

    Returns
    -------
    pd.DataFrame
        Weekly timetable for that section, sorted by day then start time.

    Example
    -------
    >>> df = get_section_schedule("CSE", 5, "A")
    """
    tt = load_timetable()
    mask = (
        (tt["Department"].str.upper() == department.upper()) &
        (tt["Semester"]  == semester) &
        (tt["Section"].str.upper() == section.upper())
    )
    cols = [
        "Day", "DayOrder", "StartTime", "EndTime",
        "CourseName", "Faculty", "Room",
    ]
    return (
        tt.loc[mask, cols]
        .sort_values(["DayOrder", "StartMinutes"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Availability queries
# ---------------------------------------------------------------------------

def get_free_faculty_at(day: str, time_str: str) -> pd.DataFrame:
    """
    Return all faculty members who have NO session at the given day + time.

    Parameters
    ----------
    day : str
        Day name, e.g. ``"Tuesday"``.
    time_str : str
        Time as ``"HH:MM"``, e.g. ``"14:00"``.

    Returns
    -------
    pd.DataFrame
        Faculty (from workload CSV) who are free, with columns:
        FacultyID, Name, Department, Designation, Course.

    Example
    -------
    >>> free = get_free_faculty_at("Tuesday", "14:00")
    """
    tt  = load_timetable()
    wdf = load_faculty_workload()

    query_min = _parse_time_str(time_str)
    day_upper = day.upper()

    # Sessions on that day that overlap with the queried minute
    occupied_mask = (
        (tt["Day"].str.upper() == day_upper) &
        (tt["StartMinutes"] <= query_min) &
        (tt["EndMinutes"]   >  query_min)
    )
    occupied_ids = set(tt.loc[occupied_mask, "FacultyID"].unique())

    all_ids = set(wdf["FacultyID"].unique())
    free_ids = all_ids - occupied_ids

    free_df = wdf.loc[
        wdf["FacultyID"].isin(free_ids),
        ["FacultyID", "Name", "Department", "Designation", "Course", "Status"],
    ].sort_values(["Department", "Name"]).reset_index(drop=True)

    logger.debug(
        "Free faculty on %s at %s: %d/%d",
        day, time_str, len(free_df), len(all_ids),
    )
    return free_df


def suggest_free_slots(faculty: str) -> pd.DataFrame:
    """
    Return every day + time slot where a faculty member has no class.

    Parameters
    ----------
    faculty : str
        Faculty name (partial or full) or FacultyID.

    Returns
    -------
    pd.DataFrame
        Rows with columns: Day, SlotStart, SlotEnd — where the faculty is free.

    Raises
    ------
    ValueError
        If no faculty matching the query is found.

    Example
    -------
    >>> slots = suggest_free_slots("Prof. Mehta")
    """
    tt = load_timetable()
    ids = _resolve_faculty_id(faculty, tt)
    if not ids:
        raise ValueError(f"No faculty found matching '{faculty}'.")

    busy = tt[tt["FacultyID"].isin(ids)][
        ["Day", "StartMinutes", "EndMinutes"]
    ]

    day_order_map = {d: i for i, d in enumerate(_ALL_DAYS)}
    free_rows = []

    for day in _ALL_DAYS:
        busy_day = busy[busy["Day"] == day]
        for slot_start, slot_end in _ALL_SLOTS:
            s = _parse_time_str(slot_start)
            e = _parse_time_str(slot_end)
            clash = any(
                _overlaps(s, e, row["StartMinutes"], row["EndMinutes"])
                for _, row in busy_day.iterrows()
            )
            if not clash:
                free_rows.append({
                    "Day": day,
                    "DayOrder": day_order_map[day],
                    "SlotStart": slot_start,
                    "SlotEnd": slot_end,
                })

    free_df = (
        pd.DataFrame(free_rows)
        .sort_values(["DayOrder", "SlotStart"])
        .drop(columns="DayOrder")
        .reset_index(drop=True)
    )
    return free_df


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def detect_faculty_clashes() -> pd.DataFrame:
    """
    Detect any faculty member assigned to two overlapping sessions.

    Returns
    -------
    pd.DataFrame
        One row per clash with columns:
        FacultyID, Faculty, Day, Slot1 (SlotID), Slot2 (SlotID),
        Time1, Time2, Course1, Course2.
        Empty DataFrame means no clashes.

    Example
    -------
    >>> clashes = detect_faculty_clashes()
    >>> if clashes.empty:
    ...     print("No faculty clashes detected.")
    """
    tt = load_timetable()
    clashes = []

    for (fid, day), group in tt.groupby(["FacultyID", "Day"]):
        slots = group.reset_index(drop=True)
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots.iloc[i], slots.iloc[j]
                if _overlaps(a["StartMinutes"], a["EndMinutes"],
                             b["StartMinutes"], b["EndMinutes"]):
                    clashes.append({
                        "FacultyID":  fid,
                        "Faculty":    a["Faculty"],
                        "Day":        day,
                        "Slot1":      a["SlotID"],
                        "Time1":      f"{a['StartTime']}–{a['EndTime']}",
                        "Course1":    a["CourseName"],
                        "Slot2":      b["SlotID"],
                        "Time2":      f"{b['StartTime']}–{b['EndTime']}",
                        "Course2":    b["CourseName"],
                    })

    if clashes:
        logger.warning("%d faculty clash(es) detected.", len(clashes))
    return pd.DataFrame(clashes)


def detect_room_clashes() -> pd.DataFrame:
    """
    Detect any room double-booked at overlapping times on the same day.

    Returns
    -------
    pd.DataFrame
        One row per clash with columns:
        Room, Day, Slot1, Time1, Faculty1, Course1,
        Slot2, Time2, Faculty2, Course2.
        Empty DataFrame means no clashes.

    Example
    -------
    >>> clashes = detect_room_clashes()
    """
    tt = load_timetable()
    clashes = []

    for (room, day), group in tt.groupby(["Room", "Day"]):
        slots = group.reset_index(drop=True)
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots.iloc[i], slots.iloc[j]
                if _overlaps(a["StartMinutes"], a["EndMinutes"],
                             b["StartMinutes"], b["EndMinutes"]):
                    clashes.append({
                        "Room":     room,
                        "Day":      day,
                        "Slot1":    a["SlotID"],
                        "Time1":    f"{a['StartTime']}–{a['EndTime']}",
                        "Faculty1": a["Faculty"],
                        "Course1":  a["CourseName"],
                        "Slot2":    b["SlotID"],
                        "Time2":    f"{b['StartTime']}–{b['EndTime']}",
                        "Faculty2": b["Faculty"],
                        "Course2":  b["CourseName"],
                    })

    if clashes:
        logger.warning("%d room clash(es) detected.", len(clashes))
    return pd.DataFrame(clashes)


def detect_consecutive_overloads() -> pd.DataFrame:
    """
    Flag faculty who have more than 3 consecutive teaching hours on any day,
    violating the university policy.

    Algorithm
    ---------
    For each (faculty, day) group, sort sessions by start time and compute
    the longest uninterrupted teaching run by checking whether consecutive
    slots are back-to-back (gap == 0 minutes).

    Returns
    -------
    pd.DataFrame
        Columns: FacultyID, Faculty, Day, ConsecutiveHours, Sessions.
        Only rows where ConsecutiveHours > 3 are returned.

    Example
    -------
    >>> overloads = detect_consecutive_overloads()
    """
    tt = load_timetable()
    results = []

    for (fid, day), group in tt.groupby(["FacultyID", "Day"]):
        slots = group.sort_values("StartMinutes").reset_index(drop=True)
        if len(slots) < 2:
            continue

        # Walk through sorted slots tracking the current run
        run_start_idx = 0
        for i in range(1, len(slots)):
            gap = slots.loc[i, "StartMinutes"] - slots.loc[i - 1, "EndMinutes"]
            if gap > 0:
                # Break in teaching — evaluate the run that just ended
                run = slots.iloc[run_start_idx:i]
                total = run["DurationMinutes"].sum()
                if total > 180:  # > 3 hours
                    results.append({
                        "FacultyID":        fid,
                        "Faculty":          slots.loc[0, "Faculty"],
                        "Day":              day,
                        "ConsecutiveHours": round(total / 60, 1),
                        "Sessions":         ", ".join(
                            f"{r['StartTime']}–{r['EndTime']} {r['CourseName']}"
                            for _, r in run.iterrows()
                        ),
                    })
                run_start_idx = i

        # Check final run
        run = slots.iloc[run_start_idx:]
        total = run["DurationMinutes"].sum()
        if total > 180:
            results.append({
                "FacultyID":        fid,
                "Faculty":          slots.loc[0, "Faculty"],
                "Day":              day,
                "ConsecutiveHours": round(total / 60, 1),
                "Sessions":         ", ".join(
                    f"{r['StartTime']}–{r['EndTime']} {r['CourseName']}"
                    for _, r in run.iterrows()
                ),
            })

    return pd.DataFrame(results).reset_index(drop=True)


def detect_section_clashes() -> pd.DataFrame:
    """
    Detect any student section scheduled for two courses at the same time.

    Returns
    -------
    pd.DataFrame
        One row per clash with columns:
        Department, Semester, Section, Day, Time1, Course1, Time2, Course2.
        Empty DataFrame means no clashes.

    Example
    -------
    >>> clashes = detect_section_clashes()
    """
    tt = load_timetable()
    clashes = []

    group_keys = ["Department", "Semester", "Section", "Day"]
    for keys, group in tt.groupby(group_keys):
        slots = group.reset_index(drop=True)
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots.iloc[i], slots.iloc[j]
                if _overlaps(a["StartMinutes"], a["EndMinutes"],
                             b["StartMinutes"], b["EndMinutes"]):
                    dept, sem, sec, day = keys
                    clashes.append({
                        "Department": dept,
                        "Semester":   sem,
                        "Section":    sec,
                        "Day":        day,
                        "Time1":      f"{a['StartTime']}–{a['EndTime']}",
                        "Course1":    a["CourseName"],
                        "Time2":      f"{b['StartTime']}–{b['EndTime']}",
                        "Course2":    b["CourseName"],
                    })

    return pd.DataFrame(clashes)