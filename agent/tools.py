"""
agent/tools.py
--------------
LangChain tool definitions for the Faculty Timetable Agent.

Each tool is a thin, well-documented wrapper that bridges a natural-language
agent call to one or more utility / RAG functions.  Tools are intentionally
kept narrow so the LLM can compose them for complex queries.

Tool catalogue
--------------
Workload tools
~~~~~~~~~~~~~~
``get_faculty_workload_tool``
    Detailed workload dict for one faculty member.

``get_department_workload_report_tool``
    Full formatted workload report for a department.

``get_faculty_workload_report_tool``
    Full formatted workload report for one faculty member.

``check_policy_compliance_tool``
    Policy compliance audit for one faculty member.

``get_overloaded_faculty_tool``
    List all faculty who exceed their maximum allowed load.

``get_underloaded_faculty_tool``
    List all faculty below their minimum required load.

``get_hod_list_tool``
    Return all Heads of Department.

``get_faculty_by_specialization_tool``
    Find faculty by specialisation keyword.

``get_all_departments_summary_tool``
    High-level workload snapshot across every department.

Timetable tools
~~~~~~~~~~~~~~~
``get_faculty_schedule_tool``
    Weekly (or single-day) schedule for one faculty member.

``get_free_faculty_at_tool``
    Which faculty are free on a given day and time.

``get_daily_schedule_tool``
    All sessions scheduled on a particular day.

``get_room_schedule_tool``
    Room utilisation schedule, optionally filtered by day.

``get_section_schedule_tool``
    Weekly timetable for a specific student section.

``suggest_free_slots_tool``
    All free teaching slots for a faculty member across the week.

``detect_faculty_clashes_tool``
    Find faculty double-booked at the same time.

``detect_room_clashes_tool``
    Find rooms double-booked at the same time.

``detect_consecutive_overloads_tool``
    Flag policy violations: more than 3 consecutive teaching hours.

``detect_section_clashes_tool``
    Find student sections with overlapping classes.

RAG tools
~~~~~~~~~
``policy_rag_tool``
    Semantic search over university policy documents.

``workload_rag_tool``
    Semantic search over embedded faculty workload records.

``timetable_rag_tool``
    Semantic search over embedded timetable records.

``multi_source_rag_tool``
    Fan-out RAG across all three collections for open-ended queries.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from utils.timetable_utils import (
    get_daily_schedule,
    get_faculty_schedule,
    get_free_faculty_at,
    get_room_schedule,
    get_section_schedule,
    suggest_free_slots,
    detect_faculty_clashes,
    detect_room_clashes,
    detect_consecutive_overloads,
    detect_section_clashes,
)
from utils.workload_utils import (
    check_policy_compliance,
    get_all_departments_summary,
    get_department_workload_summary,
    get_experienced_faculty,
    get_faculty_by_specialization,
    get_faculty_workload,
    get_hod_list,
    get_overloaded_faculty,
    get_underloaded_faculty,
    get_workload_report,
)
from rag.retriever import (
    get_multi_source_retriever,
    get_policy_retriever,
    get_timetable_retriever,
    get_workload_retriever,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    """Strip whitespace and newlines injected by the ReAct parser."""
    return s.strip().strip("\n").strip()


def _df_to_str(df) -> str:
    """Convert a DataFrame to a compact string for the LLM."""
    if df.empty:
        return "No results found."
    return df.to_string(index=False)


def _rag_query(retriever, query: str) -> str:
    """Run a retriever query and format the top documents as a string."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] ({source})\n{doc.page_content}")
    return "\n\n".join(parts)


# ===========================================================================
# WORKLOAD TOOLS
# ===========================================================================

@tool
def get_faculty_workload_tool(faculty: str) -> str:
    """
    Retrieve detailed workload information for a single faculty member.

    Use this when the user asks about a specific professor's courses, hours,
    load breakdown, or remaining capacity.

    Parameters
    ----------
    faculty : str
        Faculty name (full or partial, e.g. "Prof. Sharma", "Sharma") or
        FacultyID (e.g. "F101").

    Returns
    -------
    str
        JSON-formatted workload details including teaching hours, research
        load, admin load, total load, utilisation %, and policy status.

    Examples
    --------
    Input: "Prof. Sharma"
    Input: "F101"
    Input: "Mehta"
    """
    faculty = _clean(faculty)
    try:
        info = get_faculty_workload(faculty)
        return json.dumps(info, indent=2)
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception("get_faculty_workload_tool failed")
        return f"Error retrieving workload for '{faculty}': {e}"


@tool
def get_faculty_workload_report_tool(faculty: str) -> str:
    """
    Generate a detailed, formatted workload report for a single faculty member.

    Use this when the user wants a full printable summary — including load
    breakdown, utilisation bar, and inline policy compliance status.

    Parameters
    ----------
    faculty : str
        Faculty name (partial or full) or FacultyID.

    Returns
    -------
    str
        Multi-line formatted report ready for display in the chat window.

    Examples
    --------
    Input: "Prof. Iyer"
    Input: "F104"
    """
    faculty = _clean(faculty)
    try:
        return get_workload_report("faculty", faculty)
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception("get_faculty_workload_report_tool failed")
        return f"Error generating report for '{faculty}': {e}"


@tool
def get_department_workload_report_tool(department: str) -> str:
    """
    Generate a formatted workload report for an entire department.

    Use this for queries like "Summarise the CSE department workload" or
    "How is the EEE department load distributed?"

    Parameters
    ----------
    department : str
        Department code, e.g. "CSE", "EEE", "ME", "ECE". Case-insensitive.

    Returns
    -------
    str
        Multi-line formatted report with per-faculty breakdown, totals,
        overload/underload counts, and load utilisation bars.

    Examples
    --------
    Input: "CSE"
    Input: "eee"
    """
    department = _clean(department)
    try:
        return get_workload_report("department", department)
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception("get_department_workload_report_tool failed")
        return f"Error generating department report for '{department}': {e}"


@tool
def check_policy_compliance_tool(faculty: str) -> str:
    """
    Audit a faculty member's workload against all university policies.

    Use this when the user asks whether a professor is within limits, has
    policy violations, or to verify if their load is compliant.

    Checks performed:
    - Teaching load vs designation maximum and minimum.
    - Total load (teaching + research + admin) vs maximum.
    - Timetable-derived daily hours limit.
    - Consecutive teaching hours limit (max 3 hrs back-to-back).

    Parameters
    ----------
    faculty : str
        Faculty name (partial or full) or FacultyID.

    Returns
    -------
    str
        JSON report with keys: faculty, compliant (bool), violations (list),
        warnings (list), summary (str).

    Examples
    --------
    Input: "Prof. Anand"
    Input: "F102"
    """
    faculty = _clean(faculty)
    try:
        report = check_policy_compliance(faculty)
        return json.dumps(report, indent=2)
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception("check_policy_compliance_tool failed")
        return f"Error checking compliance for '{faculty}': {e}"


@tool
def get_overloaded_faculty_tool(dummy: str = "") -> str:
    """
    Return a list of all faculty members who exceed their maximum allowed
    teaching load according to university policy.

    Use this when the user asks "Who is overloaded?", "Which professors are
    exceeding their limit?", or to identify workload imbalances.

    Parameters
    ----------
    dummy : str
        Unused. Pass an empty string or any value.

    Returns
    -------
    str
        Table of overloaded faculty with columns: FacultyID, Name,
        Department, Designation, HoursPerWeek, MaxAllowed, ExcessHours.
        Returns "No overloaded faculty found." if everyone is within limits.

    Examples
    --------
    Input: ""
    """
    try:
        df = get_overloaded_faculty()
        if df.empty:
            return "No overloaded faculty found. All faculty are within their maximum load limits."
        return _df_to_str(df)
    except Exception as e:
        logger.exception("get_overloaded_faculty_tool failed")
        return f"Error retrieving overloaded faculty: {e}"


@tool
def get_underloaded_faculty_tool(threshold: str = "") -> str:
    """
    Return all faculty whose teaching load is below the required minimum.

    Use this to identify faculty with capacity to take on additional courses,
    or to flag potential under-utilisation.

    Parameters
    ----------
    threshold : str
        Optional custom minimum hours as a string, e.g. "6". If empty or
        omitted, uses the policy-defined minimum per designation.

    Returns
    -------
    str
        Table with columns: FacultyID, Name, Department, Designation,
        HoursPerWeek, MinRequired, ShortfallHours.

    Examples
    --------
    Input: ""          → uses policy minimums per designation
    Input: "6"         → flags anyone below 6 hrs/week
    """
    threshold = _clean(threshold)
    try:
        t = int(threshold.strip()) if threshold.strip().isdigit() else None
        df = get_underloaded_faculty(threshold=t)
        if df.empty:
            return "No underloaded faculty found. All faculty meet their minimum load requirements."
        return _df_to_str(df)
    except Exception as e:
        logger.exception("get_underloaded_faculty_tool failed")
        return f"Error retrieving underloaded faculty: {e}"


@tool
def get_hod_list_tool(dummy: str = "") -> str:
    """
    Return all Heads of Department along with their department, designation,
    and current workload.

    Use this when the user asks "Who are the HODs?", "List all department
    heads", or needs to contact the head of a specific department.

    Parameters
    ----------
    dummy : str
        Unused. Pass any value or empty string.

    Returns
    -------
    str
        Table with columns: FacultyID, Name, Department, Designation,
        HoursPerWeek, TotalLoad.

    Examples
    --------
    Input: ""
    """
    try:
        df = get_hod_list()
        if df.empty:
            return "No HODs found in the workload data."
        return _df_to_str(df)
    except Exception as e:
        logger.exception("get_hod_list_tool failed")
        return f"Error retrieving HOD list: {e}"


@tool
def get_faculty_by_specialization_tool(keyword: str) -> str:
    """
    Find faculty members whose specialisation matches a given keyword.

    Use this when the user asks "Who specialises in Machine Learning?",
    "Find faculty with VLSI expertise", or similar specialisation queries.

    Parameters
    ----------
    keyword : str
        Partial or full specialisation term. Case-insensitive.
        Examples: "Machine Learning", "VLSI", "Thermal", "Networks".

    Returns
    -------
    str
        Table of matching faculty with columns: FacultyID, Name, Department,
        Designation, Course, Specialization, HoursPerWeek, TotalLoad.

    Examples
    --------
    Input: "Machine Learning"
    Input: "network"
    Input: "thermal"
    """
    keyword = _clean(keyword)
    try:
        df = get_faculty_by_specialization(keyword)
        if df.empty:
            return f"No faculty found with specialisation matching '{keyword}'."
        return _df_to_str(df)
    except Exception as e:
        logger.exception("get_faculty_by_specialization_tool failed")
        return f"Error searching by specialisation '{keyword}': {e}"


@tool
def get_all_departments_summary_tool(dummy: str = "") -> str:
    """
    Return a high-level workload snapshot across every department in the
    institution.

    Use this for institution-wide overviews: "How is workload distributed
    across all departments?", "Which department has the highest load?",
    or to give the admin a bird's-eye view.

    Parameters
    ----------
    dummy : str
        Unused. Pass any value or empty string.

    Returns
    -------
    str
        JSON list, one entry per department, with keys: department,
        faculty_count, total_teaching_hours, average_teaching_hours,
        max_teaching_hours, min_teaching_hours, total_load_all,
        overloaded_count, underloaded_count.

    Examples
    --------
    Input: ""
    """
    try:
        df = get_all_departments_summary()
        if df.empty:
            return "No department data found."
        return _df_to_str(df)
    except Exception as e:
        logger.exception("get_all_departments_summary_tool failed")
        return f"Error retrieving all-departments summary: {e}"


@tool
def get_experienced_faculty_tool(min_years: str = "10") -> str:
    """
    Return faculty members with at least a given number of years of experience.

    Use this for queries like "List senior faculty", "Who has more than 15
    years of experience?", or to identify mentors for new hires.

    Parameters
    ----------
    min_years : str
        Minimum years of experience as a string. Default "10".

    Returns
    -------
    str
        Table with columns: FacultyID, Name, Department, Designation,
        YearsOfExperience, Specialization, HoursPerWeek.

    Examples
    --------
    Input: "10"
    Input: "15"
    """
    min_years = _clean(min_years)
    try:
        years = int(min_years.strip()) if min_years.strip().isdigit() else 10
        df = get_experienced_faculty(min_years=years)
        if df.empty:
            return f"No faculty found with {years}+ years of experience."
        return _df_to_str(df)
    except Exception as e:
        logger.exception("get_experienced_faculty_tool failed")
        return f"Error retrieving experienced faculty: {e}"


# ===========================================================================
# TIMETABLE TOOLS
# ===========================================================================

@tool
def get_faculty_schedule_tool(input: str) -> str:
    """
    Return the timetable for a specific faculty member, optionally for one day.

    Use this when the user asks "What is Prof. Sharma's schedule?",
    "When does Prof. Mehta teach on Monday?", or "Show me F103's timetable".

    Parameters
    ----------
    input : str
        Either just the faculty name/ID, or "faculty_name, day" separated by
        a comma. Examples: "Prof. Sharma", "F101, Monday", "Rao, Tuesday".

    Returns
    -------
    str
        Table of sessions sorted by day then start time, with columns:
        Day, StartTime, EndTime, CourseName, CourseCode, Section,
        Department, Semester, Room, ClassType, EnrolledStudents.

    Examples
    --------
    Input: "Prof. Sharma"
    Input: "F101, Monday"
    Input: "Mehta, Wednesday"
    """
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",", 1)]
        faculty = parts[0]
        day = parts[1] if len(parts) == 2 else None
        df = get_faculty_schedule(faculty, day=day)
        if df.empty:
            day_str = f" on {day}" if day else ""
            return f"No sessions found for '{faculty}'{day_str}."
        return _df_to_str(df)
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception("get_faculty_schedule_tool failed")
        return f"Error retrieving schedule for '{input}': {e}"


@tool
def get_free_faculty_at_tool(input: str) -> str:
    """
    Find all faculty members who have no class at a given day and time.

    Use this when the user asks "Which faculty is free on Tuesday at 2 PM?",
    "Who can I schedule for Monday 10 AM?", or availability queries.

    Parameters
    ----------
    input : str
        Day and time separated by a comma: "Day, HH:MM".
        Examples: "Tuesday, 14:00", "Monday, 10:00", "Friday, 09:00".

    Returns
    -------
    str
        Two sections:
        - FREE faculty (no class at that slot).
        - BUSY faculty (who ARE teaching at that slot, with course/room info).

    Examples
    --------
    Input: "Tuesday, 14:00"
    Input: "Monday, 10:00"
    Input: "Wednesday, 09:00"
    """
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",", 1)]
        if len(parts) != 2:
            return (
                "Please provide input as 'Day, HH:MM', "
                "e.g. 'Tuesday, 14:00'."
            )
        day, time_str = parts
        free_df = get_free_faculty_at(day, time_str)

        lines = [f"Faculty availability on {day} at {time_str}:\n"]

        if free_df.empty:
            lines.append("FREE: No faculty are free at this slot.")
        else:
            lines.append(f"FREE ({len(free_df)} faculty):")
            lines.append(_df_to_str(free_df))

        return "\n".join(lines)
    except Exception as e:
        logger.exception("get_free_faculty_at_tool failed")
        return f"Error checking availability for '{input}': {e}"


@tool
def get_daily_schedule_tool(day: str) -> str:
    """
    Return the complete schedule for all departments on a given day.

    Use this when the user asks "What is happening on Monday?",
    "Show me Tuesday's full timetable", or similar day-wide queries.

    Parameters
    ----------
    day : str
        Day name, e.g. "Monday", "Tuesday". Case-insensitive.

    Returns
    -------
    str
        All sessions on that day sorted by start time then department, with
        columns: StartTime, EndTime, Faculty, Department, CourseName,
        CourseCode, Section, Room, ClassType.

    Examples
    --------
    Input: "Monday"
    Input: "wednesday"
    """
    day = _clean(day)
    try:
        df = get_daily_schedule(day)
        if df.empty:
            return f"No sessions scheduled on {day}."
        return f"Full schedule for {day}:\n\n" + _df_to_str(df)
    except Exception as e:
        logger.exception("get_daily_schedule_tool failed")
        return f"Error retrieving schedule for '{day}': {e}"


@tool
def get_room_schedule_tool(input: str) -> str:
    """
    Return the occupancy schedule for a specific room, optionally on one day.

    Use this when the user asks "What is scheduled in Room 201?",
    "Is Lab 101 free on Thursday?", or room-utilisation queries.

    Parameters
    ----------
    input : str
        Room name alone, or "Room, Day". Examples:
        "Room 201", "Lab 101, Thursday", "Room 305".

    Returns
    -------
    str
        Sessions in that room sorted by day then start time, with columns:
        Day, StartTime, EndTime, Faculty, CourseName, Section, ClassType,
        EnrolledStudents.

    Examples
    --------
    Input: "Room 201"
    Input: "Lab 101, Thursday"
    """
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",", 1)]
        room = parts[0]
        day = parts[1] if len(parts) == 2 else None
        df = get_room_schedule(room, day=day)
        if df.empty:
            day_str = f" on {day}" if day else ""
            return f"No sessions found for '{room}'{day_str}."
        day_str = f" on {day}" if day else ""
        return f"Schedule for {room}{day_str}:\n\n" + _df_to_str(df)
    except Exception as e:
        logger.exception("get_room_schedule_tool failed")
        return f"Error retrieving room schedule for '{input}': {e}"


@tool
def get_section_schedule_tool(input: str) -> str:
    """
    Return the full weekly timetable for a specific student section.

    Use this when a student or coordinator asks "Show the timetable for
    CSE Semester 5 Section A", or "What are the classes for ECE Sem 3 B?"

    Parameters
    ----------
    input : str
        Comma-separated "Department, Semester, Section".
        Examples: "CSE, 5, A", "EEE, 3, B", "ME, 1, A".

    Returns
    -------
    str
        Weekly timetable for that section sorted by day then start time.

    Examples
    --------
    Input: "CSE, 5, A"
    Input: "EEE, 3, B"
    """
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",")]
        if len(parts) != 3:
            return (
                "Please provide input as 'Department, Semester, Section', "
                "e.g. 'CSE, 5, A'."
            )
        department, semester_str, section = parts
        semester = int(semester_str)
        df = get_section_schedule(department, semester, section)
        if df.empty:
            return (
                f"No timetable found for {department} "
                f"Semester {semester} Section {section}."
            )
        return (
            f"Timetable for {department} Sem {semester} "
            f"Section {section}:\n\n" + _df_to_str(df)
        )
    except ValueError as e:
        return f"Invalid input: {e}. Please use 'Department, Semester, Section'."
    except Exception as e:
        logger.exception("get_section_schedule_tool failed")
        return f"Error retrieving section schedule for '{input}': {e}"


@tool
def suggest_free_slots_tool(faculty: str) -> str:
    """
    List all available (free) teaching slots for a faculty member across the
    entire week, based on the standard institution slot grid.

    Use this when the admin wants to schedule a new class for a professor:
    "When is Prof. Rao free this week?", "Find a free slot for F103".

    Parameters
    ----------
    faculty : str
        Faculty name (partial or full) or FacultyID.

    Returns
    -------
    str
        Table of free slots with columns: Day, SlotStart, SlotEnd.
        Returns a message if no free slots are found.

    Examples
    --------
    Input: "Prof. Rao"
    Input: "F103"
    Input: "Iyer"
    """
    faculty = _clean(faculty)
    try:
        df = suggest_free_slots(faculty)
        if df.empty:
            return f"No free slots found for '{faculty}' — they may be fully booked."
        return f"Free teaching slots for {faculty}:\n\n" + _df_to_str(df)
    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.exception("suggest_free_slots_tool failed")
        return f"Error finding free slots for '{faculty}': {e}"


@tool
def detect_faculty_clashes_tool(dummy: str = "") -> str:
    """
    Scan the entire timetable for any faculty member assigned to two
    overlapping sessions simultaneously.

    Use this for timetable audits: "Are there any scheduling conflicts?",
    "Check for faculty clashes", "Is any professor double-booked?"

    Parameters
    ----------
    dummy : str
        Unused. Pass any value or empty string.

    Returns
    -------
    str
        Table of clashes with columns: FacultyID, Faculty, Day, Slot1,
        Time1, Course1, Slot2, Time2, Course2.
        Returns a clear message if no clashes are found.

    Examples
    --------
    Input: ""
    """
    try:
        df = detect_faculty_clashes()
        if df.empty:
            return "✓ No faculty clashes detected. All faculty timetables are conflict-free."
        return f"⚠ {len(df)} faculty clash(es) detected:\n\n" + _df_to_str(df)
    except Exception as e:
        logger.exception("detect_faculty_clashes_tool failed")
        return f"Error detecting faculty clashes: {e}"


@tool
def detect_room_clashes_tool(dummy: str = "") -> str:
    """
    Scan the entire timetable for any room that has been double-booked at
    overlapping times on the same day.

    Use this for: "Are any rooms double-booked?", "Check room conflicts",
    "Audit room allocations for clashes."

    Parameters
    ----------
    dummy : str
        Unused. Pass any value or empty string.

    Returns
    -------
    str
        Table of clashes with columns: Room, Day, Slot1, Time1, Faculty1,
        Course1, Slot2, Time2, Faculty2, Course2.
        Returns a clear message if no clashes exist.

    Examples
    --------
    Input: ""
    """
    try:
        df = detect_room_clashes()
        if df.empty:
            return "✓ No room clashes detected. All room allocations are conflict-free."
        return f"⚠ {len(df)} room clash(es) detected:\n\n" + _df_to_str(df)
    except Exception as e:
        logger.exception("detect_room_clashes_tool failed")
        return f"Error detecting room clashes: {e}"


@tool
def detect_consecutive_overloads_tool(dummy: str = "") -> str:
    """
    Identify faculty who have more than 3 consecutive teaching hours on any
    single day — a violation of university scheduling policy.

    Use this for: "Who is violating the consecutive hours policy?",
    "Check for back-to-back overloads", "Policy audit for teaching hours".

    Parameters
    ----------
    dummy : str
        Unused. Pass any value or empty string.

    Returns
    -------
    str
        Table with columns: FacultyID, Faculty, Day, ConsecutiveHours,
        Sessions.  Returns a clean message if no violations exist.

    Examples
    --------
    Input: ""
    """
    try:
        df = detect_consecutive_overloads()
        if df.empty:
            return (
                "✓ No consecutive teaching overloads detected. "
                "All faculty comply with the 3-hour consecutive limit."
            )
        return (
            f"⚠ {len(df)} consecutive-hours violation(s) detected:\n\n"
            + _df_to_str(df)
        )
    except Exception as e:
        logger.exception("detect_consecutive_overloads_tool failed")
        return f"Error detecting consecutive overloads: {e}"


@tool
def detect_section_clashes_tool(dummy: str = "") -> str:
    """
    Find any student section that has been scheduled for two courses at the
    same time (same department, semester, section, and day).

    Use this for: "Are there any section timetable conflicts?",
    "Check for overlapping classes for students."

    Parameters
    ----------
    dummy : str
        Unused. Pass any value or empty string.

    Returns
    -------
    str
        Table with columns: Department, Semester, Section, Day, Time1,
        Course1, Time2, Course2.  Returns a clean message if no clashes.

    Examples
    --------
    Input: ""
    """
    try:
        df = detect_section_clashes()
        if df.empty:
            return "✓ No section clashes detected. All student sections have conflict-free timetables."
        return f"⚠ {len(df)} section clash(es) detected:\n\n" + _df_to_str(df)
    except Exception as e:
        logger.exception("detect_section_clashes_tool failed")
        return f"Error detecting section clashes: {e}"


# ===========================================================================
# RAG TOOLS
# ===========================================================================

@tool
def policy_rag_tool(query: str) -> str:
    """
    Search university policy documents using semantic similarity.

    Use this for questions about rules, limits, guidelines, or institutional
    regulations: "What is the maximum workload for a Professor?",
    "Can a faculty teach more than 3 consecutive hours?",
    "What are the substitution rules?"

    Parameters
    ----------
    query : str
        Natural-language policy question.

    Returns
    -------
    str
        Top relevant policy excerpts retrieved from ChromaDB.

    Examples
    --------
    Input: "maximum workload per professor"
    Input: "consecutive teaching hours rule"
    Input: "substitution and leave policy"
    """
    query = _clean(query)
    try:
        retriever = get_policy_retriever(k=5)
        return _rag_query(retriever, query)
    except Exception as e:
        logger.exception("policy_rag_tool failed")
        return f"Error querying policy documents: {e}"


@tool
def workload_rag_tool(query: str) -> str:
    """
    Search embedded faculty workload records using semantic similarity.

    Use this for natural-language questions about a professor's courses,
    load profile, or department when exact structured lookup isn't enough:
    "Who in the CSE department teaches algorithms?",
    "List professors with high research load."

    Parameters
    ----------
    query : str
        Natural-language workload question.

    Returns
    -------
    str
        Top relevant workload records retrieved from ChromaDB, rendered as
        natural-language sentences.

    Examples
    --------
    Input: "Prof. Sharma's courses and load"
    Input: "faculty with high research load in EEE"
    Input: "who teaches fluid mechanics"
    """
    query = _clean(query)
    try:
        retriever = get_workload_retriever(k=5)
        return _rag_query(retriever, query)
    except Exception as e:
        logger.exception("workload_rag_tool failed")
        return f"Error querying workload records: {e}"


@tool
def timetable_rag_tool(query: str) -> str:
    """
    Search embedded timetable records using semantic similarity.

    Use this for flexible availability and scheduling queries when structured
    filters aren't practical: "Find lab sessions for CSE this week",
    "When is Room 203 typically occupied?",
    "Which faculty teach in the afternoon?"

    Parameters
    ----------
    query : str
        Natural-language timetable question.

    Returns
    -------
    str
        Top relevant timetable slot records retrieved from ChromaDB.

    Examples
    --------
    Input: "CSE lab sessions this week"
    Input: "afternoon classes in Room 203"
    Input: "who teaches on Saturday"
    """
    query = _clean(query)
    try:
        retriever = get_timetable_retriever(k=5)
        return _rag_query(retriever, query)
    except Exception as e:
        logger.exception("timetable_rag_tool failed")
        return f"Error querying timetable records: {e}"


@tool
def multi_source_rag_tool(query: str) -> str:
    """
    Search across all three knowledge sources (policies, workload, timetable)
    simultaneously using semantic similarity.

    Use this as the default RAG tool for open-ended or cross-cutting queries
    that span multiple data sources: "Tell me everything about Prof. Iyer",
    "Summarise the CSE department's schedule and workload",
    "Is Prof. Mehta's load and timetable compliant?"

    The retriever queries workload first, then timetable, then policies, and
    merges and deduplicates results by relevance.

    Parameters
    ----------
    query : str
        Open-ended natural-language question.

    Returns
    -------
    str
        Merged relevant documents from all three collections, labelled by
        source, ready for the LLM to synthesise an answer.

    Examples
    --------
    Input: "Tell me about Prof. Iyer's schedule and workload"
    Input: "Summarise CSE department timetable and load"
    Input: "Is the EEE department's workload within policy?"
    """
    query = _clean(query)
    try:
        retriever = get_multi_source_retriever(k_per_source=4)
        return _rag_query(retriever, query)
    except Exception as e:
        logger.exception("multi_source_rag_tool failed")
        return f"Error querying multi-source knowledge base: {e}"


# ===========================================================================
# Tool registry — import this list in agent.py
# ===========================================================================

ALL_TOOLS = [
    # Workload
    get_faculty_workload_tool,
    get_faculty_workload_report_tool,
    get_department_workload_report_tool,
    check_policy_compliance_tool,
    get_overloaded_faculty_tool,
    get_underloaded_faculty_tool,
    get_hod_list_tool,
    get_faculty_by_specialization_tool,
    get_all_departments_summary_tool,
    get_experienced_faculty_tool,
    # Timetable
    get_faculty_schedule_tool,
    get_free_faculty_at_tool,
    get_daily_schedule_tool,
    get_room_schedule_tool,
    get_section_schedule_tool,
    suggest_free_slots_tool,
    detect_faculty_clashes_tool,
    detect_room_clashes_tool,
    detect_consecutive_overloads_tool,
    detect_section_clashes_tool,
    # RAG
    policy_rag_tool,
    workload_rag_tool,
    timetable_rag_tool,
    multi_source_rag_tool,
]