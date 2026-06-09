"""
agent/tools.py  –  LangChain tool definitions for the Faculty Timetable Agent.
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
    return s.strip().strip("\n").strip()

def _df_to_str(df, max_rows: int = 15) -> str:
    if df.empty:
        return "No results found."
    try:
        return df.head(max_rows).to_markdown(index=False)
    except ImportError:
        # tabulate not installed — fall back to plain text
        return df.head(max_rows).to_string(index=False)

def _rag_query(retriever, query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('source','?')})\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


# ===========================================================================
# WORKLOAD TOOLS
# ===========================================================================

@tool
def get_faculty_workload_tool(faculty: str) -> str:
    """Get teaching/research/admin load for a faculty. Input: name or ID (e.g. 'Prof. Sharma', 'F101')."""
    faculty = _clean(faculty)
    try:
        return json.dumps(get_faculty_workload(faculty), indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_faculty_workload_report_tool(faculty: str) -> str:
    """Formatted workload report with load bar and compliance status. Input: name or ID."""
    faculty = _clean(faculty)
    try:
        report = get_workload_report("faculty", faculty)
        return f"```\n{report}\n```"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_department_workload_report_tool(department: str) -> str:
    """Formatted workload report for a whole department. Input: dept code e.g. 'CSE', 'EEE'."""
    department = _clean(department)
    try:
        report = get_workload_report("department", department)
        return f"```\n{report}\n```"
    except Exception as e:
        return f"Error: {e}"


@tool
def check_policy_compliance_tool(faculty: str) -> str:
    """Audit a faculty member against all workload policies. Input: name or ID."""
    faculty = _clean(faculty)
    try:
        return json.dumps(check_policy_compliance(faculty), indent=2)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_overloaded_faculty_tool(dummy: str = "") -> str:
    """List all faculty exceeding their maximum allowed teaching load. No input needed."""
    try:
        df = get_overloaded_faculty()
        return "All faculty within limits." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_underloaded_faculty_tool(threshold: str = "") -> str:
    """List faculty below minimum teaching load. Input: optional floor in hours e.g. '6', or blank for policy defaults."""
    threshold = _clean(threshold)
    try:
        t = int(threshold) if threshold.isdigit() else None
        df = get_underloaded_faculty(threshold=t)
        return "All faculty meet minimum load." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_hod_list_tool(dummy: str = "") -> str:
    """Return all Heads of Department with load details. No input needed."""
    try:
        df = get_hod_list()
        return "No HODs found." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_faculty_by_specialization_tool(keyword: str) -> str:
    """Find faculty by specialisation keyword (case-insensitive partial match). Input: e.g. 'Machine Learning', 'VLSI'."""
    keyword = _clean(keyword)
    try:
        df = get_faculty_by_specialization(keyword)
        return f"No faculty found for '{keyword}'." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_all_departments_summary_tool(dummy: str = "") -> str:
    """Institution-wide workload summary, one row per department. No input needed."""
    try:
        df = get_all_departments_summary()
        return "No data found." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_experienced_faculty_tool(min_years: str = "10") -> str:
    """List faculty with at least N years of experience. Input: number as string e.g. '10', '15'. Default 10."""
    min_years = _clean(min_years)
    try:
        years = int(min_years) if min_years.isdigit() else 10
        df = get_experienced_faculty(min_years=years)
        return f"No faculty with {years}+ years found." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# TIMETABLE TOOLS
# ===========================================================================

@tool
def get_faculty_schedule_tool(input: str) -> str:
    """Get a faculty's timetable. Input: 'Name or ID' or 'Name, Day' e.g. 'Prof. Sharma', 'F101, Monday'."""
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",", 1)]
        faculty = parts[0]
        day = parts[1] if len(parts) == 2 else None
        df = get_faculty_schedule(faculty, day=day)
        if df.empty:
            return f"No sessions found for '{faculty}'."
        df = df[["Day", "StartTime", "EndTime", "CourseName", "Room", "Section", "ClassType"]]
        return _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_free_faculty_at_tool(input: str) -> str:
    """Find faculty with no class at a given day and time. Input: 'Day, HH:MM' e.g. 'Tuesday, 14:00'."""
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",", 1)]
        if len(parts) != 2:
            return "Provide input as 'Day, HH:MM' e.g. 'Tuesday, 14:00'."
        day, time_str = parts
        free_df = get_free_faculty_at(day, time_str)
        if free_df.empty:
            return f"No faculty are free on {day} at {time_str}."
        return f"FREE on {day} at {time_str} ({len(free_df)} faculty):\n\n" + _df_to_str(free_df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_daily_schedule_tool(day: str) -> str:
    """All sessions across all departments for a given day. Input: day name e.g. 'Monday'."""
    day = _clean(day)
    try:
        df = get_daily_schedule(day)
        if df.empty:
            return f"No sessions on {day}."
        df = df[["StartTime", "EndTime", "Faculty", "CourseName", "Room", "Department", "Section"]]
        return _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_room_schedule_tool(input: str) -> str:
    """Get a room's occupancy schedule. Input: 'Room Name' or 'Room Name, Day' e.g. 'Room 201', 'Lab 101, Thursday'."""
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",", 1)]
        room = parts[0]
        day = parts[1] if len(parts) == 2 else None
        df = get_room_schedule(room, day=day)
        if df.empty:
            tag = f" on {day}" if day else ""
            return f"No sessions for '{room}'{tag}."
        df = df[["Day", "StartTime", "EndTime", "Faculty", "CourseName", "Section"]]
        return _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_section_schedule_tool(input: str) -> str:
    """Weekly timetable for a student section. Input: 'Department, Semester, Section' e.g. 'CSE, 5, A'."""
    input = _clean(input)
    try:
        parts = [p.strip() for p in input.split(",")]
        if len(parts) != 3:
            return "Provide input as 'Department, Semester, Section' e.g. 'CSE, 5, A'."
        dept, sem_str, section = parts
        df = get_section_schedule(dept, int(sem_str), section)
        if df.empty:
            return f"No timetable for {dept} Sem {sem_str} Sec {section}."
        df = df[["Day", "StartTime", "EndTime", "CourseName", "Faculty", "Room", "ClassType"]]
        return _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def suggest_free_slots_tool(faculty: str) -> str:
    """All free teaching slots across the week for a faculty. Input: name or ID e.g. 'Prof. Rao', 'F103'."""
    faculty = _clean(faculty)
    try:
        df = suggest_free_slots(faculty)
        return f"No free slots for '{faculty}'." if df.empty else _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_faculty_clashes_tool(dummy: str = "") -> str:
    """Scan timetable for faculty double-booked at the same time. No input needed."""
    try:
        df = detect_faculty_clashes()
        return "✓ No faculty clashes." if df.empty else f"⚠ {len(df)} clash(es):\n\n" + _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_room_clashes_tool(dummy: str = "") -> str:
    """Scan timetable for rooms double-booked at the same time. No input needed."""
    try:
        df = detect_room_clashes()
        return "✓ No room clashes." if df.empty else f"⚠ {len(df)} clash(es):\n\n" + _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_consecutive_overloads_tool(dummy: str = "") -> str:
    """Find faculty with more than 3 consecutive teaching hours (policy violation). No input needed."""
    try:
        df = detect_consecutive_overloads()
        return "✓ No consecutive overloads." if df.empty else f"⚠ {len(df)} violation(s):\n\n" + _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


@tool
def detect_section_clashes_tool(dummy: str = "") -> str:
    """Find student sections scheduled for two courses at the same time. No input needed."""
    try:
        df = detect_section_clashes()
        return "✓ No section clashes." if df.empty else f"⚠ {len(df)} clash(es):\n\n" + _df_to_str(df)
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# RAG TOOLS
# ===========================================================================

@tool
def policy_rag_tool(query: str) -> str:
    """Search university policy documents (max load, leave rules, invigilation, etc.). Input: natural-language question."""
    try:
        return _rag_query(get_policy_retriever(k=3), _clean(query))
    except Exception as e:
        return f"Error: {e}"


@tool
def workload_rag_tool(query: str) -> str:
    """Semantic search over faculty workload records. Use for fuzzy lookups e.g. 'who teaches algorithms'. Input: question."""
    try:
        return _rag_query(get_workload_retriever(k=3), _clean(query))
    except Exception as e:
        return f"Error: {e}"


@tool
def timetable_rag_tool(query: str) -> str:
    """Semantic search over timetable records. Use for open-ended queries e.g. 'Friday labs'. Input: question."""
    try:
        return _rag_query(get_timetable_retriever(k=3), _clean(query))
    except Exception as e:
        return f"Error: {e}"


@tool
def multi_source_rag_tool(query: str) -> str:
    """Fan-out search across policies, workload, and timetable. Use for cross-cutting queries. Input: question."""
    try:
        return _rag_query(get_multi_source_retriever(k_per_source=3), _clean(query))
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# Tool registry
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