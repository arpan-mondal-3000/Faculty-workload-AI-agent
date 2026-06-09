"""
utils/workload_utils.py
-----------------------
Faculty workload query, analysis, and report-generation utilities for the
Faculty Timetable Agent.

All functions operate on the normalised DataFrames returned by
``csv_loader.load_faculty_workload()`` and ``csv_loader.load_timetable()``.
They never modify source data.

Public API
----------
``get_faculty_workload(faculty)``
    Return workload details for a specific faculty member.

``get_department_workload_summary(department)``
    Summarise total and per-faculty workload for a department.

``get_overloaded_faculty()``
    Return faculty whose TotalLoad exceeds their MaxHoursAllowed.

``get_underloaded_faculty(threshold)``
    Return faculty whose TotalLoad is below a given threshold.

``get_workload_report(scope, identifier)``
    Generate a human-readable workload report for a faculty or department.

``get_all_departments_summary()``
    High-level workload summary across every department.

``get_hod_list()``
    Return all Heads of Department with their department and load.

``get_faculty_by_specialization(keyword)``
    Find faculty whose specialisation matches a keyword.

``check_policy_compliance(faculty)``
    Check whether a faculty member's load complies with university policies.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from utils.csv_loader import load_faculty_workload, load_timetable

logger = logging.getLogger(__name__)

# Policy constants (mirrored from policies.txt — single source of truth here
# so tools can enforce them programmatically)
_MAX_LOAD_BY_DESIGNATION = {
    "Professor":            12,
    "Associate Professor":  14,
    "Assistant Professor":  16,
}
_MIN_LOAD_BY_DESIGNATION = {
    "Professor":            4,
    "Associate Professor":  6,
    "Assistant Professor":  8,
}
_MAX_DAILY_HOURS          = 5
_MAX_CONSECUTIVE_HOURS    = 3
_MAX_COURSES_CSE          = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_faculty_row(query: str, wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows from ``wdf`` matching a faculty name (partial, case-insensitive)
    or an exact FacultyID.

    Parameters
    ----------
    query : str
        Name fragment or FacultyID.
    wdf : pd.DataFrame
        Workload DataFrame.

    Returns
    -------
    pd.DataFrame
        Matching rows (may be more than one if partial name is ambiguous).
    """
    q = query.strip()

    # Exact FacultyID match
    exact = wdf[wdf["FacultyID"].str.upper() == q.upper()]
    if not exact.empty:
        return exact

    # Partial name match (case-insensitive)
    return wdf[wdf["Name"].str.upper().str.contains(q.upper(), na=False)]


def _fmt_load_bar(load: int, max_load: int, width: int = 20) -> str:
    """
    Return a simple ASCII progress bar representing load utilisation.

    Example
    -------
    ``_fmt_load_bar(9, 12)`` → ``"███████████████░░░░░  75%"``
    """
    pct = min(load / max_load, 1.0) if max_load > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {pct * 100:.0f}%"


# ---------------------------------------------------------------------------
# Single-faculty queries
# ---------------------------------------------------------------------------

def get_faculty_workload(faculty: str) -> dict:
    """
    Return workload details for a single faculty member as a dictionary.

    Parameters
    ----------
    faculty : str
        Faculty name (partial or full) or FacultyID.

    Returns
    -------
    dict
        Keys: faculty_id, name, department, designation, course, course_code,
        hours_per_week, max_hours_allowed, research_load, admin_load,
        total_load, remaining_capacity, load_utilisation_pct, is_hod,
        specialization, years_of_experience, status.

    Raises
    ------
    ValueError
        If no match is found or the query is ambiguous (multiple matches).

    Example
    -------
    >>> info = get_faculty_workload("Prof. Sharma")
    >>> print(info["total_load"], "hrs total")
    """
    wdf = load_faculty_workload()
    rows = _resolve_faculty_row(faculty, wdf)

    if rows.empty:
        raise ValueError(f"No faculty found matching '{faculty}'.")
    if len(rows) > 1:
        names = rows["Name"].tolist()
        raise ValueError(
            f"Ambiguous query '{faculty}' matched multiple faculty: {names}. "
            "Please be more specific."
        )

    row = rows.iloc[0]
    return {
        "faculty_id":           row["FacultyID"],
        "name":                 row["Name"],
        "department":           row["Department"],
        "designation":          row["Designation"],
        "course":               row["Course"],
        "course_code":          row["CourseCode"],
        "hours_per_week":       int(row["HoursPerWeek"]),
        "max_hours_allowed":    int(row["MaxHoursAllowed"]),
        "research_load":        int(row["ResearchLoad"]),
        "admin_load":           int(row["AdminLoad"]),
        "total_load":           int(row["TotalLoad"]),
        "remaining_capacity":   int(row["RemainingCapacity"]),
        "load_utilisation_pct": float(row["LoadUtilisationPct"]),
        "is_hod":               bool(row["IsHOD_bool"]),
        "specialization":       row["Specialization"],
        "years_of_experience":  int(row["YearsOfExperience"]),
        "room_preference":      row["RoomPreference"],
        "status":               row["Status"],
    }


def check_policy_compliance(faculty: str) -> dict:
    """
    Check whether a faculty member's workload complies with all university
    policies and return a structured compliance report.

    Checks performed
    ----------------
    * Teaching load vs max allowed for their designation.
    * Teaching load vs minimum required for their designation.
    * Total load (teaching + research + admin) vs max allowed.
    * Timetable-derived: daily hours, consecutive hours per day.

    Parameters
    ----------
    faculty : str
        Faculty name or FacultyID.

    Returns
    -------
    dict
        Keys: faculty, compliant (bool), violations (list[str]),
        warnings (list[str]), summary (str).

    Example
    -------
    >>> report = check_policy_compliance("Prof. Anand")
    >>> for v in report["violations"]:
    ...     print("VIOLATION:", v)
    """
    from utils.timetable_utils import (
        get_faculty_schedule,
        detect_consecutive_overloads,
    )

    info = get_faculty_workload(faculty)
    violations: list[str] = []
    warnings:   list[str] = []

    designation = info["designation"]
    max_allowed = _MAX_LOAD_BY_DESIGNATION.get(designation, 12)
    min_required = _MIN_LOAD_BY_DESIGNATION.get(designation, 4)

    # 1. Max teaching load
    if info["hours_per_week"] > max_allowed:
        violations.append(
            f"Teaching load ({info['hours_per_week']} hrs) exceeds the maximum "
            f"allowed for {designation} ({max_allowed} hrs)."
        )

    # 2. Min teaching load
    if info["hours_per_week"] < min_required:
        warnings.append(
            f"Teaching load ({info['hours_per_week']} hrs) is below the minimum "
            f"required for {designation} ({min_required} hrs)."
        )

    # 3. Total load
    if info["total_load"] > max_allowed:
        violations.append(
            f"Total load ({info['total_load']} hrs) exceeds max allowed "
            f"({max_allowed} hrs) for {designation}."
        )

    # 4. Daily hour limit from timetable
    try:
        schedule = get_faculty_schedule(faculty)
        daily_totals = (
            schedule.groupby("Day")["DurationMinutes"]
            .sum()
            .apply(lambda m: m / 60)
        )
        for day, hrs in daily_totals.items():
            if hrs > _MAX_DAILY_HOURS:
                violations.append(
                    f"Daily teaching on {day} ({hrs:.1f} hrs) exceeds the "
                    f"max allowed {_MAX_DAILY_HOURS} hrs/day."
                )
    except ValueError:
        pass  # Faculty not in timetable yet

    # 5. Consecutive hours
    overloads = detect_consecutive_overloads()
    fid = info["faculty_id"]
    consecutive = overloads[overloads["FacultyID"] == fid]
    for _, row in consecutive.iterrows():
        violations.append(
            f"Consecutive teaching on {row['Day']}: {row['ConsecutiveHours']} hrs "
            f"(limit is {_MAX_CONSECUTIVE_HOURS} hrs). Sessions: {row['Sessions']}."
        )

    compliant = len(violations) == 0
    summary_parts = []
    if compliant and not warnings:
        summary_parts.append(
            f"{info['name']} is fully compliant. "
            f"Teaching load: {info['hours_per_week']} hrs "
            f"(limit: {max_allowed} hrs, min: {min_required} hrs)."
        )
    else:
        if violations:
            summary_parts.append(f"{len(violation)} policy violation(s) found.")
        if warnings:
            summary_parts.append(f"{len(warnings)} warning(s) noted.")

    return {
        "faculty":    info["name"],
        "faculty_id": info["faculty_id"],
        "compliant":  compliant,
        "violations": violations,
        "warnings":   warnings,
        "summary":    " ".join(summary_parts),
    }


# ---------------------------------------------------------------------------
# Department-level queries
# ---------------------------------------------------------------------------

def get_department_workload_summary(department: str) -> dict:
    """
    Summarise workload statistics for an entire department.

    Parameters
    ----------
    department : str
        Department code, e.g. ``"CSE"``. Case-insensitive.

    Returns
    -------
    dict
        Keys: department, faculty_count, total_teaching_hours,
        average_teaching_hours, max_teaching_hours, min_teaching_hours,
        total_load_all, overloaded_count, underloaded_count,
        hod_count, faculty (list of per-faculty dicts).

    Raises
    ------
    ValueError
        If the department is not found.

    Example
    -------
    >>> summary = get_department_workload_summary("CSE")
    >>> print(summary["total_teaching_hours"], "hrs total")
    """
    wdf = load_faculty_workload()
    dept_df = wdf[wdf["Department"].str.upper() == department.upper()]

    if dept_df.empty:
        available = sorted(wdf["Department"].unique().tolist())
        raise ValueError(
            f"Department '{department}' not found. "
            f"Available: {available}"
        )

    max_by_desig = dept_df["Designation"].map(_MAX_LOAD_BY_DESIGNATION).fillna(12)
    overloaded   = int((dept_df["HoursPerWeek"] > max_by_desig).sum())

    min_by_desig = dept_df["Designation"].map(_MIN_LOAD_BY_DESIGNATION).fillna(4)
    underloaded  = int((dept_df["HoursPerWeek"] < min_by_desig).sum())

    faculty_list = []
    for _, row in dept_df.sort_values("Name").iterrows():
        faculty_list.append({
            "faculty_id":         row["FacultyID"],
            "name":               row["Name"],
            "designation":        row["Designation"],
            "course":             row["Course"],
            "course_code":        row["CourseCode"],
            "hours_per_week":     int(row["HoursPerWeek"]),
            "total_load":         int(row["TotalLoad"]),
            "load_utilisation_pct": float(row["LoadUtilisationPct"]),
            "is_hod":             bool(row["IsHOD_bool"]),
        })

    return {
        "department":             department.upper(),
        "faculty_count":          len(dept_df),
        "total_teaching_hours":   int(dept_df["HoursPerWeek"].sum()),
        "average_teaching_hours": round(float(dept_df["HoursPerWeek"].mean()), 1),
        "max_teaching_hours":     int(dept_df["HoursPerWeek"].max()),
        "min_teaching_hours":     int(dept_df["HoursPerWeek"].min()),
        "total_load_all":         int(dept_df["TotalLoad"].sum()),
        "overloaded_count":       overloaded,
        "underloaded_count":      underloaded,
        "hod_count":              int(dept_df["IsHOD_bool"].sum()),
        "faculty":                faculty_list,
    }


def get_all_departments_summary() -> pd.DataFrame:
    """
    Return a one-row-per-department summary across the entire institution.

    Returns
    -------
    pd.DataFrame
        Columns: Department, FacultyCount, TotalTeachingHours,
        AvgTeachingHours, OverloadedCount, UnderloadedCount.
        Sorted by Department name.

    Example
    -------
    >>> df = get_all_departments_summary()
    >>> print(df.to_string(index=False))
    """
    wdf = load_faculty_workload()
    rows = []

    for dept, group in wdf.groupby("Department"):
        max_by_desig = group["Designation"].map(_MAX_LOAD_BY_DESIGNATION).fillna(12)
        min_by_desig = group["Designation"].map(_MIN_LOAD_BY_DESIGNATION).fillna(4)
        rows.append({
            "Department":         dept,
            "FacultyCount":       len(group),
            "TotalTeachingHours": int(group["HoursPerWeek"].sum()),
            "AvgTeachingHours":   round(float(group["HoursPerWeek"].mean()), 1),
            "TotalLoad":          int(group["TotalLoad"].sum()),
            "OverloadedCount":    int((group["HoursPerWeek"] > max_by_desig).sum()),
            "UnderloadedCount":   int((group["HoursPerWeek"] < min_by_desig).sum()),
        })

    return pd.DataFrame(rows).sort_values("Department").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Policy-based filters
# ---------------------------------------------------------------------------

def get_overloaded_faculty() -> pd.DataFrame:
    """
    Return faculty whose teaching load exceeds their designation's maximum.

    Returns
    -------
    pd.DataFrame
        Columns: FacultyID, Name, Department, Designation, HoursPerWeek,
        MaxAllowed, ExcessHours.

    Example
    -------
    >>> df = get_overloaded_faculty()
    """
    wdf = load_faculty_workload()
    wdf["MaxAllowed"] = wdf["Designation"].map(_MAX_LOAD_BY_DESIGNATION).fillna(12)
    overloaded = wdf[wdf["HoursPerWeek"] > wdf["MaxAllowed"]].copy()
    overloaded["ExcessHours"] = overloaded["HoursPerWeek"] - overloaded["MaxAllowed"]
    cols = ["FacultyID", "Name", "Department", "Designation",
            "HoursPerWeek", "MaxAllowed", "ExcessHours"]
    return overloaded[cols].sort_values("ExcessHours", ascending=False).reset_index(drop=True)


def get_underloaded_faculty(threshold: Optional[int] = None) -> pd.DataFrame:
    """
    Return faculty whose teaching load is below the designation minimum
    or a custom threshold.

    Parameters
    ----------
    threshold : int, optional
        Custom minimum hours. If omitted, uses policy minimums per designation.

    Returns
    -------
    pd.DataFrame
        Columns: FacultyID, Name, Department, Designation, HoursPerWeek,
        MinRequired, ShortfallHours.

    Example
    -------
    >>> df = get_underloaded_faculty()
    >>> df = get_underloaded_faculty(threshold=6)
    """
    wdf = load_faculty_workload()

    if threshold is not None:
        wdf["MinRequired"] = threshold
    else:
        wdf["MinRequired"] = wdf["Designation"].map(_MIN_LOAD_BY_DESIGNATION).fillna(4)

    under = wdf[wdf["HoursPerWeek"] < wdf["MinRequired"]].copy()
    under["ShortfallHours"] = under["MinRequired"] - under["HoursPerWeek"]
    cols = ["FacultyID", "Name", "Department", "Designation",
            "HoursPerWeek", "MinRequired", "ShortfallHours"]
    return under[cols].sort_values("ShortfallHours", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_hod_list() -> pd.DataFrame:
    """
    Return all Heads of Department with their department, name, and load.

    Returns
    -------
    pd.DataFrame
        Columns: FacultyID, Name, Department, Designation,
        HoursPerWeek, TotalLoad.

    Example
    -------
    >>> hods = get_hod_list()
    """
    wdf = load_faculty_workload()
    hods = wdf[wdf["IsHOD_bool"]][
        ["FacultyID", "Name", "Department", "Designation", "HoursPerWeek", "TotalLoad"]
    ].sort_values("Department").reset_index(drop=True)
    return hods


def get_faculty_by_specialization(keyword: str) -> pd.DataFrame:
    """
    Find faculty whose specialisation contains the given keyword.

    Parameters
    ----------
    keyword : str
        Partial or full specialisation term, e.g. ``"Machine Learning"``,
        ``"VLSI"``, ``"Thermal"``. Case-insensitive.

    Returns
    -------
    pd.DataFrame
        Matching faculty rows with key columns.

    Example
    -------
    >>> df = get_faculty_by_specialization("Network")
    """
    wdf = load_faculty_workload()
    mask = wdf["Specialization"].str.upper().str.contains(keyword.upper(), na=False)
    cols = ["FacultyID", "Name", "Department", "Designation",
            "Course", "Specialization", "HoursPerWeek", "TotalLoad"]
    return wdf.loc[mask, cols].sort_values(["Department", "Name"]).reset_index(drop=True)


def get_experienced_faculty(min_years: int = 10) -> pd.DataFrame:
    """
    Return faculty with at least ``min_years`` of experience.

    Parameters
    ----------
    min_years : int
        Minimum years of experience filter. Default 10.

    Returns
    -------
    pd.DataFrame

    Example
    -------
    >>> senior = get_experienced_faculty(min_years=15)
    """
    wdf = load_faculty_workload()
    mask = wdf["YearsOfExperience"] >= min_years
    cols = ["FacultyID", "Name", "Department", "Designation",
            "YearsOfExperience", "Specialization", "HoursPerWeek"]
    return (
        wdf.loc[mask, cols]
        .sort_values("YearsOfExperience", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def get_workload_report(scope: str, identifier: str) -> str:
    """
    Generate a formatted plain-text workload report.

    Parameters
    ----------
    scope : str
        ``"faculty"`` or ``"department"``.
    identifier : str
        Faculty name/ID (when scope is ``"faculty"``) or department code
        (when scope is ``"department"``).

    Returns
    -------
    str
        Human-readable multi-line report suitable for displaying in the
        Streamlit chat window or returning as an LLM tool result.

    Raises
    ------
    ValueError
        If scope is not ``"faculty"`` or ``"department"``.

    Example
    -------
    >>> print(get_workload_report("faculty", "Prof. Sharma"))
    >>> print(get_workload_report("department", "CSE"))
    """
    scope = scope.strip().lower()

    if scope == "faculty":
        info = get_faculty_workload(identifier)
        max_allowed = _MAX_LOAD_BY_DESIGNATION.get(info["designation"], 12)
        bar = _fmt_load_bar(info["hours_per_week"], max_allowed)

        lines = [
            "=" * 56,
            f"  FACULTY WORKLOAD REPORT",
            "=" * 56,
            f"  Name          : {info['name']}",
            f"  ID            : {info['faculty_id']}",
            f"  Designation   : {info['designation']}",
            f"  Department    : {info['department']}",
            f"  Specialization: {info['specialization']}",
            f"  Experience    : {info['years_of_experience']} years",
            f"  HOD           : {'Yes' if info['is_hod'] else 'No'}",
            f"  Status        : {info['status']}",
            "-" * 56,
            f"  Course        : {info['course']} ({info['course_code']})",
            f"  Teaching Load : {info['hours_per_week']} hrs/week",
            f"  Research Load : {info['research_load']} hrs/week",
            f"  Admin Load    : {info['admin_load']} hrs/week",
            f"  Total Load    : {info['total_load']} hrs/week",
            "-" * 56,
            f"  Max Allowed   : {max_allowed} hrs/week ({info['designation']})",
            f"  Remaining Cap : {info['remaining_capacity']} hrs",
            f"  Utilisation   : {bar}",
            "-" * 56,
        ]

        # Inline compliance check
        try:
            compliance = check_policy_compliance(identifier)
            if compliance["compliant"]:
                lines.append("  Policy Status : ✓ COMPLIANT")
            else:
                lines.append(f"  Policy Status : ✗ {len(compliance['violations'])} VIOLATION(S)")
                for v in compliance["violations"]:
                    lines.append(f"    • {v}")
            for w in compliance.get("warnings", []):
                lines.append(f"  Warning       : {w}")
        except Exception:
            pass

        lines.append("=" * 56)
        return "\n".join(lines)

    elif scope == "department":
        summary = get_department_workload_summary(identifier)
        lines = [
            "=" * 56,
            f"  DEPARTMENT WORKLOAD REPORT — {summary['department']}",
            "=" * 56,
            f"  Faculty Count         : {summary['faculty_count']}",
            f"  Total Teaching Hours  : {summary['total_teaching_hours']} hrs/week",
            f"  Average Teaching Load : {summary['average_teaching_hours']} hrs/week",
            f"  Max Individual Load   : {summary['max_teaching_hours']} hrs/week",
            f"  Min Individual Load   : {summary['min_teaching_hours']} hrs/week",
            f"  Total Load (all)      : {summary['total_load_all']} hrs/week",
            f"  Overloaded Faculty    : {summary['overloaded_count']}",
            f"  Underloaded Faculty   : {summary['underloaded_count']}",
            "-" * 56,
            "  FACULTY BREAKDOWN:",
            "-" * 56,
        ]
        for f in summary["faculty"]:
            hod_tag = " [HOD]" if f["is_hod"] else ""
            bar = _fmt_load_bar(f["hours_per_week"],
                                _MAX_LOAD_BY_DESIGNATION.get(f["designation"], 12),
                                width=12)
            lines.append(
                f"  {f['name']:<22}{hod_tag:<7}"
                f"{f['hours_per_week']:>2} hrs  {bar}"
            )
            lines.append(f"    └─ {f['course']} ({f['course_code']}), "
                         f"{f['designation']}")
        lines.append("=" * 56)
        return "\n".join(lines)

    else:
        raise ValueError(
            f"Invalid scope '{scope}'. Use 'faculty' or 'department'."
        )