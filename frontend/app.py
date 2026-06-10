"""
Streamlit chat interface for the Faculty Timetable Agent.
"""

from __future__ import annotations

import logging
import warnings
import sys
import time
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="FacultyBot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Canvas ── */
.main .block-container {
    max-width: 860px;
    padding: 2.5rem 2rem 5rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1729;
    border-right: 1px solid #1e2d4a;
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] * {
    color: #c8d3e8 !important;
}
section[data-testid="stSidebar"] .sidebar-section {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #4a5e80 !important;
    margin: 1.5rem 0 0.5rem;
    padding-left: 1rem;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: none;
    border-radius: 0;
    border-left: 2px solid transparent;
    padding: 0.45rem 1rem;
    text-align: left;
    font-size: 13px;
    color: #8fa3c2 !important;
    transition: border-color 0.15s, color 0.15s, background 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-left-color: #e8a020;
    color: #e8c87a !important;
    background: rgba(232,160,32,0.06);
}

/* Sidebar control buttons */
.ctrl-btn > button {
    border: 1px solid #1e2d4a !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.75rem !important;
    font-size: 12px !important;
    background: #141f35 !important;
    color: #8fa3c2 !important;
    transition: border-color 0.15s, color 0.15s !important;
}
.ctrl-btn > button:hover {
    border-color: #e8a020 !important;
    color: #e8c87a !important;
}

/* ── Page header ── */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #eaecf0;
}
.page-header h1 {
    font-size: 3rem;
    font-weight: 600;
    color: white;
    margin: 0 0 0.2rem;
    letter-spacing: -0.3px;
}
.page-header p {
    font-size: 13px;
    color: #8a93a2;
    margin: 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    padding: 0.25rem 0;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Signature left-border: targets the markdown container inside assistant messages ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    border-left: 3px solid #e8a020;
    padding-left: 1.1rem;
    margin: 0.25rem 0;
    line-height: 1.75;
}

/* ── Assistant text colors ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
    margin: 0 0 0.5rem;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] li {
    color: #e2e8f0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown strong {
    color: #f1c97a !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown h1,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown h2,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown h3 {
    color: #f0f4ff !important;
    margin: 0.9rem 0 0.4rem;
    font-weight: 600;
}

/* ── Tables in assistant replies ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) table {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 0.75rem 0 !important;
    font-size: 13px !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    overflow-x: auto;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) th {
    background: #1e2d4a !important;
    color: #c8d3e8 !important;
    padding: 7px 12px !important;
    text-align: left !important;
    font-weight: 600 !important;
    border: 1px solid #2a3f60 !important;
    white-space: nowrap;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) td {
    padding: 6px 12px !important;
    border: 1px solid #2a3f60 !important;
    color: #d1d9e8 !important;
    vertical-align: top;
    white-space: nowrap;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) tr:nth-child(even) td {
    background: #141f35 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) tr:hover td {
    background: #1a2c48 !important;
}

/* ── Code blocks in assistant replies ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) pre {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    background: #0d1a2e !important;
    border: 1px solid #2a3f60 !important;
    border-left: 3px solid #e8a020 !important;
    border-radius: 6px !important;
    padding: 1rem 1.25rem !important;
    color: #c8d3e8 !important;
    overflow-x: auto !important;
    white-space: pre !important;
    margin: 0.5rem 0 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) pre code {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    color: inherit !important;
    font-size: inherit !important;
    white-space: pre !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) code {
    font-family: 'DM Mono', monospace !important;
    font-size: 12.5px !important;
    background: #0d1526 !important;
    border: 1px solid #2a3f60 !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    color: #a8c7fa !important;
}

/* ── User message ── */
.user-bubble {
    background: #f6f7f9;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    color: #1a2035;
    font-size: 14.5px;
    line-height: 1.6;
    display: inline-block;
    max-width: 90%;
    float: right;
    clear: both;
}

/* ── Meta line under assistant messages ── */
.msg-meta {
    display: flex;
    gap: 8px;
    margin-top: 6px;
    margin-left: 1.1rem;
}
.chip {
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
}
.chip-time  { background: #f0f1f3; color: #6b7280; }
.chip-ok    { background: #edfaf2; color: #1a7f4e; }
.chip-err   { background: #fff2f2; color: #b91c1c; }

/* ── Tool steps expander ── */
details {
    margin-top: 8px;
    margin-left: 1.1rem;
    border: 1px solid #e9eaec;
    border-radius: 8px;
    background: #fafafa;
    overflow: hidden;
}
details summary {
    font-size: 11.5px;
    font-weight: 600;
    color: #6b7280;
    padding: 8px 12px;
    cursor: pointer;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    list-style: none;
}
details[open] summary {
    border-bottom: 1px solid #e9eaec;
}
.step-block {
    padding: 10px 14px;
    border-bottom: 1px solid #f0f0f0;
}
.step-block:last-child { border-bottom: none; }
.step-tool {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    color: #0f1729;
    margin-bottom: 4px;
}
.step-input {
    font-size: 11.5px;
    color: #8a93a2;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.step-result {
    font-size: 11.5px;
    font-family: 'DM Mono', monospace;
    background: #f6f7f9;
    border: 1px solid #e5e7eb;
    border-radius: 5px;
    padding: 6px 8px;
    white-space: pre-wrap;
    color: #374151;
    max-height: 160px;
    overflow-y: auto;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
    color: #9ca3af;
}
.empty-state .icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    opacity: 0.4;
}
.empty-state h3 {
    font-size: 1rem;
    font-weight: 500;
    color: #4b5563;
    margin: 0 0 0.5rem;
}
.empty-state p {
    font-size: 13px;
    max-width: 380px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}
.hint-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    max-width: 560px;
    margin: 0 auto;
    text-align: left;
}
.hint-item {
    background: #f9fafb;
    border: 1px solid #e9eaec;
    border-radius: 8px;
    padding: 10px 13px;
    font-size: 12.5px;
    color: #374151;
    line-height: 1.4;
}

/* ── Chat input ── */
.stChatInputContainer textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    border-radius: 10px !important;
    border-color: #d1d5db !important;
}
.stChatInputContainer textarea:focus {
    border-color: #e8a020 !important;
    box-shadow: 0 0 0 3px rgba(232,160,32,0.1) !important;
}

/* ── Status dot ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-ok  { background: #22c55e; }
.dot-err { background: #ef4444; }
.dot-loading { background: #f59e0b; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.status-text {
    font-size: 12px;
    color: #8fa3c2;
    font-family: 'DM Mono', monospace;
}

div[data-testid="stToggle"] label {
    font-size: 12.5px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, val in [
    ("messages", []),
    ("agent_loaded", False),
    ("total_queries", 0),
    ("total_time", 0.0),
]:
    if key not in st.session_state:
        st.session_state[key] = val


@st.cache_resource(show_spinner=False)
def load_agent():
    from agent.agent import get_agent_executor
    return get_agent_executor()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # Status
    st.markdown("<div class='sidebar-section'>Status</div>", unsafe_allow_html=True)
    with st.spinner(""):
        try:
            load_agent()
            st.session_state.agent_loaded = True
            st.markdown("""
                <div class='status-row'>
                    <div class='dot dot-ok'></div>
                    <span class='status-text'>Agent ready</span>
                </div>""", unsafe_allow_html=True)
        except Exception as _e:
            st.session_state.agent_loaded = False
            st.markdown("""
                <div class='status-row'>
                    <div class='dot dot-err'></div>
                    <span class='status-text'>Agent failed</span>
                </div>""", unsafe_allow_html=True)
            st.error(str(_e))

    # Examples
    st.markdown("<div class='sidebar-section'>Try asking</div>", unsafe_allow_html=True)
    EXAMPLES = [
        "What is Prof. Sharma's workload?",
        "Summarise CSE department workload.",
        "Who is free on Tuesday at 14:00?",
        "Show Prof. Rao's weekly schedule.",
        "Are there any faculty clashes?",
        "Max workload for a Professor?",
        "Generate EEE department report.",
        "Is Prof. Mehta policy compliant?",
    ]
    for q in EXAMPLES:
        if st.button(q, key=f"ex_{q[:28]}"):
            st.session_state["_inject_query"] = q

    # Controls
    st.markdown("<div class='sidebar-section'>Controls</div>", unsafe_allow_html=True)

    show_steps = st.toggle("Show reasoning steps", value=False)

    with st.container():
        st.markdown("<div class='ctrl-btn'>", unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.total_time = 0.0
            try:
                from agent.agent import reset_memory
                reset_memory()
            except Exception:
                pass
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='ctrl-btn'>", unsafe_allow_html=True)
        if st.button("Reload CSV data", use_container_width=True):
            try:
                from utils.csv_loader import reload_all
                reload_all()
                st.toast("CSV caches cleared.", icon="✓")
            except Exception as e:
                st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='ctrl-btn'>", unsafe_allow_html=True)
        if st.button("Rebuild vector DB", use_container_width=True):
            with st.spinner("Re-ingesting…"):
                try:
                    from rag.ingest import ingest_all
                    ingest_all(force_reingest=True)
                    st.toast("Vector DB rebuilt.", icon="✓")
                except Exception as e:
                    st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

    # Session stats footer
    if st.session_state.total_queries > 0:
        avg = st.session_state.total_time / st.session_state.total_queries
        st.markdown(
            f"<div style='padding:1rem 1rem 0;font-size:11px;"
            f"color:#3a4f6e;font-family:DM Mono,monospace;'>"
            f"{st.session_state.total_queries} queries · {avg:.1f}s avg</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("""
<div class='page-header'>
    <h1>FacultyBot</h1>
    <p>Workload management and timetable assistant — Greenfield Institute of Technology</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_steps(steps: list[dict]) -> None:
    if not steps:
        return
    label = f"{len(steps)} tool call{'s' if len(steps) != 1 else ''}"
    blocks = ""
    for i, step in enumerate(steps, 1):
        preview = step["result"][:400]
        if len(step["result"]) > 400:
            preview += "\n… (truncated)"
        blocks += f"""
        <div class='step-block'>
            <div class='step-tool'>{i}. {step['tool']}</div>
            <div class='step-input'>in: {step['tool_input']}</div>
            <div class='step-result'>{preview}</div>
        </div>"""
    st.markdown(
        f"<details><summary>{label}</summary>{blocks}</details>",
        unsafe_allow_html=True,
    )


def _meta_chips(duration: float, success: bool) -> str:
    t_chip = f"<span class='chip chip-time'>{duration}s</span>"
    s_chip = (
        f"<span class='chip chip-ok'>ok</span>" if success
        else f"<span class='chip chip-err'>error</span>"
    )
    return f"<div class='msg-meta'>{t_chip}&nbsp;{s_chip}</div>"


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(
                f"<div class='user-bubble'>{msg['content']}</div>"
                "<div style='clear:both'></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(msg["content"])
            if msg.get("duration"):
                st.markdown(
                    _meta_chips(msg["duration"], msg.get("success", True)),
                    unsafe_allow_html=True,
                )
            if show_steps and msg.get("steps"):
                _render_steps(msg["steps"])


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='icon'>📋</div>
        <h3>Ask anything about faculty or timetables</h3>
        <p>Query workloads, schedules, room availability, policy compliance, and more.</p>
        <div class='hint-grid'>
            <div class='hint-item'>What is Prof. Sharma's workload this week?</div>
            <div class='hint-item'>Which faculty are free on Tuesday at 14:00?</div>
            <div class='hint-item'>Summarise the CSE department workload.</div>
            <div class='hint-item'>Are there any scheduling conflicts?</div>
            <div class='hint-item'>Show Prof. Rao's timetable for Monday.</div>
            <div class='hint-item'>What is the max workload for a Professor?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Query injection from sidebar buttons
# ---------------------------------------------------------------------------
injected = st.session_state.pop("_inject_query", None)


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input(
    "Ask about workload, schedules, conflicts, or policies...",
    disabled=not st.session_state.agent_loaded,
) or injected

if user_input:
    # User turn
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(
            f"<div class='user-bubble'>{user_input}</div>"
            "<div style='clear:both'></div>",
            unsafe_allow_html=True,
        )

    # Assistant turn
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<span style='color:#9ca3af;font-size:13px;'>Thinking...</span>",
            unsafe_allow_html=True,
        )

        t0 = time.perf_counter()
        try:
            from agent.agent import run_query
            result = run_query(user_input)
        except Exception as exc:
            result = {
                "answer": f"Something went wrong: `{exc}`\n\nCheck the terminal logs or try again.",
                "steps": [],
                "duration": round(time.perf_counter() - t0, 2),
                "success": False,
                "error": str(exc),
            }

        placeholder.empty()

        st.markdown(result["answer"])
        st.markdown(
            _meta_chips(result["duration"], result["success"]),
            unsafe_allow_html=True,
        )
        if show_steps and result["steps"]:
            _render_steps(result["steps"])

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  result["answer"],
        "steps":    result["steps"],
        "duration": result["duration"],
        "success":  result["success"],
    })
    st.session_state.total_queries += 1
    st.session_state.total_time += result["duration"]