"""
frontend/app.py
---------------
Streamlit chat interface for the Faculty Timetable Agent.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FacultyBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — monochrome, tight, academic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main layout */
.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1A1A3D;
    border-right: 1px solid #ececec;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 10px;
    border: 1px solid #ececec;
    background: white;
    color: #222;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #cfcfcf;
    background: #fafafa;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 0 2rem 0;
    margin-bottom: 1rem;
}

.app-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}

.app-header p {
    color: #666;
    font-size: 1rem;
    max-width: 700px;
    margin: 0 auto;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 12px;
}

[data-testid="stChatMessageContent"] {
    font-size: 15px;
    line-height: 1.7;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]) {
    border: 1px solid #f0f0f0;
}

/* Input */
.stChatInputContainer {
    background: white;
    border-top: none;
    padding-top: 1rem;
}

/* Badges */
.badge {
    display: inline-block;
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 999px;
    font-weight: 500;
    margin-top: 6px;
}

.badge-time {
    background: #f5f5f5;
    color: #666;
}

.badge-ok {
    background: #eef8f0;
    color: #2d7a46;
}

.badge-err {
    background: #fff0f0;
    color: #b42318;
}

/* Tool reasoning */
details {
    border: 1px solid #ececec;
    border-radius: 10px;
    padding: 0.75rem;
    background: #fafafa;
}

details summary {
    cursor: pointer;
    font-weight: 600;
}

/* Labels */
.sidebar-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888;
    margin-bottom: 8px;
    font-weight: 600;
}

hr.divider {
    border: none;
    border-top: 1px solid #ececec;
    margin: 1rem 0;
}

/* Code */
pre, code {
    border-radius: 8px !important;
}

/* Welcome cards */
.example-card {
    border: 1px solid #ececec;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    background: white;
}
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_loaded" not in st.session_state:
    st.session_state.agent_loaded = False
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0


@st.cache_resource(show_spinner=False)
def load_agent():
    from agent.agent import get_agent_executor
    return get_agent_executor()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-label'>Status</div>", unsafe_allow_html=True)

    with st.spinner("Loading…"):
        try:
            _agent = load_agent()
            st.session_state.agent_loaded = True
            st.markdown("<span class='badge badge-ok'>Agent ready</span>", unsafe_allow_html=True)
        except Exception as _e:
            st.session_state.agent_loaded = False
            st.markdown("<span class='badge badge-err'>Agent failed</span>", unsafe_allow_html=True)
            st.error(str(_e))

    # st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Stats
    # st.markdown("<div class='sidebar-label'>Session</div>", unsafe_allow_html=True)
    # col_a, col_b = st.columns(2)
    # col_a.metric("Queries", st.session_state.total_queries)
    # avg = (st.session_state.total_time / st.session_state.total_queries
    #        if st.session_state.total_queries > 0 else 0.0)
    # col_b.metric("Avg", f"{avg:.1f}s")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Examples — concise, one section
    st.markdown("<div class='sidebar-label'>Examples</div>", unsafe_allow_html=True)

    EXAMPLES = [
        "What is Prof. Sharma's workload?",
        "Summarise the CSE department workload.",
        "Which faculty is free on Tuesday at 14:00?",
        "Show Prof. Rao's weekly schedule.",
        "Are there any faculty scheduling conflicts?",
        "What is the maximum workload for a Professor?",
    ]

    for q in EXAMPLES:
        if st.button(q, key=f"ex_{q[:30]}"):
            st.session_state["_inject_query"] = q

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Controls
    st.markdown("<div class='sidebar-label'>Controls</div>", unsafe_allow_html=True)

    show_steps = st.toggle("Show reasoning steps", value=True)

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

    if st.button("Reload CSV caches", use_container_width=True):
        try:
            from utils.csv_loader import reload_all
            reload_all()
            st.success("CSV caches cleared.")
        except Exception as e:
            st.error(f"Reload failed: {e}")

    if st.button("Re-ingest vector DB", use_container_width=True):
        with st.spinner("Re-ingesting…"):
            try:
                from rag.ingest import ingest_all
                ingest_all(force_reingest=True)
                st.success("Vector DB rebuilt.")
            except Exception as e:
                st.error(f"Ingest failed: {e}")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>FacultyBot</h1>
    <p>Faculty workload and timetable assistant</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div style="
        border:1px solid #ececec;
        border-radius:16px;
        padding:24px;
        margin-bottom:20px;
    ">
        <h4 style="margin-top:0;">Suggested Questions</h4>

        • What is Prof. Sharma's workload?
        • Which faculty is free on Tuesday at 14:00?
        • Summarise the CSE department workload.
        • Are there any scheduling conflicts?
        • What is the maximum workload allowed?
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def _render_steps(steps: list[dict]) -> None:
    if not steps:
        return
    label = f"{len(steps)} tool call{'s' if len(steps) != 1 else ''}"
    with st.expander(label, expanded=False):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}. `{step['tool']}`**")
            st.markdown(
                f"<div style='font-size:0.75rem;color:#868e96;"
                f"font-family:JetBrains Mono,monospace;margin-bottom:4px;'>"
                f"Input: {step['tool_input']}</div>",
                unsafe_allow_html=True,
            )
            preview = step["result"][:500]
            if len(step["result"]) > 500:
                preview += "\n… (truncated)"
            st.code(preview, language="text")
            if i < len(steps):
                st.markdown("<hr style='margin:6px 0;border-color:#e9ecef;'>",
                            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("duration"):
                ok = msg.get("success", True)
                st.markdown(
                    f"<span class='badge badge-time'>{msg['duration']}s</span>"
                    f"&nbsp;<span class='badge {'badge-ok' if ok else 'badge-err'}'>"
                    f"{'ok' if ok else 'error'}</span>",
                    unsafe_allow_html=True,
                )
            if show_steps and msg.get("steps"):
                _render_steps(msg["steps"])


# ---------------------------------------------------------------------------
# Query injection from sidebar
# ---------------------------------------------------------------------------
injected = st.session_state.pop("_inject_query", None)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input(
    "Ask about workload, schedules, conflicts, or policies…",
    disabled=not st.session_state.agent_loaded,
) or injected

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<span style='color:#adb5bd;font-size:0.83rem;'>Thinking…</span>",
            unsafe_allow_html=True,
        )

        t0 = time.perf_counter()
        try:
            from agent.agent import run_query
            result = run_query(user_input)
        except Exception as exc:
            result = {
                "answer": f"Error: `{exc}`\n\nCheck the terminal logs or try again.",
                "steps": [],
                "duration": round(time.perf_counter() - t0, 2),
                "success": False,
                "error": str(exc),
            }

        placeholder.empty()
        st.markdown(result["answer"])

        ok = result["success"]
        st.markdown(
            f"<span class='badge badge-time'>{result['duration']}s</span>"
            f"&nbsp;<span class='badge {'badge-ok' if ok else 'badge-err'}'>"
            f"{'ok' if ok else 'error'}</span>",
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