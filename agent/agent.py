"""
agent/agent.py
--------------
Agent assembly for the Faculty Timetable Agent.

Architecture
------------
Pattern  : ReAct (Reasoning + Acting) via LangChain's ``create_react_agent``
LLM      : llama-3.1-8b-instant on Groq  (fast, low-latency)
Tools    : 24 tools across workload, timetable, and RAG categories
Memory   : ``ConversationBufferWindowMemory`` — keeps the last ``k`` turns so
           the agent can handle multi-turn follow-up questions without
           exceeding Groq's context window.

Public API
----------
``get_agent_executor()``
    Return the singleton ``AgentExecutor``.  Import this in ``main.py`` and
    the Streamlit ``app.py``.

``run_query(question, session_id)``
    Convenience wrapper that invokes the agent and returns a structured
    response dict with the answer, intermediate steps, and metadata.

``reset_memory()``
    Clear conversation history (call between sessions in the Streamlit app).
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

from agent.llm import get_llm
from agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System / Agent Prompt
# ---------------------------------------------------------------------------

# The ReAct prompt must contain the mandatory placeholders:
#   {tools}           — injected tool descriptions
#   {tool_names}      — comma-separated tool name list
#   {input}           — the user's question
#   {agent_scratchpad} — the Thought/Action/Observation chain
#   {chat_history}    — injected by ConversationBufferWindowMemory

SYSTEM_PROMPT = """
You are FacultyBot, an AI assistant for faculty workload, timetable management, scheduling conflicts, and policy compliance.

Available tools:
{tools}

Tool names:
{tool_names}

Tool Selection:
- Use structured tools whenever a faculty, department, room, section, or time is explicitly mentioned.
- Use policy_rag_tool, workload_rag_tool, timetable_rag_tool, or multi_source_rag_tool only for semantic, policy, vague, or cross-source questions.
- For compliance checks, always use check_policy_compliance_tool.
- Chain tools only when necessary.

Rules:
- Never fabricate tool outputs.
- Never guess schedules, workloads, rooms, or policy limits.
- Never call the same tool twice with identical input.
- Ask for clarification if multiple faculty match.
- Report tool errors honestly.
- If no data exists, say so.
- For greetings, answer directly without tools.

Format:

Thought: reasoning
Action: tool_name
Action Input: input
Observation: tool result

(repeat as needed)

Thought: I have enough information.
Final Answer: response

Answer Style:
- Use concise professional responses.
- Use tables for lists when appropriate.
- Preserve tool-generated reports inside code blocks.
- Prefix conflict reports with ✓ or ⚠.
- Cite policy rules when discussing compliance.

Chat History:
{chat_history}

Question:
{input}

{agent_scratchpad}
"""

# ---------------------------------------------------------------------------
# Memory (module-level so reset_memory() can clear it)
# ---------------------------------------------------------------------------

_memory: ConversationBufferWindowMemory | None = None


def _get_memory() -> ConversationBufferWindowMemory:
    """
    Return (or lazily create) the conversation memory buffer.

    Keeps the last 6 human/AI turns (~12 messages) — enough context for
    follow-up questions without overflowing Groq's 8 k-token context.
    """
    global _memory
    if _memory is None:
        _memory = ConversationBufferWindowMemory(
            k=6,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,   # plain string, not message objects
        )
    return _memory


def reset_memory() -> None:
    """
    Clear the conversation history.

    Call this from the Streamlit sidebar's "New Conversation" button or
    between test runs to start fresh.

    Example
    -------
    >>> from agent.agent import reset_memory
    >>> reset_memory()
    """
    global _memory
    if _memory is not None:
        _memory.clear()
        logger.info("Conversation memory cleared.")


# ---------------------------------------------------------------------------
# Agent executor factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_agent_executor() -> AgentExecutor:
    """
    Build and return the ``AgentExecutor``.

    This is called once and cached.  The memory object is NOT cached inside
    the executor because it must be mutable across turns — it is injected
    via ``_get_memory()`` which returns the module-level singleton.

    Note: ``lru_cache`` caches the executor structure, but memory state
    lives outside it, so ``reset_memory()`` works correctly.
    """
    llm = get_llm()

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

    react_agent = create_react_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=react_agent,
        tools=ALL_TOOLS,
        memory=_get_memory(),
        verbose=True,               # logs Thought/Action/Observation to stdout
        handle_parsing_errors="Please respond with a Final Answer directly if no tool is needed.", # gracefully recover from malformed LLM output
        max_iterations=5,          # prevent infinite tool-calling loops
        max_execution_time=120,     # 2-minute hard timeout per query
        return_intermediate_steps=True,
        early_stopping_method="generate",
    )

    logger.info(
        "AgentExecutor built — %d tools, model: llama-3.1-8b-instant",
        len(ALL_TOOLS),
    )
    return executor


def get_agent_executor() -> AgentExecutor:
    """
    Return the singleton ``AgentExecutor``.

    This is the primary entry point for ``main.py`` and ``frontend/app.py``.

    Returns
    -------
    AgentExecutor
        Fully configured agent with all 24 tools and conversation memory.

    Example
    -------
    >>> from agent.agent import get_agent_executor
    >>> executor = get_agent_executor()
    >>> result = executor.invoke({"input": "What is Prof. Sharma's workload?"})
    >>> print(result["output"])
    """
    return _build_agent_executor()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_query(question: str, session_id: str = "default") -> dict[str, Any]:
    """
    Run a user query through the agent and return a structured response.

    This wrapper adds timing metadata and normalises the output so callers
    (``main.py``, Streamlit ``app.py``, tests) always get the same shape back.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    session_id : str
        Optional identifier for logging — useful when running multiple
        concurrent Streamlit sessions.  Default ``"default"``.

    Returns
    -------
    dict with keys:
        ``answer``   (str)  — the agent's Final Answer.
        ``steps``    (list) — intermediate (tool_name, result) pairs.
        ``duration`` (float)— wall-clock seconds for the full run.
        ``success``  (bool) — False if an exception was caught.
        ``error``    (str | None) — exception message if success is False.

    Example
    -------
    >>> result = run_query("Which faculty is free on Tuesday at 2 PM?")
    >>> print(result["answer"])
    >>> print(f"Answered in {result['duration']:.1f}s")
    """
    executor = get_agent_executor()
    start = time.perf_counter()

    logger.info("[session=%s] Query: %r", session_id, question)

    try:
        raw = executor.invoke({"input": question})
        duration = time.perf_counter() - start

        # Unpack intermediate steps into a friendlier format
        steps = []
        for action, observation in raw.get("intermediate_steps", []):
            steps.append({
                "tool":       action.tool,
                "tool_input": action.tool_input,
                "log":        action.log.strip(),
                "result":     str(observation)[:500],  # truncate for logging
            })

        logger.info(
            "[session=%s] Answered in %.2fs using %d tool call(s).",
            session_id, duration, len(steps),
        )

        return {
            "answer":   raw.get("output", ""),
            "steps":    steps,
            "duration": round(duration, 2),
            "success":  True,
            "error":    None,
        }

    except Exception as exc:
        duration = time.perf_counter() - start
        logger.exception("[session=%s] Agent error after %.2fs", session_id, duration)

        return {
            "answer": (
                "I encountered an error while processing your request. "
                "Please try rephrasing your question or check the logs."
            ),
            "steps":    [],
            "duration": round(duration, 2),
            "success":  False,
            "error":    str(exc),
        }