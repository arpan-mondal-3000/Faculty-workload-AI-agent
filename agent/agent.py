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


SYSTEM_PROMPT = """You are FacultyBot, an academic scheduling assistant for University of Calcutta.

TOOLS:
{tools}

Tool names: {tool_names}

RULES:
- Greetings / general knowledge / thanks → answer directly, NO tools.
- Named faculty or ID → get_faculty_workload_tool or get_faculty_schedule_tool
- Department mentioned → get_department_workload_report_tool
- Day + time → get_free_faculty_at_tool
- Room mentioned → get_room_schedule_tool
- Clashes / conflicts → detect_faculty_clashes_tool or detect_room_clashes_tool
- Compliance / policy + faculty → check_policy_compliance_tool
- Report requested → get_faculty_workload_report_tool or get_department_workload_report_tool
- Vague / cross-cutting → multi_source_rag_tool
- Policy rule question → policy_rag_tool
- Never fabricate data. Never call the same tool twice with the same input.
- If a tool errors, report it. If data is missing, say so.
- Ask for clarification only when a name matches multiple people.
- Responses: concise and professional; if a tool returns a table, reproduce it exactly in your Final Answer — never summarise tabular data into prose; use code blocks for preformatted reports.

FORMAT — no tool needed:
Thought: No tool needed.
Final Answer: <answer>

FORMAT — tool needed:
Thought: <reasoning>
Action: <tool_name>
Action Input: <input>
Observation: <result>
...repeat as needed...
Thought: I have enough information.
Final Answer: <answer>

Chat History: {chat_history}
Question: {input}
{agent_scratchpad}"""

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

_memory: ConversationBufferWindowMemory | None = None


def _get_memory() -> ConversationBufferWindowMemory:
    global _memory
    if _memory is None:
        _memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=False,
        )
    return _memory


def reset_memory() -> None:
    """
    Clear the conversation history.
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
        verbose=True,
        handle_parsing_errors=(
            "Your last response could not be parsed. "
            "If no tool is needed, respond ONLY with:\n"
            "Thought: This question does not require any tools. I can answer directly.\n"
            "Final Answer: <your answer>\n"
            "Do not add anything else before 'Thought:'."
        ),
        max_iterations=5,
        max_execution_time=120,
        return_intermediate_steps=True,
        early_stopping_method="generate",
    )

    logger.info(
        "AgentExecutor built — %d tools, model: llama-3.1-8b-instant",
        len(ALL_TOOLS),
    )
    return executor


def get_agent_executor() -> AgentExecutor:
    return _build_agent_executor()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_query(question: str, session_id: str = "default") -> dict[str, Any]:
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