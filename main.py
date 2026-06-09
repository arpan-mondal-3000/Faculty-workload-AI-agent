"""
main.py  -  Entry point for the Faculty Timetable Agent.

    python main.py app              # launch Streamlit web app
    python main.py app --port 8502
    python main.py cli              # interactive terminal session
    python main.py cli --verbose    # show tool calls
    python main.py ingest           # build ChromaDB (skips if already built)
    python main.py ingest --force   # wipe and rebuild
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import warnings
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _check_config() -> None:
    from config import validate
    errors = validate()
    if errors:
        print("\nConfiguration errors:\n")
        for e in errors:
            print(f"  - {e}")
        print("\nFix the above in your .env file and re-run.\n")
        sys.exit(1)


# ===========================================================================
# app
# ===========================================================================

def cmd_app(args: argparse.Namespace) -> None:
    _check_config()

    app_path = Path(__file__).parent / "frontend" / "app.py"
    if not app_path.exists():
        print(f"frontend/app.py not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"\nStarting app on http://localhost:{args.port}\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApp stopped.")
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


# ===========================================================================
# cli
# ===========================================================================

_BANNER = """
Faculty Timetable Agent
-----------------------
Type a question and press Enter.
Commands: /help  /clear  /history  /exit
"""

_HELP = """
Commands
--------
  /help      Show this message
  /clear     Clear conversation memory
  /history   Show conversation history
  /exit      Quit  (Ctrl-C also works)

Example queries
---------------
  What is Prof. Sharma's workload this week?
  Which faculty is free on Tuesday at 2 PM?
  Summarise the CSE department workload.
  Are there any scheduling clashes?
  Is Prof. Mehta compliant with university policy?
"""


def cmd_cli(args: argparse.Namespace) -> None:
    _check_config()
    print(_BANNER)

    print("Loading agent (first load may take ~30 s) ...", end="", flush=True)
    try:
        from agent.agent import get_agent_executor, reset_memory, run_query
        get_agent_executor()
        print(" done\n")
    except Exception as e:
        print(f"\nFailed to load agent: {e}")
        sys.exit(1)

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print(_HELP)
            continue

        if user_input.lower() == "/clear":
            reset_memory()
            history.clear()
            print("Memory cleared.\n")
            continue

        if user_input.lower() == "/history":
            if not history:
                print("  (no history yet)\n")
            else:
                for i, (q, a) in enumerate(history, 1):
                    print(f"  [{i}] You: {q}")
                    preview = a[:120].replace("\n", " ")
                    print(f"       Bot: {preview}{'...' if len(a) > 120 else ''}\n")
            continue

        print()
        try:
            result = run_query(user_input)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        if args.verbose and result.get("steps"):
            for i, step in enumerate(result["steps"], 1):
                preview = step["result"][:200].replace("\n", " ")
                print(f"  [{i}] {step['tool']}({step['tool_input']!r})")
                print(f"       -> {preview}{'...' if len(step['result']) > 200 else ''}")
            print()

        print(f"Bot > {result['answer']}")
        n = len(result["steps"])
        print(f"\n  {result['duration']}s  {n} tool call{'s' if n != 1 else ''}\n")
        history.append((user_input, result["answer"]))


# ===========================================================================
# ingest
# ===========================================================================

def cmd_ingest(args: argparse.Namespace) -> None:
    _check_config()

    if args.force:
        print("--force set: existing collections will be deleted.\n")

    print("Starting ingestion ...\n")
    try:
        from rag.ingest import ingest_all
        policies_store, workload_store, timetable_store = ingest_all(
            force_reingest=args.force
        )
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)

    def _count(store) -> str:
        try:
            return str(store._collection.count())
        except Exception:
            return "unknown"

    print("Ingestion complete.\n")
    print(f"  policies  : {_count(policies_store)} documents")
    print(f"  workload  : {_count(workload_store)} documents")
    print(f"  timetable : {_count(timetable_store)} documents")

    from config import settings
    print(f"\n  Stored at : {settings.CHROMA_PERSIST_DIR}")
    print("\nRun 'python main.py app' to start the web interface.\n")


# ===========================================================================
# Parser + entry point
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Faculty Timetable Agent - LLM-powered academic scheduling assistant.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p_app = sub.add_parser("app", help="Launch the Streamlit web interface.")
    p_app.add_argument("--port", type=int, default=8501, help="Port (default: 8501).")
    p_app.set_defaults(func=cmd_app)

    p_cli = sub.add_parser("cli", help="Interactive terminal session.")
    p_cli.add_argument("--verbose", "-v", action="store_true", help="Show tool calls.")
    p_cli.set_defaults(func=cmd_cli)

    p_ingest = sub.add_parser("ingest", help="Build or rebuild ChromaDB.")
    p_ingest.add_argument("--force", action="store_true", help="Wipe and re-embed.")
    p_ingest.set_defaults(func=cmd_ingest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        from config import settings
        logging.getLogger().setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    except Exception:
        pass
    args.func(args)


if __name__ == "__main__":
    main()