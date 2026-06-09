"""
main.py
-------
Entry point for the Faculty Timetable Agent.

Usage
-----
    # Launch the Streamlit web app
    python main.py app

    # Interactive CLI session in the terminal
    python main.py cli

    # Run the data ingestion pipeline (build / rebuild ChromaDB)
    python main.py ingest
    python main.py ingest --force      # wipe existing DB and re-embed

Run ``python main.py --help`` or ``python main.py <command> --help`` for
full option details.
"""

from __future__ import annotations

import warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — configured before any project imports so all modules inherit it
# ---------------------------------------------------------------------------

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

_configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config validation (runs for every subcommand)
# ---------------------------------------------------------------------------

def _check_config() -> None:
    """Validate config and abort with a clear message if anything is missing."""
    from config import validate
    errors = validate()
    if errors:
        print("\n❌  Configuration errors found:\n")
        for e in errors:
            print(f"   • {e}")
        print(
            "\nFix the above issues in your .env file, then re-run.\n"
            "See .env.example for a full template.\n"
        )
        sys.exit(1)


# ===========================================================================
# Subcommand: app
# ===========================================================================

def cmd_app(args: argparse.Namespace) -> None:
    """
    Launch the Streamlit web application.

    Delegates to ``streamlit run frontend/app.py`` so hot-reload and all
    Streamlit features work as expected.
    """
    _check_config()

    app_path = Path(__file__).parent / "frontend" / "app.py"
    if not app_path.exists():
        print(f"❌  frontend/app.py not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port",        str(args.port),
        "--server.headless",    "true",
        "--browser.gatherUsageStats", "false",
    ]

    print(f"\n🚀  Starting FacultyBot on http://localhost:{args.port}\n")
    logger.info("Launching Streamlit: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nApp stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌  Streamlit exited with code {e.returncode}")
        sys.exit(e.returncode)


# ===========================================================================
# Subcommand: cli
# ===========================================================================

_CLI_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          FacultyBot — Timetable & Workload Assistant         ║
║             LLaMA 3.1 8B · Groq · ChromaDB                  ║
╠══════════════════════════════════════════════════════════════╣
║  Type your question and press Enter.                         ║
║  Commands:  /help · /clear · /history · /exit               ║
╚══════════════════════════════════════════════════════════════╝
"""

_CLI_HELP = """
Available commands
──────────────────
  /help       Show this help message
  /clear      Clear conversation memory and start fresh
  /history    Show the current conversation history
  /exit       Quit the CLI  (Ctrl-C also works)

Example queries
───────────────
  What is Prof. Sharma's workload this week?
  Which faculty is free on Tuesday at 2 PM?
  Summarise the CSE department workload.
  Are there any scheduling clashes?
  Is Prof. Mehta compliant with university policy?
  Generate a workload report for the EEE department.
"""

def _print_steps(steps: list[dict], verbose: bool) -> None:
    """Print intermediate tool calls if verbose mode is on."""
    if not verbose or not steps:
        return
    print(f"\n  ┌─ {len(steps)} tool call(s) ─────────────────────────────")
    for i, step in enumerate(steps, 1):
        print(f"  │  [{i}] {step['tool']}({step['tool_input']!r})")
        result_preview = step["result"][:200].replace("\n", " ")
        if len(step["result"]) > 200:
            result_preview += "…"
        print(f"  │      → {result_preview}")
    print("  └──────────────────────────────────────────────────────\n")


def cmd_cli(args: argparse.Namespace) -> None:
    """
    Run an interactive question-answer session in the terminal.

    Supports multi-turn conversation (memory is preserved between turns),
    slash commands for session management, and an optional --verbose flag
    that prints each tool call and its result.
    """
    _check_config()

    print(_CLI_BANNER)

    # Lazy import — avoids loading the heavy model before config is validated
    print("⏳  Loading agent (first load may take ~30 s) …", end="", flush=True)
    try:
        from agent.agent import get_agent_executor, reset_memory, run_query
        get_agent_executor()          # warm up
        print(" ✓\n")
    except Exception as e:
        print(f"\n❌  Failed to load agent: {e}")
        logger.exception("Agent load failure")
        sys.exit(1)

    history: list[tuple[str, str]] = []   # (question, answer) pairs

    while True:
        try:
            user_input = input("You › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────
        if user_input.lower() in ("/exit", "/quit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print(_CLI_HELP)
            continue

        if user_input.lower() == "/clear":
            reset_memory()
            history.clear()
            print("✓  Conversation memory cleared.\n")
            continue

        if user_input.lower() == "/history":
            if not history:
                print("  (no history yet)\n")
            else:
                print()
                for i, (q, a) in enumerate(history, 1):
                    print(f"  [{i}] You:    {q}")
                    # Show first 120 chars of answer to keep it readable
                    preview = a[:120].replace("\n", " ")
                    if len(a) > 120:
                        preview += "…"
                    print(f"       Bot:    {preview}\n")
            continue

        # ── Agent query ───────────────────────────────────────────
        print()
        try:
            result = run_query(user_input)
        except Exception as e:
            print(f"❌  Error: {e}\n")
            logger.exception("run_query failed")
            continue

        # Tool calls (verbose mode)
        _print_steps(result["steps"], verbose=args.verbose)

        # Answer
        print("Bot › ", end="")
        print(result["answer"])

        # Timing footer
        status = "✓" if result["success"] else "✗"
        step_count = len(result["steps"])
        print(
            f"\n  {status} {result['duration']}s · "
            f"{step_count} tool call{'s' if step_count != 1 else ''}\n"
        )

        history.append((user_input, result["answer"]))


# ===========================================================================
# Subcommand: ingest
# ===========================================================================

def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Run the data ingestion pipeline to build (or rebuild) ChromaDB collections.

    Three collections are created:
      • policies   — chunks from data/policies.txt
      • workload   — one doc per row in data/faculty_workload.csv
      • timetable  — one doc per row in data/timetable.csv

    Pass --force to wipe existing collections and re-embed from scratch.
    """
    _check_config()

    if args.force:
        print("⚠   --force flag set: existing ChromaDB collections will be deleted.\n")

    print("⏳  Starting ingestion pipeline …\n")

    # Lazy import
    try:
        from rag.ingest import ingest_all
    except Exception as e:
        print(f"❌  Import error: {e}")
        logger.exception("Ingest import failure")
        sys.exit(1)

    try:
        policies_store, workload_store, timetable_store = ingest_all(
            force_reingest=args.force
        )
    except Exception as e:
        print(f"\n❌  Ingestion failed: {e}")
        logger.exception("Ingestion failure")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════╗")
    print("║         Ingestion complete ✓             ║")
    print("╠══════════════════════════════════════════╣")

    def _collection_count(store) -> int:
        try:
            return store._collection.count()
        except Exception:
            return -1

    rows = [
        ("policies",  _collection_count(policies_store)),
        ("workload",  _collection_count(workload_store)),
        ("timetable", _collection_count(timetable_store)),
    ]
    for name, count in rows:
        count_str = str(count) if count >= 0 else "unknown"
        print(f"║  {name:<12}  {count_str:>6} documents       ║")

    from config import settings
    print("╠══════════════════════════════════════════╣")
    print(f"║  Stored at: {settings.CHROMA_PERSIST_DIR[-28:]:<30}║")
    print("╚══════════════════════════════════════════╝")
    print("\nRun `python main.py app` to start the web interface.\n")


# ===========================================================================
# Argument parser
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Faculty Timetable Agent — LLM-powered academic scheduling assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py app                  # start the Streamlit web app
  python main.py app --port 8502      # use a custom port
  python main.py cli                  # interactive terminal session
  python main.py cli --verbose        # show tool calls in the terminal
  python main.py ingest               # build ChromaDB (skips if already built)
  python main.py ingest --force       # wipe and rebuild ChromaDB from scratch
        """,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # ── app ───────────────────────────────────────────────────────
    p_app = subparsers.add_parser(
        "app",
        help="Launch the Streamlit web interface.",
        description="Start the FacultyBot Streamlit app in your browser.",
    )
    p_app.add_argument(
        "--port", type=int, default=8501,
        help="Port to run Streamlit on (default: 8501).",
    )
    p_app.set_defaults(func=cmd_app)

    # ── cli ───────────────────────────────────────────────────────
    p_cli = subparsers.add_parser(
        "cli",
        help="Interactive question-answer session in the terminal.",
        description=(
            "Ask questions about faculty workload and timetables directly "
            "in the terminal. Supports multi-turn conversation and slash commands."
        ),
    )
    p_cli.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each tool call and its result alongside the answer.",
    )
    p_cli.set_defaults(func=cmd_cli)

    # ── ingest ────────────────────────────────────────────────────
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Build (or rebuild) the ChromaDB vector store.",
        description=(
            "Reads data/faculty_workload.csv, data/timetable.csv, and "
            "data/policies.txt, converts them to embeddings, and stores "
            "them in ChromaDB. Safe to run multiple times — skips "
            "collections that already exist unless --force is passed."
        ),
    )
    p_ingest.add_argument(
        "--force", action="store_true",
        help="Delete existing ChromaDB collections and re-embed from scratch.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    return parser


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Re-configure logging now that we know the log level from config
    try:
        from config import settings
        _configure_logging(settings.LOG_LEVEL)
    except Exception:
        pass  # fall back to INFO

    args.func(args)


if __name__ == "__main__":
    main()