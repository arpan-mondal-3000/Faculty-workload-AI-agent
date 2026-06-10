# FacultyBot — Faculty Workload & Timetable Assistant

An AI agent that answers natural-language queries about faculty schedules,
workloads, room availability, and university policy compliance.

---

## Tech Stack

| Layer | Library |
|---|---|
| LLM | LLaMA 3.1 8B via Groq API |
| Agent framework | LangChain (ReAct agent) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Data | pandas (CSV processing) |
| Frontend | Streamlit |

---

## Project Structure

```
├── agent/
│   ├── agent.py        # ReAct agent assembly and memory
│   ├── llm.py          # Groq LLM setup
│   └── tools.py        # All 24 LangChain tools
├── rag/
│   ├── embeddings.py   # Embedding model wrapper
│   ├── ingest.py       # CSV + policy ingestion into 
│   └── retriever.py    # Per-collection and multi-source retrievers
├── utils/
│   ├── csv_loader.py   # Cached DataFrame loaders
│   ├── timetable_utils.py  # Schedule queries and clash detection
│   └── workload_utils.py   # Workload queries and report generation
├── data/
│   ├── faculty_workload.csv
│   ├── timetable.csv
│   └── policies.txt
├── frontend/
│   └── app.py          # Streamlit chat UI
├── config.py
└── main.py             # CLI entry point
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/arpan-mondal-3000/Faculty-workload-AI-agent.git
cd Faculty-workload-AI-agent
pip install -e .
```

**2. Create a `.env` file**

```env
GROQ_API_KEY=your_groq_api_key_here
FACULTY_WORKLOAD_PATH=data/faculty_workload.csv
TIMETABLE_PATH=data/timetable.csv
POLICIES_PATH=data/policies.txt
CHROMA_PERSIST_DIR=.chromadb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
LOG_LEVEL=INFO
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**3. Ingest data into ChromaDB**

```bash
python main.py ingest
```

This reads the three data files, embeds them, and writes the vectors to disk.
Only needs to be run once, or again with `--force` if the data changes.

```bash
python main.py ingest --force   # wipe and rebuild
```

---

## Running the App

**Web interface (recommended)**

```bash
python main.py app
```

Opens at `http://localhost:8501`. Use `--port 8502` to change the port.

**Terminal (CLI)**

```bash
python main.py cli
python main.py cli --verbose    # also prints tool calls
```

Supports slash commands: `/help`, `/clear`, `/history`, `/exit`.

---

## What You Can Ask

**Workload queries**
- *What is Prof. Sharma's workload this week?*
- *Summarise the CSE department workload.*
- *Which faculty are overloaded?*
- *Who specialises in Machine Learning?*

**Schedule queries**
- *Show Prof. Rao's timetable for Monday.*
- *Which faculty are free on Tuesday at 14:00?*
- *What is scheduled in Room 201 this week?*
- *Show the timetable for CSE Semester 5 Section A.*

**Conflict detection**
- *Are there any faculty scheduling clashes?*
- *Are any rooms double-booked?*
- *Which faculty have too many consecutive classes?*

**Policy and compliance**
- *Is Prof. Mehta compliant with university policy?*
- *What is the maximum workload for a Professor?*
- *What are the rules for faculty substitution?*

**Reports**
- *Generate a workload report for Prof. Iyer.*
- *Generate an EEE department workload report.*

---

## Data Files

| File | Description |
|---|---|
| `faculty_workload.csv` | Faculty name, department, designation, courses, and hours |
| `timetable.csv` | Day, time, course, faculty, room, section for each slot |
| `policies.txt` | University rules on workload limits, scheduling, and leave |

To update data, edit the CSV/text files and run `python main.py ingest --force`.