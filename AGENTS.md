# RAGResearch — Agent Architecture

Multi-agent RAG system built with LangGraph. Takes a user query, researches it (web + local documents), and produces a structured project plan with GitHub issues, calendar events, and notes.

## Modes

**Project mode** — full pipeline: research → summary → tasks → GitHub issues + code snippets → README.

**Learning mode** — detects whether the query can be answered from local files or needs web research, then generates structured notes and optionally a calendar.

## Graph flows

```
Project:
  research_agent → (tools loop) → summarization → task_planner
    → scaffolding → github_issues → readme → END

Learning:
  detect_intent → local_files: notes → task_planner → END
              └→ research: research_agent → summarization → task_planner
                   → (calendar →)? notes → END
```

## Nodes

| Node | File | What it does                                                                                         |
|------|------|------------------------------------------------------------------------------------------------------|
| `research_agent` | `agents/nodes/research.py` | ReAct agent with DuckDuckGo, Tavily and RAG tools; loops until it decides to stop                    |
| `research_tools` | `agents/nodes/research.py` | Executes tool calls emitted by `research_agent`                                                      |
| `summarization` | `agents/nodes/summarization.py` | Consolidates raw research messages into a coherent summary                                           |
| `task_planner` | `agents/nodes/summarization.py` | Extracts a list of actionable tasks `{title, description, priority, duration_minutes}` from the summary |
| `scaffolding` | `agents/nodes/scaffolding.py` | Generates a project-level file structure (`filepath`, `purpose`, `code`); stored in `state.scaffold` |
| `github_issues` | `agents/nodes/github_issues.py` | Creates a GitHub repo (optionally) and opens one issue per task |
| `readme` | `agents/nodes/readme.py` | Generates a project README from the summary and task list                                            |
| `calendar` | `agents/nodes/calendar.py` | Exports tasks to Google Calendar                                                                     |
| `notes` | `agents/nodes/notes.py` | Generates structured learning notes (learning mode)                                                  |
| `detect_intent` | `agents/nodes/detect_intent.py` | Classifies query as `local_files` or `research`                                                      |

## State (`agents/state.py`)

`AgentState` is a LangGraph `TypedDict` shared across all nodes:

```python
query: str              # original user query
messages: list          # LangChain message history (append-only via add_messages)
local_result: dict      # result from local RAG search
web_result: dict        # result from web search
summary: str            # output of summarization_node
tasks: list[dict]       # output of task_planner_node
mode: 'project'|'learning'
create_repo: bool       # whether to auto-create a GitHub repo
use_calendar: bool      # whether to export to Google Calendar
github_issues: list[dict]
scaffold: list[dict]    # [{filepath, purpose, code}]
readme: str
notes: str
calendar_events: list[dict]
intent: 'local_files'|'research'
```

## RAG (`rag/`)

- `loader.py` — loads PDF, DOCX, TXT files
- `splitter.py` — splits documents into overlapping chunks
- `vector_storage.py` — manages two ChromaDB collections: `research` (user docs) and `code` (code files)
- `minio_storage.py` — stores raw files in a MinIO (S3-compatible) bucket

## LLM factory (`agents/__init__.py`)

`get_llm()` returns a LangChain chat model based on `LLM_PROVIDER` env var:
- `openai` — `ChatOpenAI` (default `gpt-4o-mini`)
- `groq` — `ChatGroq`
- `ollama` — `ChatOllama` (default `llama3.2`)

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | no | `openai` / `groq` / `ollama` (default: `openai`) |
| `OPENAI_API_KEY` | if OpenAI | OpenAI key |
| `GROQ_API_KEY` | if Groq | Groq key |
| `GITHUB_TOKEN` | for GitHub nodes | Personal access token with `repo` scope |
| `GITHUB_REPO` | no | `owner/repo` to use instead of auto-creating |
| `GOOGLE_CREDENTIALS_FILE` | for calendar | Path to Google OAuth2 credentials JSON |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | for MinIO | MinIO connection details |

## Running

```bash
pip install -r requirements.txt
docker compose up -d          # starts MinIO
streamlit run ui/app.py       # Streamlit UI
# or directly:
python main.py
```

## Tests

```bash
pytest tests/
```

Tests use `pytest` with `unittest.mock` — no live API calls are made. GitHub-related tests are in `tests/test_github_issues.py`.
