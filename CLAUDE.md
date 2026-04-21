# CLAUDE.md

See [AGENTS.md](AGENTS.md) for a full description of the project architecture, agent nodes, state schema, RAG pipeline, and environment variables.

## Quick reference

- Entry point: `main.py` (CLI) or `streamlit run ui/app.py` (UI)
- Agent graph: `agents/graph.py` — `build_project_graph()` and `build_learning_graph()`
- State type: `agents/state.py` — `AgentState`
- LLM factory: `agents/__init__.py` — `get_llm()`
- GitHub integration: `agents/nodes/github_issues.py`
- Tests: `pytest tests/`

## Dev setup

```bash
pip install -r requirements.txt
docker compose up -d
cp .env.example .env  # then fill in keys
```
