from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from agents import get_llm
from agents.state import AgentState

DETECT_STACK_PROMPT = """Identify the full technology stack requested or implied by this project query.
Query: {query}

Return a JSON object with exactly these keys:
  "primary_language" – the main backend language (e.g. Python, TypeScript, Go)
  "stack"            – list of all technologies/frameworks explicitly mentioned or strongly implied
                       (e.g. ["FastAPI", "React", "PostgreSQL", "Redis"])
  "is_fullstack"     – true if the project has both a backend and a frontend

Return ONLY valid JSON, no markdown.
"""

SCAFFOLDING_PROMPT = """You are a software architect generating a project scaffold.

Original request: {query}

Project description:
{summary}

Tech stack: {stack}

Design a realistic, production-quality file structure that covers the ENTIRE stack described above.

Rules:
- Honor every framework and technology explicitly named in the original request — do NOT substitute (e.g. if FastAPI is requested, use FastAPI not Flask; if React is requested, create a React frontend)
- For fullstack projects create separate top-level directories for backend and frontend (e.g. backend/, frontend/)
- Group related functionality into the same file (auth routes in one file, not login.py + register.py)
- Use conventional layout for each technology (FastAPI → routers/, models/, schemas/; React → src/components/, src/pages/)
- Write real, working code. For straightforward logic (CRUD routes, React components, models, config) write the full implementation. For genuinely complex algorithms you may leave a `# TODO` only if you explain exactly what goes there in a comment.
  Target quality: a React component should have real hooks, real JSX and real API calls. A FastAPI route should query the DB and return the right schema. A SQLAlchemy model should have all columns and relationships.
  Never write an empty function body with just `pass` or `// TODO implement this`.
- Cover the core domain logic of the project (e.g. for habit tracker: habits CRUD, streak calculation, daily check-ins) — not just auth/user boilerplate
- Aim for 10–20 files that cover the full project
- filepath must be a valid relative path with the correct extension

Return a JSON array. Each element must have exactly these keys:
  "filepath"  – relative path of the file
  "purpose"   – one sentence describing what this file does
  "code"      – scaffold source code

Return ONLY valid JSON. No markdown, no explanations.
"""


_json = JsonOutputParser()

_STACK_FALLBACK = {"primary_language": "Python", "stack": ["Python"], "is_fullstack": False}


def _detect_stack(query: str) -> dict:
    chain = PromptTemplate(template=DETECT_STACK_PROMPT, input_variables=["query"]) | get_llm() | _json
    try:
        return chain.invoke({"query": query})
    except Exception:
        return _STACK_FALLBACK


def scaffolding_node(state: AgentState) -> AgentState:
    print("Scaffolding node executing...")
    summary = state.get('summary', '')
    query = state.get('query', '')

    if not summary:
        return {"scaffold": [], "language": "Python"}

    stack_info = _detect_stack(query)
    language = stack_info.get("primary_language", "Python")
    stack_label = ", ".join(stack_info.get("stack", [language]))
    print(f"[Scaffold] Stack: {stack_label} | fullstack: {stack_info.get('is_fullstack')}")

    chain = (
        PromptTemplate(template=SCAFFOLDING_PROMPT, input_variables=["query", "summary", "stack"])
        | get_llm(task="code")
        | _json
    )

    try:
        scaffold = chain.invoke({"query": query, "summary": summary, "stack": stack_label})
        if not isinstance(scaffold, list):
            scaffold = []
    except Exception as e:
        print(f"[Scaffold] Failed to parse LLM JSON: {e}")
        scaffold = []

    for entry in scaffold:
        print(f"[Scaffold] {entry.get('filepath')} — {entry.get('purpose', '')}")

    return {"scaffold": scaffold, "language": language}
