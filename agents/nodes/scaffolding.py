import json
import re
from langchain_core.prompts import PromptTemplate

from agents import get_llm
from agents.state import AgentState

DETECT_LANGUAGE_PROMPT = """Identify the primary programming language requested or implied by this project query.
Query: {query}
Rules:
- Return a single word, e.g. Python, TypeScript, Go, Rust, Java, C++, C#, Ruby, PHP
- If unclear, return Python
Return ONLY the language name, nothing else.
"""

SCAFFOLDING_PROMPT = """You are a software architect generating a {language} project scaffold.

Project description:
{summary}

Design a realistic, idiomatic {language} project file structure for this project.
Think about the project as a whole — what modules, packages, and files a real developer would create.

Rules:
- Group related functionality into the same file (e.g. auth routes in one file, not login.py + register.py separately)
- Never create files whose sole purpose is to describe a task or list installation steps
- Use conventional project layout for {language} (e.g. src/, app/, tests/, etc.)
- Each file must contain real, syntactically valid {language} code with function/class stubs
- Aim for 5–15 files that cover the full project structure
- filepath must be a valid relative path with the correct extension for {language}

Return a JSON array. Each element must have exactly these keys:
  "filepath"  – relative path of the file
  "purpose"   – one sentence describing what this file does
  "code"      – {language} source code with stubs

Return ONLY valid JSON. No markdown, no explanations.
"""


def _detect_language(query: str) -> str:
    llm = get_llm()
    prompt = PromptTemplate(template=DETECT_LANGUAGE_PROMPT, input_variables=["query"])
    response = (prompt | llm).invoke({"query": query})
    return response.content.strip() or "Python"


def scaffolding_node(state: AgentState) -> AgentState:
    print("Scaffolding node executing...")
    summary = state.get('summary', '')

    if not summary:
        return {"scaffold": [], "language": "Python"}

    language = state.get('language') or _detect_language(state.get('query', ''))
    print(f"[Scaffold] Detected language: {language}")

    llm = get_llm()
    prompt = PromptTemplate(
        template=SCAFFOLDING_PROMPT,
        input_variables=["language", "summary"]
    )
    chain = prompt | llm
    response = chain.invoke({
        "language": language,
        "summary": summary,
    })

    raw = response.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        scaffold = json.loads(raw)
        if not isinstance(scaffold, list):
            scaffold = []
    except json.JSONDecodeError:
        print(f"[Scaffold] Failed to parse LLM JSON: {raw[:200]}")
        scaffold = []

    for entry in scaffold:
        print(f"[Scaffold] {entry.get('filepath')} — {entry.get('purpose', '')}")

    return {"scaffold": scaffold, "language": language}
