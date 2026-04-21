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

Project summary:
{summary}

Tasks (JSON):
{tasks}

For each task generate one source file that provides a skeleton implementation in {language}.
Rules:
- Filepath must be a valid relative path with the correct extension for {language}
- Code must be syntactically valid {language} with class/function stubs and docstrings/comments
- task_title must exactly match the title from the tasks list

Return a JSON array. Each element must have exactly these keys:
  "task_title"  – exact title from tasks
  "filepath"    – relative path of the file
  "code"        – {language} source code

Return ONLY valid JSON. No markdown, no explanations.
"""


def _detect_language(query: str) -> str:
    llm = get_llm()
    prompt = PromptTemplate(template=DETECT_LANGUAGE_PROMPT, input_variables=["query"])
    response = (prompt | llm).invoke({"query": query})
    return response.content.strip() or "Python"


def scaffolding_node(state: AgentState) -> AgentState:
    print("Scaffolding node executing...")
    tasks = state.get('tasks', [])
    summary = state.get('summary', '')

    if not tasks:
        return {"scaffold": [], "language": "Python"}

    language = state.get('language') or _detect_language(state.get('query', ''))
    print(f"[Scaffold] Detected language: {language}")

    llm = get_llm()
    prompt = PromptTemplate(
        template=SCAFFOLDING_PROMPT,
        input_variables=["language", "summary", "tasks"]
    )
    chain = prompt | llm
    response = chain.invoke({
        "language": language,
        "summary": summary,
        "tasks": json.dumps(tasks, ensure_ascii=False)
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
        print(f"[Scaffold] {entry.get('filepath')} ← {entry.get('task_title')}")

    return {"scaffold": scaffold, "language": language}
