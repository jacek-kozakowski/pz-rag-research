from langchain_core.messages import HumanMessage, SystemMessage

from agents import get_llm

DETECT_MODE_PROMPT = """You are a classifier. Decide if the user wants to BUILD a software project or LEARN about a topic.

Reply with ONLY one word:
- "project" — user wants to build, create, develop, implement, or plan a software project
  Examples: "build a REST API", "create a habit tracker app", "set up FastAPI with React", "I need a CLI tool that..."
- "learning" — user wants to understand, study, get notes, or research a topic
  Examples: "explain quicksort", "notatki z algorytmów", "how does backpropagation work", "summarize my OS lectures", "what is a B-tree"

When in doubt, prefer "learning".
"""

_PROJECT_KEYWORDS = frozenset([
    "build", "create", "develop", "implement", "make", "write", "code",
    "set up", "setup", "deploy", "scaffold", "generate project", "new app",
    "zbuduj", "stwórz", "stworz", "napisz aplikację", "napisz aplikacje", "zrób aplikację", "zrob aplikacje",
])

_LEARNING_KEYWORDS = frozenset([
    "explain", "what is", "how does", "learn", "study", "notes", "notatki",
    "summarize", "understand", "tutorial", "lecture", "course", "wytłumacz", "wytlumacz"
    "omów","omow", "opisz", "jak działa", "jak dziala", "co to jest",
])


def _keyword_hint(query: str) -> str | None:
    q = query.lower()
    if any(kw in q for kw in _PROJECT_KEYWORDS):
        return "project"
    if any(kw in q for kw in _LEARNING_KEYWORDS):
        return "learning"
    return None


def detect_mode(query: str) -> str:
    """Classify query as 'project' or 'learning' using LLM, with keyword pre-check."""
    hint = _keyword_hint(query)

    llm = get_llm(task="query_planner")
    response = llm.invoke([
        SystemMessage(content=DETECT_MODE_PROMPT),
        HumanMessage(content=query),
    ])
    raw = response.content.strip().lower()

    if "project" in raw:
        llm_result = "project"
    else:
        llm_result = "learning"

    # If keyword hint and LLM disagree, trust LLM — keywords are just a signal
    print(f"[detect_mode] keyword_hint={hint}, llm={llm_result} → {llm_result}")
    return llm_result