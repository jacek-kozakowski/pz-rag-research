from langchain_core.messages import HumanMessage, SystemMessage

from agents import get_llm
from agents.state import AgentState

DETECT_INTENT_PROMPT = """You are a classifier. Decide if the user wants notes from their own uploaded course files, or a general research on a topic.

Reply with ONLY one word:
- "local_files" — if the user asks for notes/summary from a specific course, subject, or their own materials (e.g. "make notes from PEA", "summarize my operating systems lectures", "notatka z algorytmów")
- "research" — if the user asks a general question or wants to learn about a topic without implying they have specific files (e.g. "how does quicksort work", "explain machine learning")
"""


def detect_intent_node(state: AgentState) -> AgentState:
    print("Detect intent node executing...")
    llm = get_llm(task="query_planner")
    response = llm.invoke([
        SystemMessage(content=DETECT_INTENT_PROMPT),
        HumanMessage(content=state["query"])
    ])
    intent = response.content.strip().lower()
    if "local_files" in intent:
        intent = "local_files"
    else:
        intent = "research"
    print(f"Detected intent: {intent}")
    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    return state.get("intent", "research")
