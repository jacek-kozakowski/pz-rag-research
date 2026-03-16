from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    web_query: str
    rag_query: str
    messages: Annotated[list, add_messages]
    local_result: dict
    web_result: dict
    summary: str
    tasks: list[dict]