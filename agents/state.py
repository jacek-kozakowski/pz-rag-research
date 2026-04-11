from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    local_result: dict
    web_result: dict
    summary: str
    tasks: list[dict]
    mode: Literal['project', 'learning']
    create_repo: bool
    use_calendar: bool
    github_issues: list[dict]
    readme: str
    notes: str
    calendar_events: list[dict]
    intent: Literal['local_files', 'research']
