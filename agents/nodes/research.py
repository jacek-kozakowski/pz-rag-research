from langchain_core.messages import SystemMessage, ToolMessage

from agents import get_llm
from agents.state import AgentState
from research.research_tools import search_local_documents_tool, search_web_tool

search_tools = [search_local_documents_tool, search_web_tool]

SYSTEM_PROMPT_SEARCH = """You are a research assistant with access to tools.

Your workflow:
1. First call search_local_documents to find relevant information locally,
2. If more information needed or topic requires current data, call search_web to find relevant information from the web
3. When you have enough information, stop calling tools.

Be thorough - collect as much relevant information as possible."""


def research_agent_node(state: AgentState) -> AgentState:
    print("Research agent node executing...")
    system_message = SystemMessage(content=SYSTEM_PROMPT_SEARCH)
    llm = get_llm().bind_tools(tools=search_tools)
    messages = [system_message] + state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


def research_tools_node_handler(state: AgentState) -> AgentState:
    """Executes tool calls and saves the results in the state."""
    last_message = state['messages'][-1]
    tool_results = []
    local_results = state.get('local_result', {})
    web_results = state.get('web_result', {})

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        if tool_name == 'search_local_documents':
            result = search_local_documents_tool.invoke(tool_args)
            local_results = result
        elif tool_name == 'search_web':
            result = search_web_tool.invoke(tool_args)
            web_results = result
        else:
            result = f"Unknown tool. Tool '{tool_name}' not found."

        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call['id'])
        )

    return {
        "messages": tool_results,
        "local_result": local_results,
        "web_result": web_results
    }


def should_continue_research(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "summarization"
