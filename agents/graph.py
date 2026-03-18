from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from agents import get_llm
from agents.state import AgentState
from agents.summarizer import summarize
from agents.planner import plan_task
from agents.exporter import export_to_md
from agents.tools import search_local_documents, search_web

tools = [ search_local_documents, search_web]


SYSTEM_PROMPT = """You are a research assistant with access to tools.
Available tools:
{tools}
Your workflow:
1. First call search_local_documents to find relevant information locally,
2. If more information needed or topic requires current data, call search_web to find relevant information from the web
3. When you have enough information, stop calling tools.

Be thorough - collect as much relevant information as possible."""

def research_agent_node(state: AgentState) -> AgentState:
    print("Research agent node executing...")

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["tools"]
    )
    system_message = SystemMessage(content=prompt.format(tools=tools))

    llm = get_llm().bind_tools(tools=tools)
    messages = [system_message] + state['messages']
    response = llm.invoke(messages)

    return {
        "messages": [response]
    }

def tools_node_handler(state: AgentState) -> AgentState:
    """Executes nodes and saves the results in the state."""
    from langchain_core.messages import ToolMessage
    import json

    last_message = state['messages'][-1]
    tool_results = []
    local_results = state.get('local_result', {})
    web_results = state.get('web_result', {})

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        if tool_name == 'list_indexed_documents':
            result = list_indexed_documents.invoke(tool_args)
        elif tool_name == 'search_local_documents':
            result = search_local_documents.invoke(tool_args)
            local_results = result
        elif tool_name == 'search_web':
            result = search_web.invoke(tool_args)
            web_results = result
        else:
            result = f"Unknown tool. Tool '{tool_name}' not found."

        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call['id']
            )
        )
    return {
        "messages": tool_results,
        "local_result": local_results,
        "web_result": web_results
    }

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "summarization"


def summarization_node(state: AgentState) -> AgentState:
    print("Summarization node executing...")
    summary_result = summarize(state['query'], state['local_result'], state['web_result'])
    return {
        "summary": summary_result['summary']
    }


def task_planner_node(state: AgentState) -> AgentState:
    print(f"Task planner node executing...")
    tasks = plan_task(state['summary'], state['query'])
    return {
        "tasks": tasks
    }

def exporter_node(state: AgentState) -> AgentState:
    print("Exporter node executing...")
    path = export_to_md(state['query'], state['summary'], state['tasks'])
    return {
        "report_path": path
    }

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('research_agent', research_agent_node)
    graph.add_node('tools', tools_node_handler)
    graph.add_node('summarization', summarization_node)
    graph.add_node('task_planner', task_planner_node)
    graph.add_node('exporter', exporter_node)

    graph.set_entry_point('research_agent')
    graph.add_conditional_edges('research_agent', should_continue)
    graph.add_edge('tools', 'research_agent')
    graph.add_edge('summarization', 'task_planner')
    graph.add_edge('task_planner', 'exporter')
    graph.add_edge('exporter', END)

    return graph.compile()