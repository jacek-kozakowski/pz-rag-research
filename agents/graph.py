from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from agents import get_llm
from agents.state import AgentState
from research.summarizer import summarize
from research.planner import plan_task
from research.exporter import export_to_md
from research.research_tools import search_local_documents_tool, search_web_tool
from research.code_tools import generate_code, generate_tests, generate_documentation
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

    return {
        "messages": [response]
    }

def research_tools_node_handler(state: AgentState) -> AgentState:
    """Executes nodes and saves the results in the state."""
    from langchain_core.messages import ToolMessage
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

def should_continue_research(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "summarization"


def summarization_node(state: AgentState) -> AgentState:
    print("Summarization node executing...")
    summary_result = summarize(
        state['query'],
        state.get('local_result', {}),
        state.get('web_result', {})
    )
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
    # TODO: add export_to_pdf
    return {
        "report_path": path
    }

def generate_code_node(state: AgentState) -> AgentState:
    print("Code generator node executing...")
    code = generate_code(state["query"], language="python")
    return {
        "summary": code
    }

def generate_tests_node(state: AgentState) -> AgentState:
    print("Test generator node executing...")
    tests = generate_tests(state["summary"], framework="pytest")
    return {
        "summary": tests
    }

def generate_documentation_node(state: AgentState) -> AgentState:
    print("Documentation generator node executing...")
    docs = generate_documentation(state["summary"])
    return {
        "summary": docs
    }

def supervisor_node(state: AgentState) -> AgentState:
    print("Supervisor node executing...")
    llm = get_llm()
    prompt = PromptTemplate(
        template="Classify this query as either 'research' or 'code'. Return only one word: research or code. Query: {query}",
        input_variables=["query"]
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    mode = response.content.strip().lower()
    
    # Validation
    if "code" in mode:
        mode = "code"
    else:
        mode = "research"
        
    print(f"Supervisor classified query as: {mode}")
    return {"mode": mode}


def build_research_graph():
    graph = StateGraph(AgentState)
    graph.add_node('research_agent', research_agent_node)
    graph.add_node('research_tools', research_tools_node_handler)
    graph.add_node('summarization', summarization_node)
    graph.add_node('task_planner', task_planner_node)
    graph.add_node('exporter', exporter_node)

    graph.set_entry_point('research_agent')
    graph.add_conditional_edges('research_agent', should_continue_research)
    graph.add_edge('research_tools', 'research_agent')
    graph.add_edge('summarization', 'task_planner')
    graph.add_edge('task_planner', 'exporter')
    graph.add_edge('exporter', END)

    return graph.compile()


def build_code_graph():
    graph = StateGraph(AgentState)
    graph.add_node('generate_code', generate_code_node)
    graph.add_node('generate_tests', generate_tests_node)
    graph.add_node('generate_documentation', generate_documentation_node)

    graph.set_entry_point('generate_code')
    graph.add_edge('generate_code', 'generate_tests')
    graph.add_edge('generate_tests', 'generate_documentation')
    graph.add_edge('generate_documentation', END)

    return graph.compile()


def build_graph():
    graph = StateGraph(AgentState)
    
    # 1. Add supervisor node
    graph.add_node('supervisor', supervisor_node)
    
    # 2. Add sub-graphs as nodes
    graph.add_node('research_flow', build_research_graph())
    graph.add_node('code_flow', build_code_graph())
    
    # 3. Routing
    graph.set_entry_point('supervisor')
    
    def route_to_flow(state: AgentState) -> str:
        return f"{state['mode']}_flow"
        
    graph.add_conditional_edges('supervisor', route_to_flow)
    
    # 4. Exit
    graph.add_edge('research_flow', END)
    graph.add_edge('code_flow', END)
    
    return graph.compile()