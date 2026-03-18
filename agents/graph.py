from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.local_researcher import ask_local
from agents.web_researcher import web_search
from agents.summarizer import summarize
from agents.query_planner import plan_queries
from agents.planner import plan_task
from agents.exporter import export_to_md


def query_planner_node(state: AgentState) -> AgentState:
    print("Query planner node executing...")
    queries = plan_queries(state['query'])
    return {
        "rag_query": queries['rag_query'],
        "web_query": queries['web_query']
    }

def local_search_node(state: AgentState) -> AgentState:
    print("Local search node executing...")
    rag_queries = state.get('rag_query', [state.get('rag_query', '')])
    local_result = ask_local(state['query'], rag_queries, k=3)
    return {
        "local_result": local_result
    }

def web_search_node(state: AgentState) -> AgentState:
    print("Web search node executing...")
    web_result = web_search(state['web_query'])
    return {
        "web_result": web_result
    }

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
    graph.add_node('query_planner', query_planner_node)
    graph.add_node('local_research', local_search_node)
    graph.add_node('web_research', web_search_node)
    graph.add_node('summarization', summarization_node)
    graph.add_node('task_planner', task_planner_node)
    graph.add_node('exporter', exporter_node)

    graph.set_entry_point('query_planner')
    graph.add_edge('query_planner', 'local_research')
    graph.add_edge('local_research', 'web_research')
    graph.add_edge('web_research', 'summarization')
    graph.add_edge('summarization', 'task_planner')
    graph.add_edge('task_planner', 'exporter')
    graph.add_edge('exporter', END)

    return graph.compile()