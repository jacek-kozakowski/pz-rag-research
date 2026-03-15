from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.local_researcher import ask_local
from agents.web_researcher import web_search
from agents.summarizer import summarize


def local_search_node(state: AgentState) -> AgentState:
    print("Local search node executing...")
    local_result = ask_local(state['query'])
    return {
        **state,
        "local_result": local_result
    }

def web_search_node(state: AgentState) -> AgentState:
    print("Web search node executing...")
    web_result = web_search(state['query'])
    return {
        **state,
        "web_result": web_result
    }

def summarization_node(state: AgentState) -> AgentState:
    print("Summarization node executing...")
    summary_result = summarize(state['query'], state['local_result'], state['web_result'])
    return {
        **state,
        "summary": summary_result['summary']
    }

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('local_research', local_search_node)
    graph.add_node('web_research', web_search_node)
    graph.add_node('summarization', summarization_node)

    graph.set_entry_point('local_research')
    graph.add_edge('local_research', 'web_research')
    graph.add_edge('web_research', 'summarization')
    graph.add_edge('summarization', END)

    return graph.compile()