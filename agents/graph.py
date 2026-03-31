from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.nodes.research import research_agent_node, research_tools_node_handler, should_continue_research
from agents.nodes.summarization import summarization_node, task_planner_node
from agents.nodes.github_issues import github_issues_node, code_snippets_node
from agents.nodes.readme import readme_node
from agents.nodes.calendar import calendar_node
from agents.nodes.notes import notes_node


def build_project_graph():
    """research_agent → summarization → task_planner → github_issues → code_snippets → readme → END"""
    graph = StateGraph(AgentState)

    graph.add_node('research_agent', research_agent_node)
    graph.add_node('research_tools', research_tools_node_handler)
    graph.add_node('summarization', summarization_node)
    graph.add_node('task_planner', task_planner_node)
    graph.add_node('github_issues', github_issues_node)
    graph.add_node('code_snippets', code_snippets_node)
    graph.add_node('readme', readme_node)

    graph.set_entry_point('research_agent')
    graph.add_conditional_edges('research_agent', should_continue_research, {
        "tools": 'research_tools',
        "summarization": 'summarization'
    })
    graph.add_edge('research_tools', 'research_agent')
    graph.add_edge('summarization', 'task_planner')
    graph.add_edge('task_planner', 'github_issues')
    graph.add_edge('github_issues', 'code_snippets')
    graph.add_edge('code_snippets', 'readme')
    graph.add_edge('readme', END)

    return graph.compile()


def build_learning_graph():
    """research_agent → summarization → task_planner → calendar → notes → END"""
    graph = StateGraph(AgentState)

    graph.add_node('research_agent', research_agent_node)
    graph.add_node('research_tools', research_tools_node_handler)
    graph.add_node('summarization', summarization_node)
    graph.add_node('task_planner', task_planner_node)
    graph.add_node('calendar', calendar_node)
    graph.add_node('notes', notes_node)

    graph.set_entry_point('research_agent')
    graph.add_conditional_edges('research_agent', should_continue_research, {
        "tools": 'research_tools',
        "summarization": 'summarization'
    })
    graph.add_edge('research_tools', 'research_agent')
    graph.add_edge('summarization', 'task_planner')
    graph.add_edge('task_planner', 'calendar')
    graph.add_edge('calendar', 'notes')
    graph.add_edge('notes', END)

    return graph.compile()
