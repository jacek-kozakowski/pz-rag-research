from agents.state import AgentState
from research.summarizer import summarize
from research.planner import plan_task, plan_task_from_notes


def summarization_node(state: AgentState) -> AgentState:
    print("Summarization node executing...")
    summary_result = summarize(
        state['query'],
        state.get('local_result', {}),
        state.get('web_result', {})
    )
    return {"summary": summary_result['summary']}


def task_planner_node(state: AgentState) -> AgentState:
    print("Task planner node executing...")
    if state.get('intent') == 'local_files':
        tasks = plan_task_from_notes(state.get('notes', ''), state['query'])
    else:
        tasks = plan_task(state.get('summary', ''), state['query'])
    return {"tasks": tasks}
