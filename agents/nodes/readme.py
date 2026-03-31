from agents import get_llm
from agents.state import AgentState


def readme_node(state: AgentState) -> AgentState:
    print("README node executing...")
    llm = get_llm()

    tasks_md = "\n".join([
        f"- [ ] **{t.get('title', '')}** ({t.get('priority', '')} priority, {t.get('duration_minutes', '')} min): {t.get('description', '')}"
        for t in state.get('tasks', [])
    ])

    response = llm.invoke(
        f"Generate a professional README.md for a project based on this research.\n\n"
        f"Project topic: {state['query']}\n\n"
        f"Research summary:\n{state.get('summary', '')}\n\n"
        f"Tasks/Roadmap:\n{tasks_md}\n\n"
        f"Structure: 1) Project Description, 2) Setup, 3) Tasks / Roadmap checklist.\n"
        f"Use proper Markdown formatting."
    )

    return {"readme": response.content}
