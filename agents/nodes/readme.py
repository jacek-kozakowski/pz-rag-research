from agents import get_llm
from agents.state import AgentState
from langchain_core.prompts import PromptTemplate

README_TEMPLATE = """
Generate a professional README.md for a project based on this research.
Project topic: {query}
Research summary:
{summary}
Tasks/Roadmap:
{tasks}
Structure: 1) Project Description, 2) Setup, 3) Tasks / Roadmap checklist.
Use proper Markdown formatting.
"""


def readme_node(state: AgentState) -> AgentState:
    print("README node executing...")
    llm = get_llm()

    tasks_md = "\n".join([
        f"- [ ] **{t.get('title', '')}** ({t.get('priority', '')} priority, {t.get('duration_minutes', '')} min): {t.get('description', '')}"
        for t in state.get('tasks', [])
    ])
    prompt = PromptTemplate(
        template=README_TEMPLATE,
        input_variables=["query", "summary", "tasks"]
    )
    chain = prompt | llm
    response = chain.invoke({"query": state['query'], "summary": state.get('summary', ''), "tasks": tasks_md})

    return {"readme": response.content}
