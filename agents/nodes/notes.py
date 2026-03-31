from agents import get_llm
from agents.state import AgentState


def notes_node(state: AgentState) -> AgentState:
    print("Notes node executing...")
    llm = get_llm()

    response = llm.invoke(
        f"Generate comprehensive learning notes from this research summary.\n\n"
        f"Topic: {state['query']}\n\n"
        f"Summary:\n{state.get('summary', '')}\n\n"
        f"Structure:\n"
        f"1. Key Concepts - list and explain the main concepts\n"
        f"2. Detailed Explanations - deeper explanation with examples\n"
        f"3. Flashcards - Q&A pairs for memorization (at least 5)\n"
        f"4. Review Questions - open-ended questions to test understanding (at least 3)\n\n"
        f"Use clear Markdown formatting."
    )

    return {"notes": response.content}
