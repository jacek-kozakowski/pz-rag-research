from agents import get_llm
from agents.state import AgentState
from langchain_core.prompts import PromptTemplate


NOTES_TEMPLATE = """
Generate comprehensive learning notes from this research summary.
Topic: {query}
Summary: {summary}
Structure:
1. Key Concepts - list and explain the main concepts
2. Detailed Explanations - deeper explanation with examples
3. Flashcards - Q&A pairs for memorization (at least 5)
4. Review Questions - open-ended questions to test understanding (at least 3)
Use clear Markdown formatting. Do not include code snippets if user did not ask for them. Focus on clarity and depth of understanding.
Notes should be in the same language as the user query.
"""

def notes_node(state: AgentState) -> AgentState:
    print("Notes node executing...")
    llm = get_llm()
    prompt = PromptTemplate(
        template=NOTES_TEMPLATE,
        input_variables=["query", "summary"]
    )
    chain = prompt | llm
    response = chain.invoke({'query': state['query'], 'summary': state.get('summary', '')})

    return {"notes": response.content}
