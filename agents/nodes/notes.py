from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from agents import get_llm
from agents.state import AgentState
from research.research_tools import search_local_documents_tool, search_web_tool

notes_tools = [search_local_documents_tool, search_web_tool]

NOTES_SYSTEM_PROMPT = """You are a learning notes specialist. Your job is to create comprehensive, in-depth learning notes.
This notes will be used by a student to learn the given topic and should be a good foundation for further study.
You have access to search tools — use them to expand on concepts that need deeper explanation beyond what the summary provides.

Your workflow:
1. Read the summary and the raw research data (local documents + web findings) provided
2. List the key concepts from the data
3. For each key concept, use search_web_tool to find deeper explanations, examples and common mistakes
4. Use search_local_documents_tool to find any additional relevant material from local documents
5. Generate the final notes using all collected information

Notes structure:
1. Key Concepts - comprehensive definitions and explanations of the main concepts
2. Detailed Explanations - deep dive with examples, mechanisms (not just what, but why and how)
3. Common Mistakes & Pitfalls - typical errors and misconceptions
4. Practical Examples - real-world usage examples
5. Flashcards - Q&A pairs for memorization (at least 5)
6. Review Questions - open-ended questions to test understanding (at least 3)

Rules:
- Do not include code snippets unless the user explicitly asked for them
- Focus on depth of understanding, not surface-level definitions
- Notes must be in the same language as the user query
- Use clear Markdown formatting
"""

def notes_node(state: AgentState) -> AgentState:
    print("Notes node executing...")
    llm = get_llm().bind_tools(notes_tools)

    local_result = state.get('local_result', {})
    web_result = state.get('web_result', {})

    context = f"Topic: {state['query']}\n\n"
    context += f"Summary:\n{state.get('summary', '')}\n\n"
    context += f"Local documents findings:\n{local_result.get('answer', 'No local data')}\n\n"
    context += f"Web research findings:\n{web_result.get('answer', 'No web data')}"

    messages = [
        SystemMessage(content=NOTES_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    while True:
        response = llm.invoke(messages)
        messages.append(response)

        if not getattr(response, 'tool_calls', None):
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            if tool_name == 'search_local_documents_tool':
                result = search_local_documents_tool.invoke(tool_call['args'])
            elif tool_name == 'search_web_tool':
                result = search_web_tool.invoke(tool_call['args'])
            else:
                result = f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

    return {"notes": response.content}
