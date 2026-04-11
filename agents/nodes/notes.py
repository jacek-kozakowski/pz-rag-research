from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from agents import get_llm
from agents.state import AgentState
from research.research_tools import search_local_documents_tool, search_web_tool

notes_tools = [search_local_documents_tool, search_web_tool]

EXTRACT_SYSTEM_PROMPT = """You are a precise content extractor. Your job is to extract ALL substantive academic content from a course document.

Extract thoroughly:
- Every key concept, definition, and theorem with exact wording
- All algorithms with their full steps, pseudocode, and time/space complexity
- All formulas and mathematical relationships
- All examples, including numerical ones
- Comparisons between methods (pros, cons, when to use)
- Edge cases, limitations, and special cases mentioned

SKIP completely:
- Lecturer name, email, office hours
- Course schedule, meeting times, deadlines
- Administrative announcements
- Any organizational/logistical information

Be specific and detailed — preserve exact names, numbers, complexity classes, and technical details.
Do NOT add outside knowledge. Only extract what is in the document.
"""

NOTES_SYSTEM_PROMPT = """You are a learning notes specialist. Your job is to create comprehensive, in-depth learning notes from course materials.
These notes will be used by a student to prepare for an exam and must be thorough enough to study from alone.

The provided extracts are the course material — treat them as the single source of truth.

Notes structure:
1. **Key Concepts** — precise definitions exactly as defined in the material, with formal notation where present
2. **Detailed Explanations** — for EACH major topic: full explanation of how and why it works, all steps/mechanisms, specific values, formulas, or rules mentioned in the material
3. **Comparisons** — compare related concepts, methods or approaches against each other where relevant
4. **Common Mistakes & Pitfalls** — typical errors and misconceptions from the material
5. **Practical Examples** — concrete examples from the documents (numerical, real-world, case studies)
6. **Flashcards** — at least 10 Q&A pairs covering definitions, key facts, and distinctions
7. **Review Questions** — at least 5 open-ended questions that test deep understanding

Rules:
- Every section must be grounded in the provided extracts — no generic filler
- Cover ALL topics from the material, not just the most obvious ones
- Be specific: exact names, numbers, formulas, classifications as they appear in the material
- Do NOT abbreviate, summarize or skip any topic — write out everything in full
- If a topic has subtopics, cover each subtopic separately and in depth
- Notes must be in the same language as the user query
- Use clear Markdown formatting
"""

NOTES_RESEARCH_SYSTEM_PROMPT = """You are a learning notes specialist. Your job is to create comprehensive, in-depth learning notes.
These notes will be used by a student to learn the given topic and should be a good foundation for further study.
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


def _notes_from_local_files(state: AgentState) -> AgentState:
    from rag.vector_storage import find_relevant_sources
    from rag.minio_storage import load_full_documents

    llm = get_llm()

    # Find relevant files
    selected = find_relevant_sources(state['query'])
    print(f"Selected files for full load: {selected}")
    if not selected:
        print("No relevant sources found, aborting notes generation")
        return {"notes": "No relevant documents found for this topic."}

    # MAP — extract then write notes for each file in parallel
    texts = load_full_documents(selected)
    query = state['query']

    def process_file(args):
        text, filename = args
        print(f"Extracting content from {filename}...")
        extract = get_llm().invoke([
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=f"Topic: {query}\n\nDocument:\n{text}")
        ]).content

        print(f"Writing notes for {filename}...")
        notes = get_llm(task="notes").invoke([
            SystemMessage(content=NOTES_SYSTEM_PROMPT),
            HumanMessage(content=f"Topic: {query}\n\nExtracted content:\n\n{extract}")
        ]).content

        return filename, notes

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, zip(texts, selected)))

    all_notes = [f"## {filename}\n\n{notes}" for filename, notes in results]
    return {"notes": "\n\n---\n\n".join(all_notes)}


def _notes_from_research(state: AgentState) -> AgentState:
    llm = get_llm().bind_tools([search_local_documents_tool, search_web_tool])

    local_result = state.get('local_result', {})
    web_result = state.get('web_result', {})

    context = f"Topic: {state['query']}\n\n"
    context += f"Summary:\n{state.get('summary', '')}\n\n"
    context += f"Local documents findings:\n{local_result.get('answer', 'No local data')}\n\n"
    context += f"Web research findings:\n{web_result.get('answer', 'No web data')}"

    messages = [
        SystemMessage(content=NOTES_RESEARCH_SYSTEM_PROMPT),
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


def notes_node(state: AgentState) -> AgentState:
    print("Notes node executing...")
    intent = state.get('intent', 'research')
    if intent == 'local_files':
        print("Using local files map-reduce approach")
        return _notes_from_local_files(state)
    else:
        print("Using research approach")
        return _notes_from_research(state)
