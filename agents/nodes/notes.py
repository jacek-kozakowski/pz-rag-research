import re

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from agents import get_llm
from agents.state import AgentState
from research.research_tools import search_local_documents_tool, search_web_tool


def _fix_latex(text: str) -> str:
    """Normalize LaTeX delimiters to $...$ and $$...$$ for Streamlit KaTeX."""
    # \(...\) -> $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    # \[...\] -> $$...$$
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    # Bare \begin{env}...\end{env} not already inside $$ → $$...$$
    text = re.sub(
        r'(?<!\$)\\begin\{([^}]+)\}(.*?)\\end\{\1\}(?!\$)',
        lambda m: f'$$\\begin{{{m.group(1)}}}{m.group(2)}\\end{{{m.group(1)}}}$$',
        text, flags=re.DOTALL
    )
    return text

_MATH_FORMAT = """MATH FORMATTING — the renderer supports ONLY $...$ (inline) and $$...$$ (display block):
  - WRONG: \\(O(n \\log n)\\)  →  RIGHT: $O(n \\log n)$
  - WRONG: \\[f(x) = x^2\\]   →  RIGHT: $$f(x) = x^2$$
  - WRONG: 𝑛, 𝒪, 𝑥 (Unicode math)  →  RIGHT: $n$, $\\mathcal{O}$, $x$
  - WRONG: O(n log n) plain text  →  RIGHT: $O(n \\log n)$
  - Every variable, formula, complexity, subscript, superscript MUST be inside $...$"""

EXTRACT_SYSTEM_PROMPT = f"""You are a precise content extractor. Extract ALL substantive academic content from the document.

Extract:
- Every key concept, definition, and theorem with exact wording
- All algorithms with full steps, pseudocode, and time/space complexity
- All formulas and mathematical relationships
- All examples, including numerical ones
- Comparisons between methods (pros, cons, when to use)
- Edge cases, limitations, and special cases

SKIP: lecturer info, course schedule, administrative announcements.

PDF MATH REPAIR — fix mangled math during extraction:
- "x\\na\\nx\\na" or "x a x a" → $x_a$
- "n\\n2\\nn\\n2" or "n 2 n 2" → $n^2$
- "k\\nk-elementowy" → "$k$-elementowy"
- "O(b d+1 )" → $O(b^{{d+1}})$
- Formulas written twice in slightly different forms — keep once in LaTeX

{_MATH_FORMAT}

Only extract what is in the document — do NOT add outside knowledge."""

NOTES_SYSTEM_PROMPT = f"""You are a learning notes specialist creating exam-ready notes from course materials.
The provided extracts are the single source of truth — do NOT add facts, definitions, or examples not present in them.
If there is no relevant information in the extracts, write "Not enough information" or "N/A".

Notes structure (translate ALL headings to match the language of the query):
1. **Key Concepts** — precise definitions exactly as in the material
2. **Detailed Explanations** — how and why each topic works, all steps and formulas from the material
3. **Comparisons** — compare related concepts where relevant
4. **Common Mistakes & Pitfalls** — errors and misconceptions from the material
5. **Practical Examples** — concrete examples from the documents
6. **Flashcards** — at least 10 Q&A pairs
7. **Review Questions** — at least 5 open-ended questions

Rules:
- Grounded strictly in the extracts — no generic filler or invented content
- Cover ALL topics from the material in full — do not skip or abbreviate
- Notes in the same language as the user query
- Do NOT add a title heading at the top
- Use clear Markdown formatting
{_MATH_FORMAT}"""


def _make_research_prompt(web_enabled: bool) -> str:
    if web_enabled:
        tools_line = "You have access to search_local_documents_tool and search_web_tool."
        workflow = (
            "1. Read the provided summary and research data\n"
            "2. Use search_local_documents_tool to find relevant material in local documents\n"
            "3. Use search_web_tool to supplement with additional explanations and examples\n"
            "4. Write notes using ONLY information gathered above — do not invent facts"
        )
    else:
        tools_line = "You have access to search_local_documents_tool only — do NOT search the web."
        workflow = (
            "1. Read the provided summary and research data\n"
            "2. Use search_local_documents_tool to find relevant material in local documents\n"
            "3. Write notes using ONLY information gathered above — do not invent facts"
        )
    return f"""You are a learning notes specialist creating comprehensive notes for a student.
Only use information found through the provided tools and research data — do not add outside knowledge or invented content.
{tools_line}

Workflow:
{workflow}

Notes structure (translate ALL headings to match the language of the query):
1. Key Concepts — precise definitions
2. Detailed Explanations — how and why it works, with examples
3. Common Mistakes & Pitfalls
4. Practical Examples
5. Flashcards — Q&A pairs (at least 5)
6. Review Questions (at least 3)

Rules:
- No code snippets unless explicitly requested
- Notes in the same language as the query
- Use clear Markdown formatting
{_MATH_FORMAT}"""


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
            HumanMessage(content=f"Topic: {query}\nSource file: {filename}\n\nExtracted content:\n\n{extract}")
        ]).content

        return filename, _fix_latex(notes)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, zip(texts, selected)))

    all_notes = [f"## {filename}\n\n{notes}" for filename, notes in results]
    return {"notes": "\n\n---\n\n".join(all_notes)}


def _notes_from_research(state: AgentState) -> AgentState:
    web_enabled = state.get('web_enabled', True)
    tools = [search_local_documents_tool, search_web_tool] if web_enabled else [search_local_documents_tool]
    llm = get_llm().bind_tools(tools)
    prompt = _make_research_prompt(web_enabled)

    local_result = state.get('local_result', {})
    web_result = state.get('web_result', {})

    context = f"Topic: {state['query']}\n\n"
    context += f"Summary:\n{state.get('summary', '')}\n\n"
    context += f"Local documents findings:\n{local_result.get('answer', 'No local data')}\n\n"
    if web_enabled:
        context += f"Web research findings:\n{web_result.get('answer', 'No web data')}"

    messages = [
        SystemMessage(content=prompt),
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
            elif tool_name == 'search_web_tool' and web_enabled:
                result = search_web_tool.invoke(tool_call['args'])
            else:
                result = f"Unknown tool: {tool_name}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))

    return {"notes": _fix_latex(response.content)}


def notes_node(state: AgentState) -> AgentState:
    print("Notes node executing...")
    intent = state.get('intent', 'research')
    if intent == 'local_files':
        print("Using local files map-reduce approach")
        return _notes_from_local_files(state)
    else:
        print("Using research approach")
        return _notes_from_research(state)
