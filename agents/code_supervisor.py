from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from agents import get_llm
from code.code_tools import generate_code, generate_tests, generate_documentation
from langchain_core.tools import tool

_code_store = {"code": "", "tests": "", "docs": ""}

@tool
def write_code(requirements: str, language: str, context: str = "") -> str:
    """Writes implementation code for the given requirements. No tests, no examples, no docs."""
    print (f"Writing code for {requirements[:20]} in {language}")
    result = generate_code(requirements, language, context)
    _code_store["code"] = result
    return result

@tool
def write_tests(code: str, framework: str ="pytest", context: str = "") -> str:
    """Writes tests for the provided code using the specified framework."""
    print(f"Writing tests for {code[:20]} using {framework}")
    result = generate_tests(code, framework, context)
    _code_store["tests"] = result
    return result

@tool
def write_documentation(code: str, context: str = "") -> str:
    """Writes documentation for the provided code."""
    print(f"Writing documentation for {code[:20]}")
    result = generate_documentation(code, context)
    _code_store["docs"] = result
    return result


@tool
def retrieve_code_context(query: str) -> dict:
    """
    Searches the indexed codebase for relevant code snippets.
    Use this FIRST to find existing code before generating new code.
    """
    from research.local_researcher import ask_local
    from research.query_planner import plan_rag_queries_code

    rag_queries = plan_rag_queries_code(query)
    return ask_local(query, rag_queries, collection_type="code")

def build_code_supervisor():
    _code_store["code"] = ""
    _code_store["tests"] = ""
    _code_store["docs"] = ""
    # ← przenieś tworzenie agentów tutaj
    code_writer = create_react_agent(
        model=get_llm(task="code"),
        tools=[retrieve_code_context, write_code],
        name="code_writer",
        prompt="You are a code writer. First use retrieve_code_context to understand existing codebase, then write clean production-ready code."
    )

    test_writer = create_react_agent(
        model=get_llm(task="code"),
        tools=[write_tests],
        name="test_writer",
        prompt="You are a test writer. Write comprehensive unit tests for the provided code."
    )

    doc_writer = create_react_agent(
        model=get_llm(task="code"),
        tools=[write_documentation],
        name="doc_writer",
        prompt="You are a documentation writer. Write clear technical documentation in Markdown."
    )

    supervisor = create_supervisor(
        agents=[code_writer, test_writer, doc_writer],
        model=get_llm(task="task_planner"),
        prompt="""You are a code generation supervisor.
        First analyze the user's query and determine the task they want to accomplish.
        Your workflow:
        1. Assign to code_writer to write the implementation 
        2. Assign to test_writer to write tests for the generated code
        3. Assign to doc_writer to write documentation
        4. When all three are done, return FINISH
        
        Always follow this order: code → tests → documentation.""",
        output_mode="full_history"
    )
    return supervisor.compile()