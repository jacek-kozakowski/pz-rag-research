from langchain_core.tools import tool


@tool
def list_indexed_documents() -> str:
    """
    Lists all documents available in the knowledge base.
    Use this FIRST to check what documents are available before searching.
    """
    from rag.vector_storage import get_indexed_files
    files = get_indexed_files()
    if not files:
        return "No documents indexed."

    return 'Indexed documents:\n' + '\n'.join(f"- {f}" for f in files)


@tool
def search_local_documents(query: str) -> dict:
    """
    Searches local documents using the provided RAG query.
    Automatically generates optimized RAG and searches local documents.
    Use when you need to find relevant information in local documents.
    """
    from agents.local_researcher import ask_local
    from agents.query_planner import plan_rag_queries

    rag_queries = plan_rag_queries(query)
    return ask_local(query, rag_queries)

@tool
def search_web(query: str) -> dict:
    """
    Searches the web using the provided web query.
    Automatically generates optimized web query and searches the web.
    Use when local documents are not enough or you feel you need to find information from the web.
    """
    from agents.web_researcher import web_search
    from agents.query_planner import plan_web_query

    web_queries = plan_web_query(query)
    return web_search(web_queries)


