from langchain_core.tools import tool


# @tool
# def list_indexed_documents() -> str:
#     """
#     Lists all documents available in the knowledge base.
#     Use this FIRST to check what documents are available before searching.
#     """
#     from rag.vector_storage import get_indexed_files
#     files = get_indexed_files()
#     if not files:
#         return "No documents indexed."
#
#     return 'Indexed documents:\n' + '\n'.join(f"- {f}" for f in files)


@tool
def search_local_documents_tool(query: str, topics: list[str] = []) -> dict:
    """
    Searches local documents using the provided RAG query.
    Automatically generates optimized RAG queries and searches local documents.
    Pass topics from decompose_topic_tool if available for more targeted search.
    Use when you need to find relevant information in local documents.
    """
    from research.local_researcher import ask_local
    from research.query_planner import plan_rag_queries, plan_rag_queries_from_topics

    if topics:
        rag_queries = plan_rag_queries_from_topics(query, topics)
    else:
        rag_queries = plan_rag_queries(query)
    return ask_local(query, rag_queries)

@tool
def search_web_tool(query: str) -> dict:
    """
    Searches the web using the provided web query.
    Automatically generates an optimized web query and searches the web.
    Use this AFTER search_local_documents to fill gaps, get additional context,
    or when local documents don't contain sufficient information on the topic.
    """
    from research.web_researcher import web_search
    from research.query_planner import plan_web_query

    web_queries = plan_web_query(query)
    return web_search(web_queries)


@tool
def decompose_topic_tool(query: str) -> list[str]:
    """
    Decomposes query into subtopics.
    Use this FIRST to break down complex topics into smaller, more manageable pieces.
    """
    from research.topic_decomposition import decompose_topic
    return decompose_topic(query)


@tool
def find_relevant_sources_tool(query: str) -> dict[str, int]:
    """
    Finds which local documents are most relevant to the query.
    Returns {filename: hit_count} — higher count means more relevant.
    Use this when the user asks about a specific course or subject,
    to identify which files belong to that course before loading them.
    """
    from rag.vector_storage import find_relevant_sources
    sources = find_relevant_sources(query, k=50)
    print(sources.keys())
    return sources


@tool
def load_full_documents_tool(source_files: list[str]) -> str:
    """
    Loads the complete text content of the specified files.
    Use this after find_relevant_sources_tool to get full document content
    instead of RAG chunks — gives more complete notes.
    Returns full text of each file with filename headers.
    """
    from rag.minio_storage import load_full_documents
    texts =  load_full_documents(source_files)
    return "\n\n".join(texts)
