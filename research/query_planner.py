from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents import get_llm
from langdetect import detect

PROMPT_TEMPLATE_RAG = """
You are a research query planner. Given a user goal, extract the core topic and generate optimized search queries.
These queries should be optimized for rag search in local vector database.

User goal: {query}

Instructions:
- Extract ONLY the technical/educational topic from the user's message, ignore deadlines and personal context
- rag_query: list of maximum 5 (less if there is no need for 5) specific technical terms for searching local documents
- Queries should be in the same language as the user query, but if the language is not supported, generate queries in English

Respond ONLY with valid JSON, no other text:
{{
    "rag_query": ["...", "...", "..."],
}}"""

PROMPT_TEMPLATE_WEB = """
You are a research query planner. Given a user goal, extract the core topic and generate optimized search queries.
These queries should be optimized for web search.
User goal: {query}

Instructions:
- Extract ONLY the technical/educational topic from the user's message, ignore deadlines and personal context
- web_query: precise technical search query in English, include specific technologies, APIs, frameworks mentioned

Respond ONLY with valid JSON, no other text:
{{
    "web_query": "..."
}}"""


PROMPT_CODE_RAG = """Generate 3 specific search queries for searching a codebase.
User goal: {query}

Rules:
- Always generate queries in English regardless of input language
- Focus on technical terms, function names, algorithms, patterns
- Each query should be 3-6 words long

Return ONLY a JSON list of 3 strings:
["query1", "query2", "query3"]"""

def plan_rag_queries(query: str) -> list[str]:
    llm = get_llm(task="query_planner")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE_RAG,
        input_variables=["query", "language"],
    )
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({"query": query})
    print("Planned RAG queries:", result['rag_query'])
    return result['rag_query']

def plan_web_query(query: str) -> str:
    llm = get_llm(task="query_planner")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE_WEB,
        input_variables=["query"]
    )

    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"query": query})
    print("Planned web query:", result['web_query'])
    return result['web_query']

def plan_rag_queries_code(query: str) -> list[str]:
    llm = get_llm(task="query_planner")
    prompt = PromptTemplate(
        template=PROMPT_CODE_RAG,
        input_variables=["query"]
    )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"query": query})


def _get_query_language(query: str) -> str:
    try:
        language = detect(query)
        language_map = {
            "pl": "Polish",
            "en": "English",
            "de": "German",
            "fr": "French"
        }
        return language_map.get(language, "English")
    except Exception:
        return "English"