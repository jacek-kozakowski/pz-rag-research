from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents import get_llm
from langdetect import detect

PROMPT_TEMPLATE_RAG = """
You are a research query planner. Given a user goal, extract the core topic and generate optimized search queries.
These queries should be optimized for rag search in local documents.

User goal: {query}
Detected language: {language}

Instructions:
- Extract ONLY the technical/educational topic from the user's message, ignore deadlines and personal context
- rag_query: list of maximum 5 (less if there is no need for 5) specific technical terms for searching local documents, in {language}

Examples:
User: "I want to learn Bluetooth programming on Windows by Thursday"
rag_query: "[Bluetooth, programowanie Windows API]"

User: "Chcę się przygotować do kolokwium z systemów operacyjnych"  
rag_query: "[systemy operacyjne, procesy, wątki, pamięć]"

Respond ONLY with valid JSON, no other text:
{{
    "rag_query": "["...", "...", "..."]",
}}"""

PROMPT_TEMPLATE_WEB = """
You are a research query planner. Given a user goal, extract the core topic and generate optimized search queries.
These queries should be optimized for web search.
User goal: {query}

Instructions:
- Extract ONLY the technical/educational topic from the user's message, ignore deadlines and personal context
- web_query: precise technical search query in English, include specific technologies, APIs, frameworks mentioned

Examples:
User: "I want to learn Bluetooth programming on Windows by Thursday"
web_query: "Bluetooth programming Windows Sockets API tutorial C Python"

User: "Chcę się przygotować do kolokwium z systemów operacyjnych"  
web_query: "operating systems processes threads memory management exam"

Respond ONLY with valid JSON, no other text:
{{
    "web_query": "..."
}}"""

def plan_rag_queries(query: str) -> list[str]:
    llm = get_llm(task="query_planner")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE_RAG,
        input_variables=["query", "language"],
    )
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({"query": query, "language": _get_query_language(query)})

    return result['rag_query']

def plan_web_query(query: str) -> str:
    llm = get_llm(task="query_planner")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE_WEB,
        input_variables=["query"]
    )

    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"query": query})
    return result['web_query']


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