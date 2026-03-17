from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents import get_llm
from langdetect import detect

PROMPT_TEMPLATE = """
You are a research query planner. Given a user goal, extract the core topic and generate optimized search queries.

User goal: {query}
Detected language: {language}

Instructions:
- Extract ONLY the technical/educational topic from the user's message, ignore deadlines and personal context
- rag_query: specific technical terms for searching local documents, in {language}
- web_query: precise technical search query in English, include specific technologies, APIs, frameworks mentioned

Examples:
User: "I want to learn Bluetooth programming on Windows by Thursday"
rag_query: "Bluetooth programowanie Windows API"
web_query: "Bluetooth programming Windows Sockets API tutorial C Python"

User: "Chcę się przygotować do kolokwium z systemów operacyjnych"  
rag_query: "systemy operacyjne procesy wątki pamięć"
web_query: "operating systems processes threads memory management exam"

Respond ONLY with valid JSON, no other text:
{{
    "rag_query": "...",
    "web_query": "..."
}}"""

def plan_queries(query: str) -> dict:
    llm = get_llm(task="query_planner")
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["query", "language"],
    )
    chain = prompt | llm | JsonOutputParser()

    result = chain.invoke({"query": query, "language": _get_query_language(query)})

    return result


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