from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents import get_llm
from langdetect import detect

PROMPT_TEMPLATE = """
You are a research query planner. Given a user question, generate optimized search queries.

User question: {query}
Detected language: {language}

Generate two search queries:
- rag_query: optimized for searching local documents, MUST be in {language} language
- web_query: optimized for web search, ALWAYS in English (more results available)

Respond ONLY with valid JSON, no other text:
{{
    "rag_query": "...",
    "web_query": "..."
}}"""

def plan_queries(query: str) -> dict:
    llm = get_llm(task="planner")
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