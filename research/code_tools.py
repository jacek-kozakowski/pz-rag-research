from langchain_core.prompts import PromptTemplate
from agents import get_llm

CODE_GEN_PROMPT = """
Wygeneruj kod w języku {language} spełniający wymagania:
{requirement}

Zwróć TYLKO kod. Nie dodawaj żadnych objaśnień, wstępów ani zakończeń.
"""

TEST_GEN_PROMPT = """
Wygeneruj testy jednostkowe dla poniższego kodu przy użyciu frameworka {framework}:
{code}

Zwróć TYLKO kod testów. Nie dodawaj żadnych objaśnień.
"""

DOC_GEN_PROMPT = """
Wygeneruj dokumentację techniczną dla poniższego kodu w formacie Markdown.
Kod:
{code}
"""

def generate_code(requirement: str, language: str) -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=CODE_GEN_PROMPT,
        input_variables=["language", "requirement"]
    )
    chain = prompt | llm
    answer = chain.invoke({"language": language, "requirement": requirement})
    return answer.content

def generate_tests(code: str, framework: str) -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=TEST_GEN_PROMPT,
        input_variables=["framework", "code"]
    )
    chain = prompt | llm
    answer = chain.invoke({"framework": framework, "code": code})
    return answer.content

def generate_documentation(code: str) -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=DOC_GEN_PROMPT,
        input_variables=["code"]
    )
    chain = prompt | llm
    answer = chain.invoke({"code": code})
    return answer.content
