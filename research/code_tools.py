from langchain_core.prompts import PromptTemplate
from agents import get_llm

CODE_GEN_PROMPT = """
Wygeneruj kod w języku {language} spełniający wymagania:
{requirement}

Użyj poniższego kontekstu jako punktu odniesienia (jeśli jest istotny):
{context}

Zwróć TYLKO kod. Nie dodawaj żadnych objaśnień, wstępów ani zakończeń.
"""

TEST_GEN_PROMPT = """
Wygeneruj testy jednostkowe dla poniższego kodu przy użyciu frameworka {framework}:
{code}

Użyj poniższego kontekstu, aby zrozumieć strukturę projektu (jeśli jest istotny):
{context}

Zwróć TYLKO kod testów. Nie dodawaj żadnych objaśnień.
"""

DOC_GEN_PROMPT = """
Wygeneruj dokumentację techniczną dla poniższego kodu w formacie Markdown.
Użyj kontekstu, aby lepiej opisać zależności i strukturę (jeśli jest istotny):
{context}

Kod do udokumentowania:
{code}
"""

def generate_code(requirement: str, language: str, context: str = "") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=CODE_GEN_PROMPT,
        input_variables=["language", "requirement", "context"]
    )
    chain = prompt | llm
    answer = chain.invoke({"language": language, "requirement": requirement, "context": context})
    return answer.content

def generate_tests(code: str, framework: str, context: str = "") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=TEST_GEN_PROMPT,
        input_variables=["framework", "code", "context"]
    )
    chain = prompt | llm
    answer = chain.invoke({"framework": framework, "code": code, "context": context})
    return answer.content

def generate_documentation(code: str, context: str = "") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=DOC_GEN_PROMPT,
        input_variables=["code", "context"]
    )
    chain = prompt | llm
    answer = chain.invoke({"code": code, "context": context})
    return answer.content
