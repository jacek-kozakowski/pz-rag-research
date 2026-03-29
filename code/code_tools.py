from langchain_core.prompts import PromptTemplate
from agents import get_llm

CODE_GEN_PROMPT = """
Generate {language} code that fulfills the following requirement:
{requirement}

Use the context below as reference if relevant:
{context}

Return ONLY the code. No explanations, no markdown blocks.
"""

TEST_GEN_PROMPT = """
Generate unit tests for the following code using the {framework} framework:
{code}

Use the context below to understand the project structure if relevant:
{context}

Return ONLY the test code. No explanations, no markdown blocks.
"""

DOC_GEN_PROMPT = """
Generate technical documentation for the following code in Markdown format.
Use the context below to better describe dependencies and structure if relevant:
{context}

Code to document:
{code}

Return ONLY the documentation in Markdown format.
"""

def generate_code(requirement: str, language: str, context: str = "") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=CODE_GEN_PROMPT,
        input_variables=["language", "requirement", "context"]
    )
    chain = prompt | llm
    response = chain.invoke({"language": language, "requirement": requirement, "context": context})
    return _extract_code(response.content)

def generate_tests(code: str, framework: str, context: str = "") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=TEST_GEN_PROMPT,
        input_variables=["framework", "code", "context"]
    )
    chain = prompt | llm
    response = chain.invoke({"framework": framework, "code": code, "context": context})
    return response.content

def generate_documentation(code: str, context: str = "") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=DOC_GEN_PROMPT,
        input_variables=["code", "context"]
    )
    chain = prompt | llm
    response = chain.invoke({"code": code, "context": context})
    return response.content


def _extract_code(response: str) -> str:
    import re
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()