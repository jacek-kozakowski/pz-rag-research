from langchain_core.prompts import PromptTemplate
from agents import get_llm

PROMPT_TEMPLATE = """You are a research analyst. Create a comprehensive, detailed report based on the sources below.
IMPORTANT: Respond in the same language as the user's question.

User's goal: {query}

Local documents findings:
{local_answer}

Web research findings:
{web_answer}

Write a DETAILED report that:
- Is at least 500 words long
- Has clear sections with headers (##)
- Includes specific facts, examples, code snippets if relevant
- Provides actionable information directly related to the user's goal
- Cites specific findings from both sources

Report:"""
def summarize(query: str, local_result: dict, web_result: dict) -> dict:

    local_answer = local_result.get('answer', "No information found in local documents")
    web_answer = web_result.get('answer', "No information found in web search")

    llm = get_llm()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["local_answer", "web_answer", "query"]
    )

    chain = prompt | llm

    summary = chain.invoke({
        "local_answer": local_answer,
        "web_answer": web_answer,
        "query": query
    })

    return {
        "summary": summary.content,
        "local_answer": local_answer,
        "web_answer": web_answer,
        "sources": {
            "local": local_result.get('sources', []),
            "web": web_result.get('source', 'none')
        }
    }