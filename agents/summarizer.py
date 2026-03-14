from langchain_core.prompts import PromptTemplate
from agents import get_llm

PROMPT_TEMPLATE = """
You are a research assistant. Your task is to combine and summarize information from two sources into a single coherent report.

Local documents research:
{local_answer}

Web research:
{web_answer}

Create a comprehensive summary that:
- Combines information from both sources
- Eliminates redundancy
- Highlights the most important findings
- Notes any contradictions between sources

Question that was researched: {query}

Summary:"""

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