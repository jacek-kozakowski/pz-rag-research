from langchain_core.prompts import PromptTemplate
from agents import get_llm

PROMPT_TEMPLATE = """You are a research analyst. Create a concise summary based on the sources below.
IMPORTANT: Respond in the same language as the user's question.

User's goal: {query}

Local documents findings:
{local_answer}

Web research findings:
{web_answer}

Write a concise summary that:
- Is 150-250 words long
- Lists the key concepts and topics found
- Highlights the most important facts
- Does NOT go into deep explanations — that will be done separately

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