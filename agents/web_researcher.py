from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.prompts import PromptTemplate
from agents import get_llm
import os

PROMPT_TEMPLATE = """
You are a helpful assistant analyzing web search results and extracting key information.
Based on the web search results answer the question.
If you can't answer the question based on provided search results or you don't know the answer, just say that you don't know.

Search results:
{context}

Question:
{question}

Answer:"""

def web_search_raw(query: str, k: int = 3):
    if os.getenv("TAVILY_API_KEY"):
        try:
            search = TavilySearch(max_results=k)
            results = search.run(query)
            context = "\n\n---\n\n".join(r["content"] for r in results)
            return context, "tavily"
        except Exception as e:
            print("Tavily search failed. Falling back to DuckDuckGo.")

    try:
        context = DuckDuckGoSearchRun().run(query)
        return context, "duckduckgo"
    except Exception as e:
        print(f"DuckDuckGo failed. {e}")
        return "", "none"

def web_search(query: str, k: int = 3):
    context, source = web_search_raw(query, k)

    if not context:
        return {
            "answer" : "No information found",
            "source" : "none",
            "raw_results" : ""
        }

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", 'question']
    )
    llm = get_llm()

    chain = prompt | llm
    result = chain.invoke({"context": context, 'question': query})
    return {
        "answer" : result.content,
        "source" :  source,
        "raw_results" : context
    }
