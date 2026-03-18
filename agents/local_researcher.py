from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from rag.vector_storage import search
from agents import get_llm
PROMPT_TEMPLATE = """
You are a helpful assistant, answering questions based on the provided documents.
Use the context below to provide a comprehensive answer.
If the context contains relevant information, use it fully - do not say you don't know if the answer is present.
Respond in language of the question.
Context:
{context}

Question:
{question}
"""


def ask_local(query: str, rag_queries: list[str], k: int = 5) -> dict:
    llm = get_llm()

    queries = rag_queries if rag_queries else [query]
    seen = set()
    all_docs = []

    for q in queries:
        results = search(q)
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = prompt | llm

    answer = chain.invoke({"context" : context,"question": query})

    return {
        "answer": answer.content,
        "sources": [doc.page_content for doc in all_docs]
    }