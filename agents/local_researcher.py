from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from rag.vector_storage import load_db
from agents import get_llm
PROMPT_TEMPLATE = """
You are a helpful assistant, answering questions based on the provided documents.
Answer only based on the context provided below. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Respond in language of the question.
Context:
{context}

Question:
{question}
"""


def ask_local(query: str, k: int = 3) -> dict:
    db = load_db()

    llm = get_llm()

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    answer = chain.invoke({"query": query})

    return {
        "answer": answer["result"],
        "sources": [doc.page_content for doc in answer["source_documents"]]
    }