from dotenv import load_dotenv
from rag.loader import load_file
from rag.splitter import split_documents
from rag.vector_storage import save_to_db
from agents.local_researcher import ask_local
from agents.web_researcher import web_search

load_dotenv()

docs = load_file("test_data/systemy_operacyjne.pdf")

chunks = split_documents(docs)

save_to_db(chunks, source_file="systemy_operacyjne.pdf")

question = "Wytłumacz wielowątkowość w programowaniu"
result_rag = ask_local(question)
result_web = web_search(question)

print("ODPOWIEDŹ local:")
print(result_rag["answer"])

print("ODPOWIEDŹ web:")
print(result_web["answer"])