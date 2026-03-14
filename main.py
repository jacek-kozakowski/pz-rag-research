from dotenv import load_dotenv
from rag.loader import load_file
from rag.splitter import split_documents
from rag.vector_storage import save_to_db
from agents.local_researcher import ask_local
from agents.web_researcher import web_search
from agents.summarizer import summarize

load_dotenv()

docs = load_file("test_data/systemy_operacyjne.pdf")

chunks = split_documents(docs)

save_to_db(chunks, source_file="systemy_operacyjne.pdf")

question = "Wielowątkowość w programowaniu"
local = ask_local(question)
web = web_search(question)
summary = summarize(question, local, web)

print(summary['summary'])