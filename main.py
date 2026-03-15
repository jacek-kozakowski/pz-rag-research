from dotenv import load_dotenv
from rag.loader import load_file
from rag.splitter import split_documents
from rag.vector_storage import save_to_db
from agents.graph import build_graph
load_dotenv()

docs = load_file("test_data/systemy_operacyjne.pdf")

chunks = split_documents(docs)

save_to_db(chunks, source_file="systemy_operacyjne.pdf")

question = "Jak działa wielowątkowość w programowaniu? Pokaż przykłady."

graph = build_graph()
result = graph.invoke({"query": question})

print(result['summary'])