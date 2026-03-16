from dotenv import load_dotenv
from rag.loader import load_file, load_from_minio
from rag.splitter import split_documents
from rag.vector_storage import save_to_db
from agents.graph import build_graph
from rag.minio_storage import upload_file
load_dotenv()

upload_file("test_data/systemy_operacyjne.pdf", "system_operacyjne.pdf")

docs = load_from_minio("system_operacyjne.pdf")

chunks = split_documents(docs)

save_to_db(chunks, source_file="systemy_operacyjne.pdf")

question = "Jak działa wielowątkowość w programowaniu? Pokaż przykłady."

graph = build_graph()
result = graph.invoke({"query": question})

print(result['summary'])