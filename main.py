from dotenv import load_dotenv
from rag.loader import load_file
from rag.splitter import split_documents
from rag.vector_storage import save_to_db, search

load_dotenv()

docs = load_file("test_data/systemy_operacyjne.pdf")

chunks = split_documents(docs)

save_to_db(chunks, source_file="systemy_operacyjne.pdf")

query = "jakie algorytmy?"
results = search(query, k=3)

for i, r in enumerate(results):
    print(f"\n--- Fragment {i+1} ---")
    print(r.page_content)