from dotenv import load_dotenv
from rag.loader import load_file
from rag.splitter import split_documents
from rag.vector_storage import save_to_db
from agents.local_researcher import ask_local

load_dotenv()

docs = load_file("test_data/systemy_operacyjne.pdf")

chunks = split_documents(docs)

save_to_db(chunks, source_file="systemy_operacyjne.pdf")

result = ask_local("jakie algorytmy sortowania zostały omówione?")

print("ODPOWIEDŹ:")
print(result["answer"])

print("\nŹRÓDŁA:")
for i, src in enumerate(result["sources"]):
    print(f"\n--- Fragment {i+1} ---")
    print(src)