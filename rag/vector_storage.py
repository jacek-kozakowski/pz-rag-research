from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from functools import lru_cache

CHROMA_PATH = "./chroma"
EMBEDDING_MODEL = "text-embedding-3-small"

@lru_cache(maxsize=1)
def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def get_indexed_files() -> set:
    if not os.path.exists(CHROMA_PATH):
        return set()
    db = load_db()
    results = db.get()
    sources = {m.get("source") for m in results["metadatas"] if m.get("source")}
    return sources

def save_to_db(chunks, source_file: str = None):
    indexed = get_indexed_files()
    print(f"DEBUG indexed files: {indexed}")
    if source_file and source_file in indexed:
        print(f"Source file {source_file} already indexed, skipping")
        return

    if source_file:
        for chunk in chunks:
            chunk.metadata["source"] = source_file

    embeddings = get_embeddings()
    if os.path.exists(CHROMA_PATH):
        db = load_db()
        db.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to existing Chroma DB")
        return db
    else:
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Saved {len(chunks)} chunks to Chroma")
    return db

def load_db():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Chroma DB not found at {CHROMA_PATH}. Run save_to_db first.")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())

def search(query: str, k: int = 3):
    db = load_db()
    return db.similarity_search(query)
