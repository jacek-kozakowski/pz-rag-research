from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from functools import lru_cache

CHROMA_PATH = "./chroma"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
HUGGINGFACE_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def get_current_embedding_model() -> str:
    return OPENAI_EMBEDDING_MODEL if os.getenv("OPENAI_API_KEY") else HUGGINGFACE_EMBEDDING_MODEL

@lru_cache(maxsize=1)
def get_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        print(f"Using OpenAI embeddings. Model: {OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        print(f"Using HuggingFace embeddings. Model: {HUGGINGFACE_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=HUGGINGFACE_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
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
            collection_metadata={
                "hnsw:space": "cosine",
                "embedding_model" : get_current_embedding_model()
            }
        )
        print(f"Saved {len(chunks)} chunks to Chroma")
    return db

def load_db():
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Chroma DB not found at {CHROMA_PATH}. Run save_to_db first.")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    stored_model = db._collection.metadata.get("embedding_model")
    if stored_model and stored_model != get_current_embedding_model():
        raise ValueError(
            f"Embedding model mismatch. Stored: {stored_model}, Current: {get_current_embedding_model()}."
            f"Delete {CHROMA_PATH} and reindex"
        )
    return db


# k - number of results to return
def search(query: str, k: int = 3):
    db = load_db()
    return db.similarity_search(query, k=k)
