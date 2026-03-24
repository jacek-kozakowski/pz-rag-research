from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from functools import lru_cache

CHROMA_RESEARCH_PATH = "./chroma_research"
CHROMA_CODE_PATH = "./chroma_code"
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


def get_indexed_files(collection_type: str = "research") -> set:
    path = CHROMA_CODE_PATH if collection_type == "code" else CHROMA_RESEARCH_PATH
    if not os.path.exists(path):
        return set()
    db = load_db(path)
    results = db.get()
    sources = {m.get("source") for m in results["metadatas"] if m.get("source")}
    return sources

def save_to_db(chunks, source_file: str = None, collection_type: str = "research"):
    indexed = get_indexed_files(collection_type)
    print(f"DEBUG indexed files in {collection_type}: {indexed}")
    if source_file and source_file in indexed:
        print(f"Source file {source_file} already indexed in {collection_type}, skipping")
        return

    if source_file:
        for chunk in chunks:
            chunk.metadata["source"] = source_file

    embeddings = get_embeddings()
    path = CHROMA_CODE_PATH if collection_type == "code" else CHROMA_RESEARCH_PATH

    if os.path.exists(path):
        db = load_db(path)
        db.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to existing Chroma DB at {path}")
        return db
    else:
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=path,
            collection_metadata={
                "hnsw:space": "cosine",
                "embedding_model" : get_current_embedding_model()
            }
        )
        print(f"Saved {len(chunks)} chunks to Chroma at {path}")
    return db

def load_db(path: str = None):
    if path is None:
        path = CHROMA_RESEARCH_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Chroma DB not found at {path}. Run save_to_db first.")

    db = Chroma(persist_directory=path, embedding_function=get_embeddings())
    stored_model = db._collection.metadata.get("embedding_model")
    if stored_model and stored_model != get_current_embedding_model():
        raise ValueError(
            f"Embedding model mismatch. Stored: {stored_model}, Current: {get_current_embedding_model()}."
            f"Delete {path} and reindex"
        )
    return db


# k - number of results to return
def search(query: str, k: int = 3, collection_type: str = "research"):
    path = CHROMA_CODE_PATH if collection_type == "code" else CHROMA_RESEARCH_PATH
    db = load_db(path)
    return db.similarity_search(query, k=k)


def delete_from_db(source_file: str, collection_type: str = "research"):
    path = CHROMA_CODE_PATH if collection_type == "code" else CHROMA_RESEARCH_PATH
    db = load_db(path)
    results = db.get()
    ids_to_delete = [
        id for id, meta in zip(results["ids"], results["metadatas"])
        if meta.get("source") == source_file
    ]

    if ids_to_delete:
        db.delete(ids_to_delete)
        print(f"Deleted {len(ids_to_delete)} chunks from Chroma collection: {collection_type}")
    else:
        print(f"No chunks found for source file {source_file} in collection: {collection_type}")