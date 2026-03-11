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
        openai_api_type=os.getenv("OPENAI_API_KEY")
    )


def save_to_db(chunks):
    embeddings = get_embeddings()
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
