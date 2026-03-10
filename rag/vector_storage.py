from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def save_to_db(chunks):
    embeddings = get_embeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"Saved {len(chunks)} chunks to Chroma")
    return db

def load_db():
    embeddings = get_embeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def search(query: str, k: int = 3):
    db = load_db()
    return db.similarity_search(query)
