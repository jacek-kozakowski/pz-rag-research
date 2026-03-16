from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os


def load_file(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    docs = loader.load()
    print(f"Loaded {len(docs)} pages/sections from {file_path}")
    return docs


def load_from_minio(filename: str):
    from rag.minio_storage import download_to_temp

    tmp_path = download_to_temp(filename)
    try:
        docs = load_file(tmp_path)
        for doc in docs:
            doc.metadata['source'] = filename
        return docs
    finally:
        os.unlink(tmp_path)