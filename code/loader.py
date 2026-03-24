import os
import tempfile
import zipfile
from langchain_core.documents import Document
from pathlib import Path

REPO_TARGET_DIR = "repos"
BASE_DIR = Path(__file__).parent.parent


SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go',
    '.rb', '.php', '.rs', '.kt', '.swift', '.scala', '.r',
    '.html', '.css', '.sql', '.sh', '.yaml', '.yml', '.json',
    '.toml', '.xml', '.md'
}

IGNORED_DIRS = {
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.idea', '.vscode', 'coverage', '.pytest_cache'
}

def load_file(file_path: str) -> Document | None:
    if Path(file_path).suffix not in SUPPORTED_EXTENSIONS:
        return None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    return Document(
        page_content=content,
        metadata={
            'source': file_path,
            'filename': Path(file_path).name,
            'extension': Path(file_path).suffix,
            "type": "code"
        }
    )

def load_directory(dir_path: str) -> list[Document]:
    docs = []

    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            ext = Path(file).suffix
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                try:
                    doc = load_file(full_path)
                    doc.metadata['path'] = os.path.relpath(full_path, dir_path)
                    docs.append(doc)
                except Exception as e:
                    print(f"Error loading {full_path}: {e}")

    return docs

def load_zip_file(zip_path: str) -> list[Document]:
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        docs = load_directory(temp_dir)
    return docs

def load_git_repo(repo_url: str) -> list[Document]:
    import subprocess

    with tempfile.TemporaryDirectory() as temp_dir:
        result = subprocess.run(['git', 'clone', '--depth', '1', repo_url, temp_dir],
                                capture_output=True,
                                text=True)
        if result.returncode != 0:
            raise ValueError(f"Failed to clone repository: {result.stderr}")
        docs = load_directory(temp_dir)

    return docs

def load_codebase(source: str) -> list[Document]:

    if source.startswith(("https://", "http://", "git@", "ssh://")) :
        return load_git_repo(source)
    elif source.endswith(".zip"):
        return load_zip_file(source)
    elif os.path.isdir(source):
        return load_directory(source)
    elif os.path.isfile(source):
        return [load_file(source)]
    else:
        raise ValueError(f"Unsupported source type: {source}")


def index_codebase(source: str):
    from rag.vector_storage import save_to_db
    from rag.splitter import split_documents

    docs = load_codebase(source)
    chunks = split_documents(docs)
    save_to_db(chunks, collection_type="code")