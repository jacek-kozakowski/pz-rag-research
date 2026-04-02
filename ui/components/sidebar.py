import os
import json
import toml
import streamlit as st
from pathlib import Path
from rag.loader import load_from_minio
from rag.splitter import split_documents
from rag.vector_storage import save_to_db, get_indexed_files, delete_from_db
from rag.minio_storage import upload_bytes, list_files, delete_file
from code.loader import index_codebase

_SECRETS_PATH = Path.home() / ".streamlit" / "secrets.toml"
_GOOGLE_CREDS_PATH = Path.home() / ".streamlit" / "google_credentials.json"

def _load_secrets() -> dict:
    if _SECRETS_PATH.exists():
        return toml.load(_SECRETS_PATH)
    return {}

def _save_secrets(data: dict):
    _SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_secrets()
    existing.update(data)
    with open(_SECRETS_PATH, "w") as f:
        toml.dump(existing, f)
    for k, v in data.items():
        if v:
            os.environ[k] = v

def render_sidebar() -> dict:
    """Renderuje sidebar, zwraca ustawienia jako dict"""
    with st.sidebar:
        st.markdown('<div class="main-title">RAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Research Assistant</div>', unsafe_allow_html=True)

        # --- SECTION: UPLOAD & INDEX ---
        st.markdown("---")
        st.markdown("### 📤 Upload & Index")
        
        col_type = st.radio("Target Collection:", ["Research", "Code"], horizontal=True)
        
        uploaded_file = st.file_uploader(
            "Drop file here",
            type=["pdf", "docx", "txt", "py", "js", "ts", "go", "rs", "java", "cpp", "c", "h", "cs"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            if st.button("INDEX →"):
                with st.spinner("Indexing..."):
                    try:
                        filename = uploaded_file.name
                        collection = col_type.lower()
                        
                        # Only upload to MinIO for research docs (optional, but keep for consistency with existing loader)
                        if collection == "research":
                            upload_bytes(uploaded_file.read(), filename)
                            docs = load_from_minio(filename)
                        else:
                            # For code files uploaded directly, we can wrap them in a Document
                            from langchain_core.documents import Document
                            content = uploaded_file.read().decode('utf-8', errors='ignore')
                            docs = [Document(page_content=content, metadata={"source": filename})]

                        chunks = split_documents(docs)
                        save_to_db(chunks, source_file=filename, collection_type=collection)
                        st.success(f"✓ {filename} ({collection})")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # --- SECTION: FOLDER INDEXING ---
        st.markdown("---")
        st.markdown("### 📂 Index Folder")

        col_folder_type = st.radio("Collection:", ["Research", "Code"], horizontal=True, key="folder_col_type")
        folder_path = st.text_input("Folder path", placeholder="/path/to/folder", key="folder_path_input")

        RESEARCH_EXTS = {".pdf", ".docx", ".txt"}
        CODE_EXTS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".cs"}

        if folder_path and st.button("INDEX FOLDER →"):
            folder = Path(folder_path)
            if not folder.is_dir():
                st.error(f"Not a directory: {folder_path}")
            else:
                collection = col_folder_type.lower()
                exts = RESEARCH_EXTS if collection == "research" else CODE_EXTS
                files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in exts]
                if not files:
                    st.warning(f"No supported files found ({', '.join(sorted(exts))})")
                else:
                    progress = st.progress(0, text=f"0 / {len(files)}")
                    ok, errors = 0, []
                    for i, file in enumerate(files):
                        try:
                            if collection == "research":
                                from rag.loader import load_file
                                docs = load_file(str(file))
                                for doc in docs:
                                    doc.metadata["source"] = file.name
                            else:
                                from langchain_core.documents import Document
                                content = file.read_text(errors="ignore")
                                docs = [Document(page_content=content, metadata={"source": file.name})]
                            chunks = split_documents(docs)
                            save_to_db(chunks, source_file=file.name, collection_type=collection)
                            ok += 1
                        except Exception as e:
                            errors.append(f"{file.name}: {e}")
                        progress.progress((i + 1) / len(files), text=f"{i + 1} / {len(files)}: {file.name}")
                    progress.empty()
                    st.success(f"✓ Indexed {ok} / {len(files)} files")
                    if errors:
                        with st.expander(f"⚠ {len(errors)} errors"):
                            st.markdown("\n".join(f"- {e}" for e in errors))

        # --- SECTION: CODEBASE INDEXING ---
        st.markdown("---")
        st.markdown("### 📁 Hub: Codebase")
        code_path = st.text_input("Path/URL/ZIP", value=".", help="Local path, GitHub URL or ZIP file")
        if st.button("INDEX CODEBASE →"):
            with st.spinner("Indexing codebase..."):
                try:
                    index_codebase(code_path)
                    st.success("✓ Code indexed")
                except Exception as e:
                    st.error(f"Error: {e}")

        # --- SECTION: FILE MANAGEMENT ---
        st.markdown("---")
        
        # TAB 1: RESEARCH
        st.markdown("#### 🔬 Research Files")
        indexed_res = get_indexed_files(collection_type="research")
        if not indexed_res:
            st.markdown('<div class="file-item">no research files</div>', unsafe_allow_html=True)
        else:
            for f in indexed_res:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f'<div class="file-item"><span class="indexed-dot active"></span>{f}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("✕", key=f"del_res_{f}"):
                        delete_from_db(f, collection_type="research")
                        st.rerun()

        # TAB 2: CODE
        st.markdown("#### 💻 Code Files")
        indexed_code = get_indexed_files(collection_type="code")
        if not indexed_code:
            st.markdown('<div class="file-item">no code files</div>', unsafe_allow_html=True)
        else:
            for f in indexed_code:
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f'<div class="file-item"><span class="indexed-dot active" style="background-color: #2ecc71;"></span>{f}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("✕", key=f"del_code_{f}"):
                        delete_from_db(f, collection_type="code")
                        st.rerun()

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            k_value = st.number_input("k (RAG)", min_value=1, max_value=10, value=3)
        with col2:
            web_enabled = st.toggle("Web", value=True)

        # --- SECTION: SETTINGS ---
        st.markdown("---")
        with st.expander("⚙️ API Settings"):
            secrets = _load_secrets()

            st.markdown("**LLM**")
            openai_key = st.text_input("OpenAI API Key", value=secrets.get("OPENAI_API_KEY", ""), type="password")
            groq_key = st.text_input("Groq API Key", value=secrets.get("GROQ_API_KEY", ""), type="password")

            st.markdown("**Web Search**")
            tavily_key = st.text_input("Tavily API Key", value=secrets.get("TAVILY_API_KEY", ""), type="password")

            st.markdown("**GitHub**")
            github_token = st.text_input("GitHub Token", value=secrets.get("GITHUB_TOKEN", ""), type="password")
            github_repo = st.text_input("GitHub Repo (owner/repo)", value=secrets.get("GITHUB_REPO", ""))

            st.markdown("**Google Calendar**")
            google_cal_id = st.text_input("Calendar ID (email)", value=secrets.get("GOOGLE_CALENDAR_ID", ""))
            google_creds_file = st.file_uploader("Service Account JSON", type=["json"])
            if google_creds_file:
                creds_data = google_creds_file.read()
                try:
                    json.loads(creds_data)
                    _GOOGLE_CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _GOOGLE_CREDS_PATH.write_bytes(creds_data)
                    st.success(f"Saved to {_GOOGLE_CREDS_PATH}")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")

            if st.button("💾 Save"):
                _save_secrets({
                    "OPENAI_API_KEY": openai_key,
                    "GROQ_API_KEY": groq_key,
                    "TAVILY_API_KEY": tavily_key,
                    "GITHUB_TOKEN": github_token,
                    "GITHUB_REPO": github_repo,
                    "GOOGLE_CALENDAR_ID": google_cal_id,
                    "GOOGLE_CREDENTIALS_PATH": str(_GOOGLE_CREDS_PATH) if _GOOGLE_CREDS_PATH.exists() else secrets.get("GOOGLE_CREDENTIALS_PATH", ""),
                })
                st.success("Saved!")

    return {"k": k_value, "web": web_enabled}