import streamlit as st
from rag.loader import load_from_minio
from rag.splitter import split_documents
from rag.vector_storage import save_to_db, get_indexed_files, delete_from_db
from rag.minio_storage import upload_bytes, list_files, delete_file
from code.loader import index_codebase

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

    return {"k": k_value, "web": web_enabled}