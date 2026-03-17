import streamlit as st
from rag.loader import load_from_minio
from rag.splitter import split_documents
from rag.vector_storage import save_to_db, get_indexed_files, delete_from_db
from rag.minio_storage import upload_bytes, list_files, delete_file

def render_sidebar() -> dict:
    """Renderuje sidebar, zwraca ustawienia jako dict"""
    with st.sidebar:
        st.markdown('<div class="main-title">RAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Research Assistant</div>', unsafe_allow_html=True)

        st.markdown("#### Upload Document")
        uploaded_file = st.file_uploader(
            "Drop file here",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            if st.button("INDEX →"):
                with st.spinner("Indexing..."):
                    try:
                        filename = uploaded_file.name
                        upload_bytes(uploaded_file.read(), filename)
                        docs = load_from_minio(filename)
                        chunks = split_documents(docs)
                        save_to_db(chunks, source_file=filename)
                        st.success(f"✓ {filename}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("#### Indexed Files")

        try:
            minio_files = list_files()
            indexed = get_indexed_files()

            if not minio_files:
                st.markdown('<div class="file-item">no files yet</div>', unsafe_allow_html=True)
            else:
                for f in minio_files:
                    is_indexed = f in indexed
                    dot_class = "indexed-dot active" if is_indexed else "indexed-dot"

                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f'<div class="file-item"><span class="{dot_class}"></span>{f}</div>',
                            unsafe_allow_html=True
                        )
                    with col2:
                        if st.button("✕", key=f"del_{f}"):
                            try:
                                delete_file(f)  # usuń z MinIO
                                delete_from_db(f)  # usuń z Chroma
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
        except Exception:
            st.markdown('<div class="file-item">minio unavailable</div>', unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            k_value = st.number_input("k (RAG)", min_value=1, max_value=10, value=3)
        with col2:
            web_enabled = st.toggle("Web", value=True)

    return {"k": k_value, "web": web_enabled}