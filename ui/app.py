import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from agents.graph import build_graph
from ui.components.styles import inject_css
from ui.components.sidebar import render_sidebar

load_dotenv()

st.set_page_config(
    page_title="RAG Research",
    page_icon="🔍",
    layout="wide"
)

inject_css()
settings = render_sidebar()

st.markdown("---")

query = st.text_area(
    "Query",
    placeholder="Ask anything about your documents...",
    height=100,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 6])
with col1:
    run = st.button("RUN →")
with col2:
    if st.button("CLEAR"):
        st.session_state.pop("result", None)
        st.rerun()

if run and query:
    with st.spinner("Researching..."):
        try:
            graph = build_graph()
            result = graph.invoke({"query": query})
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"Error: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown("#### Summary")
    st.markdown(f'<div class="result-box">{result["summary"]}</div>', unsafe_allow_html=True)

    with st.expander("Local Research"):
        local = result.get("local_result", {})
        st.markdown(local.get("answer", "—"))
        sources = local.get("sources", [])
        if sources:
            st.markdown("**Fragments:**")
            for src in sources:
                st.markdown(f'<div class="result-box" style="font-size:0.82rem;color:#555;">{src}</div>', unsafe_allow_html=True)

    with st.expander("Web Research"):
        web = result.get("web_result", {})
        st.markdown(web.get("answer", "—"))
        src_tag = web.get("source", "")
        if src_tag:
            st.markdown(f'<span class="source-tag">{src_tag}</span>', unsafe_allow_html=True)