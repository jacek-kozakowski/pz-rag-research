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

    plan = result.get("tasks", [])
    if plan:
        st.markdown("#### Plan")
        priority_colors = {"high": "#ff6b6b", "medium": "#ffd93d", "low": "#6bcb77"}

        for i, task in enumerate(plan):
            priority = task.get("priority", "medium")
            color = priority_colors.get(priority, "#555")

            st.markdown(f"""
                <div class="result-box" style="margin-bottom:0.5rem; border-left: 3px solid {color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#555;">
                            {i + 1:02d}
                        </span>
                        <span class="source-tag" style="color:{color}; border-color:{color};">
                            {priority}
                        </span>
                    </div>
                    <div style="font-weight:500; margin: 0.4rem 0 0.2rem;">
                        {task.get("title", "")}
                    </div>
                    <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem;">
                        {task.get("description", "")}
                    </div>
                    <div style="display:flex; gap:1rem;">
                        <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#555;">
                            📅 {task.get("deadline", "")}
                        </span>
                        <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#555;">
                            ⏱ {task.get("duration_minutes", "")} min
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)