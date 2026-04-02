import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from agents.graph import build_project_graph, build_learning_graph
from ui.components.styles import inject_css
from ui.components.sidebar import render_sidebar
from langchain_core.messages import HumanMessage

load_dotenv()

# Load Streamlit secrets into os.environ (secrets take priority over .env)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass

st.set_page_config(
    page_title="RAG Research",
    page_icon="🔍",
    layout="wide"
)

inject_css()
settings = render_sidebar()

st.markdown("---")

mode = st.radio(
    "Mode",
    options=["project", "learning"],
    horizontal=True,
    format_func=lambda m: "🛠 Project" if m == "project" else "📚 Learning"
)

query = st.text_area(
    "Query",
    placeholder="Ask anything about your documents...",
    height=100,
    label_visibility="collapsed"
)

create_repo = False
if mode == "project":
    create_repo = st.checkbox(
        "Create new GitHub repo for this project",
        help="Requires GITHUB_TOKEN. If GITHUB_REPO is already set in .env, that repo will be used instead."
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
            graph = build_project_graph() if mode == "project" else build_learning_graph()
            result = graph.invoke({
                "query": query,
                "mode": mode,
                "create_repo": create_repo,
                "messages": [HumanMessage(content=query)]
            })
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"Error: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]

    st.markdown("#### Summary")
    st.markdown(f'<div class="result-box">{result.get("summary", "")}</div>', unsafe_allow_html=True)

    with st.expander("Local Research"):
        local = result.get("local_result", {})
        st.markdown(local.get("answer", "—"))
        for src in local.get("sources", []):
            st.markdown(f'<div class="result-box" style="font-size:0.82rem;color:#555;">{src}</div>', unsafe_allow_html=True)

    with st.expander("Web Research"):
        web = result.get("web_result", {})
        st.markdown(web.get("answer", "—"))
        if src_tag := web.get("source", ""):
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
                        <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#555;">{i + 1:02d}</span>
                        <span class="source-tag" style="color:{color}; border-color:{color};">{priority}</span>
                    </div>
                    <div style="font-weight:500; margin: 0.4rem 0 0.2rem;">{task.get("title", "")}</div>
                    <div style="font-size:0.85rem; color:#888; margin-bottom:0.5rem;">{task.get("description", "")}</div>
                    <div style="display:flex; gap:1rem;">
                        <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#555;">⏱ {task.get("duration_minutes", "")} min</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # --- Project mode outputs ---
    github_issues = result.get("github_issues", [])
    if github_issues:
        st.markdown("#### GitHub Issues")
        for issue in github_issues:
            st.markdown(f"- [#{issue['number']} {issue['title']}]({issue['url']})")

    readme = result.get("readme", "")
    if readme:
        with st.expander("README.md", expanded=True):
            st.markdown(readme)

    # --- Learning mode outputs ---
    calendar_events = result.get("calendar_events", [])
    if calendar_events:
        st.markdown("#### Calendar Events")
        for event in calendar_events:
            url_part = f" — [open]({event['url']})" if event.get('url') else ""
            st.markdown(f"- **{event['title']}** @ {event['start']}{url_part}")

    notes = result.get("notes", "")
    if notes:
        with st.expander("Learning Notes", expanded=True):
            st.markdown(notes)
