import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from agents.graph import build_project_graph, build_learning_graph
from agents.nodes.detect_mode import detect_mode
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

query = st.text_area(
    "Query",
    placeholder="Ask anything about your documents...",
    height=100,
    label_visibility="collapsed"
)

col_cb1, col_cb2 = st.columns(2)
with col_cb1:
    create_repo = st.checkbox(
        "Create new GitHub repo for this project",
        help="Requires GITHUB_TOKEN. Only used when project mode is detected."
    )
with col_cb2:
    use_calendar = st.checkbox(
        "Add events to Google Calendar",
        help="Requires GOOGLE_CREDENTIALS_PATH and GOOGLE_CALENDAR_ID. Only used when learning mode is detected."
    )

col1, col2, col3 = st.columns([1, 1, 6])
with col1:
    run = st.button("RUN →")
with col2:
    if st.button("CLEAR"):
        st.session_state.pop("result", None)
        st.session_state.pop("detected_mode", None)
        st.rerun()

if run and query:
    try:
        with st.spinner("Detecting mode..."):
            mode = detect_mode(query)
        st.session_state["detected_mode"] = mode

        with st.spinner("Researching..."):
            graph = build_project_graph() if mode == "project" else build_learning_graph()
            result = graph.invoke({
                "query": query,
                "mode": mode,
                "create_repo": create_repo,
                "use_calendar": use_calendar,
                "web_enabled": settings["web"],
                "messages": [HumanMessage(content=query)]
            })
            st.session_state["result"] = result
    except Exception as e:
        st.error(f"Error: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]

    detected_mode = st.session_state.get("detected_mode", "")
    if detected_mode:
        label = "🛠 Project" if detected_mode == "project" else "📚 Learning"
        st.markdown(f'<span class="source-tag">{label}</span>', unsafe_allow_html=True)

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
    if plan and not result.get("github_issues"):
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

    scaffold = result.get("scaffold", [])
    if scaffold:
        st.markdown("#### Project Scaffold")
        language = result.get("language", "python").lower()

        # File tree
        tree_lines = []
        dirs_seen = set()
        for entry in scaffold:
            parts = entry["filepath"].split("/")
            for depth in range(len(parts) - 1):
                dir_path = "/".join(parts[: depth + 1])
                if dir_path not in dirs_seen:
                    dirs_seen.add(dir_path)
                    tree_lines.append("  " * depth + f"📁 {parts[depth]}/")
            indent = "  " * (len(parts) - 1)
            tree_lines.append(f"{indent}📄 {parts[-1]}")
        st.code("\n".join(tree_lines), language=None)

        # Per-file expanders
        for entry in scaffold:
            ext = entry["filepath"].rsplit(".", 1)[-1] if "." in entry["filepath"] else language
            lang_map = {"ts": "typescript", "tsx": "tsx", "js": "javascript", "jsx": "jsx",
                        "py": "python", "go": "go", "rs": "rust", "java": "java",
                        "yaml": "yaml", "yml": "yaml", "json": "json", "md": "markdown",
                        "toml": "toml", "env": "bash", "sh": "bash", "sql": "sql",
                        "html": "html", "css": "css"}
            code_lang = lang_map.get(ext, language)
            with st.expander(f"`{entry['filepath']}` — {entry.get('purpose', '')}"):
                st.code(entry.get("code", ""), language=code_lang)

        zip_buffer = None
        try:
            import io, zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for entry in scaffold:
                    zf.writestr(entry["filepath"], entry.get("code", ""))
            buf.seek(0)
            zip_buffer = buf
        except Exception:
            pass
        if zip_buffer:
            st.download_button(
                label="Download scaffold as ZIP",
                data=zip_buffer,
                file_name="scaffold.zip",
                mime="application/zip",
            )

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
