def inject_css():
    import streamlit as st
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

        * { font-family: 'DM Sans', sans-serif; }

        .stApp {
            background: #0a0a0a;
            color: #e8e8e8;
        }

        .main-title {
            font-family: 'DM Mono', monospace;
            font-size: 2.2rem;
            font-weight: 500;
            color: #e8e8e8;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 0.85rem;
            color: #555;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: #111 !important;
            border-right: 1px solid #222;
        }

        .stTextArea textarea, .stTextInput input {
            background: #111 !important;
            border: 1px solid #2a2a2a !important;
            color: #e8e8e8 !important;
            border-radius: 4px !important;
            font-family: 'DM Sans', sans-serif !important;
        }

        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #444 !important;
            box-shadow: none !important;
        }

        .stButton button {
            background: #e8e8e8 !important;
            color: #0a0a0a !important;
            border: none !important;
            border-radius: 4px !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.8rem !important;
            letter-spacing: 0.05em !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.15s ease !important;
        }

        .stButton button:hover {
            background: #fff !important;
            transform: translateY(-1px);
        }

        .result-box {
            background: #111;
            border: 1px solid #222;
            border-radius: 4px;
            padding: 1.5rem;
            margin-top: 1rem;
            line-height: 1.7;
            font-size: 0.95rem;
            color: #ccc;
        }

        .source-tag {
            display: inline-block;
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            color: #555;
            border: 1px solid #2a2a2a;
            border-radius: 2px;
            padding: 0.1rem 0.5rem;
            margin: 0.2rem;
        }

        .file-item {
            font-family: 'DM Mono', monospace;
            font-size: 0.78rem;
            color: #666;
            padding: 0.4rem 0;
            border-bottom: 1px solid #1a1a1a;
        }

        .indexed-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #3a3a3a;
            margin-right: 8px;
        }

        .indexed-dot.active { background: #7fff7f; }

        .stFileUploader {
            border: 1px dashed #2a2a2a !important;
            border-radius: 4px !important;
            background: #0d0d0d !important;
        }

        .stSpinner > div { border-top-color: #e8e8e8 !important; }

        div[data-testid="stExpander"] {
            background: #111;
            border: 1px solid #1e1e1e;
            border-radius: 4px;
        }

        hr { border-color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)