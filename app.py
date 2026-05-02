"""
app.py
------
Streamlit UI for the RAG PDF Chatbot.
This file contains ONLY UI logic — all RAG work is delegated to rag_pipeline.py.

Run:
    python -m streamlit run app.py
"""

import logging

import streamlit as st

st.title("App is running 🚀")

# Internal modules
from config import config
from rag_pipeline import PipelineError, ask, process_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be the FIRST Streamlit call in the script)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Page background ────────────────────────────────── */
.stApp { background-color: #0d1117; }
.main .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 900px; }

/* ── Sidebar ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #010409;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f0f6fc !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebar"] hr { border-color: #21262d !important; }

/* ── Headings ───────────────────────────────────────── */
h1, h2, h3 { color: #f0f6fc !important; }

/* ── Chat bubbles ───────────────────────────────────── */
.bubble-user {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 12px 12px 3px 12px;
    padding: 13px 17px;
    margin: 6px 0 6px 80px;
    color: #c9d1d9;
    font-size: 0.94rem;
    line-height: 1.65;
    white-space: pre-wrap;
}
.bubble-bot {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #2ea043;
    border-radius: 3px 12px 12px 12px;
    padding: 13px 17px;
    margin: 6px 80px 6px 0;
    color: #c9d1d9;
    font-size: 0.94rem;
    line-height: 1.75;
    white-space: pre-wrap;
}
.bubble-error {
    background: #2d1316;
    border: 1px solid #6e3b3b;
    border-left: 3px solid #f85149;
    border-radius: 3px 12px 12px 12px;
    padding: 13px 17px;
    margin: 6px 80px 6px 0;
    color: #ff7b72;
    font-size: 0.9rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.label-user { text-align:right; color:#8b949e; font-size:0.7rem; margin-bottom:3px; font-family:'IBM Plex Mono',monospace; }
.label-bot  { color:#2ea043; font-size:0.7rem; margin-bottom:3px; font-family:'IBM Plex Mono',monospace; font-weight:600; }
.label-err  { color:#f85149; font-size:0.7rem; margin-bottom:3px; font-family:'IBM Plex Mono',monospace; font-weight:600; }

/* ── Source pill ────────────────────────────────────── */
.src-pill {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #8b949e;
    margin: 4px 3px 0 0;
}

/* ── Feature cards (welcome screen) ────────────────── */
.feat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px 18px;
    text-align: center;
    height: 100%;
}
.feat-icon { font-size: 1.8rem; margin-bottom: 8px; }
.feat-title { color: #f0f6fc; font-weight: 600; font-size: 0.95rem; margin-bottom: 5px; }
.feat-desc  { color: #8b949e; font-size: 0.82rem; line-height: 1.5; }

/* ── Status badges ──────────────────────────────────── */
.badge-ready   { background:#0f2a1a; border:1px solid #2ea043; color:#3fb950; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-idle    { background:#1a1f26; border:1px solid #30363d; color:#8b949e;  padding:4px 12px; border-radius:20px; font-size:0.78rem; font-family:'IBM Plex Mono',monospace; }

/* ── Misc fixes ─────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px;
}
[data-testid="stChatInput"] textarea {
    background: #1c2128 !important;
    color: #c9d1d9 !important;
    border-color: #30363d !important;
}
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
}
.stButton > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}
.stButton > button:hover { background: #30363d; border-color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state — initialise once per browser session
# ─────────────────────────────────────────────────────────────────────────────

_STATE_DEFAULTS: dict = {
    "vectorstore":  None,    # FAISS index after PDF is processed
    "chat_history": [],      # list[dict]: {role, content, meta}
    "pdf_name":     None,    # filename of the processed PDF
    "pdf_chunks":   0,       # number of indexed chunks
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 RAG PDF Chatbot")
    st.caption("Ask questions about any PDF using AI")
    st.divider()

    # ── API key ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 Groq API Key")
    key_input = st.text_input(
        "API Key",
        value=config.groq_api_key,
        type="password",
        placeholder="gsk_…",
        help="Get a free key at console.groq.com",
        label_visibility="collapsed",
    )
    if key_input:
        config.groq_api_key = key_input   # runtime override

    if not config.groq_api_key:
        st.warning("⚠️ API key required to chat", icon="⚠️")
    else:
        st.success("API key loaded", icon="✅")

    st.divider()

    # ── PDF upload ────────────────────────────────────────────────────────────
    st.markdown("### 📂 Upload PDF")
    uploaded = st.file_uploader(
        "PDF file",
        type=["pdf"],
        label_visibility="collapsed",
        help="Text-based PDFs only. Scanned/image PDFs are not supported.",
    )

    if uploaded is not None:
        st.caption(f"File: `{uploaded.name}` ({uploaded.size / 1024:.1f} KB)")

        if st.button("⚡ Process PDF", type="primary", use_container_width=True):
            with st.spinner("Parsing and indexing PDF…"):
                try:
                    vs = process_pdf(uploaded)
                    st.session_state.vectorstore  = vs
                    st.session_state.pdf_name     = uploaded.name
                    st.session_state.pdf_chunks   = vs.index.ntotal
                    st.session_state.chat_history = []
                    st.success(f"✅ Indexed {vs.index.ntotal} chunks", icon="✅")
                except PipelineError as exc:
                    st.error(f"❌ {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error:\n{exc}")

    st.divider()

    # ── Document status ───────────────────────────────────────────────────────
    st.markdown("### 📊 Status")

    if st.session_state.vectorstore:
        st.markdown('<span class="badge-ready">● Ready</span>', unsafe_allow_html=True)
        st.markdown(
            f"**File:** `{st.session_state.pdf_name}`\n\n"
            f"**Chunks:** {st.session_state.pdf_chunks}"
        )
    else:
        st.markdown('<span class="badge-idle">○ No document loaded</span>', unsafe_allow_html=True)

    st.divider()

    # ── Config panel ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Configuration"):
        st.markdown(f"""
| Setting | Value |
|:---|:---|
| Model | `{config.llm_model}` |
| Embeddings | `{config.embedding_model}` |
| Chunk size | {config.chunk_size} chars |
| Chunk overlap | {config.chunk_overlap} chars |
| Retrieval k | {config.retrieval_k} chunks |
| Temperature | {config.llm_temperature} |
        """)

    st.divider()

    # ── Clear button ──────────────────────────────────────────────────────────
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="color:#f0f6fc;font-size:2rem;font-weight:600;margin-bottom:2px;">'
    '📄 PDF Chatbot</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#8b949e;margin-top:0;margin-bottom:1rem;">'
    f'Powered by <code>{config.llm_model}</code> via Groq · '
    f'Embeddings: <code>{config.embedding_model}</code></p>',
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.chat_history:
    # Welcome / onboarding screen
    if st.session_state.vectorstore:
        st.markdown(
            '<div style="text-align:center;padding:50px 20px;color:#8b949e;">'
            '<div style="font-size:2.5rem;">💬</div>'
            '<div style="margin-top:10px;font-size:1.05rem;">Document indexed and ready.</div>'
            '<div style="font-size:0.85rem;margin-top:6px;">Ask your first question below.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        cols = st.columns(3, gap="medium")
        steps = [
            ("📤", "1. Upload", "Select a PDF using the sidebar file uploader."),
            ("⚡", "2. Process", "Click 'Process PDF' to chunk and index the document."),
            ("💬", "3. Ask", "Type any question about the document in the box below."),
        ]
        for col, (icon, title, desc) in zip(cols, steps):
            with col:
                st.markdown(
                    f'<div class="feat-card">'
                    f'<div class="feat-icon">{icon}</div>'
                    f'<div class="feat-title">{title}</div>'
                    f'<div class="feat-desc">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
else:
    for msg in st.session_state.chat_history:
        role    = msg["role"]
        content = msg["content"]
        meta    = msg.get("meta", {})
        is_err  = msg.get("is_error", False)

        if role == "user":
            st.markdown('<div class="label-user">You</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bubble-user">{content}</div>', unsafe_allow_html=True)

        else:
            if is_err:
                st.markdown('<div class="label-err">⚠ Error</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bubble-error">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="label-bot">● Assistant</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bubble-bot">{content}</div>', unsafe_allow_html=True)

                # Source pills
                if meta.get("sources"):
                    pills = "".join(
                        f'<span class="src-pill">📄 {s}</span>'
                        for s in meta["sources"]
                    )
                    st.markdown(pills, unsafe_allow_html=True)

                # Retrieved chunks (collapsed by default)
                if meta.get("chunks"):
                    with st.expander(f"🔍 Retrieved context ({len(meta['chunks'])} chunks)"):
                        for i, chunk in enumerate(meta["chunks"], 1):
                            st.markdown(
                                f'<div style="background:#1c2128;border-left:3px solid #21262d;'
                                f'border-radius:0 6px 6px 0;padding:10px 14px;margin-bottom:8px;">'
                                f'<span style="color:#58a6ff;font-size:0.75rem;font-family:'
                                f'\'IBM Plex Mono\',monospace;font-weight:600;">Chunk {i}</span>'
                                f'<div style="color:#8b949e;font-size:0.8rem;font-family:'
                                f'\'IBM Plex Mono\',monospace;margin-top:6px;line-height:1.6;">'
                                f'{chunk}</div></div>',
                                unsafe_allow_html=True,
                            )


# ─────────────────────────────────────────────────────────────────────────────
# Chat input — disabled until a document is loaded
# ─────────────────────────────────────────────────────────────────────────────

chat_disabled = st.session_state.vectorstore is None

question = st.chat_input(
    placeholder=(
        "Ask a question about your PDF…"
        if not chat_disabled
        else "Upload and process a PDF to start chatting"
    ),
    disabled=chat_disabled,
)

# ─────────────────────────────────────────────────────────────────────────────
# Handle submitted question
# ─────────────────────────────────────────────────────────────────────────────

if question:
    # Guard: API key
    if not config.groq_api_key:
        st.error("❌ Groq API key is missing. Enter it in the sidebar or add it to your .env file.")
        st.stop()

    # Guard: document
    if st.session_state.vectorstore is None:
        st.warning("⚠️ No document loaded. Upload and process a PDF first.")
        st.stop()

    # Append user message immediately
    st.session_state.chat_history.append({
        "role":     "user",
        "content":  question,
        "meta":     {},
        "is_error": False,
    })

    # Run pipeline inside spinner
    with st.spinner("Searching document and generating answer…"):
        try:
            result = ask(st.session_state.vectorstore, question)
            st.session_state.chat_history.append({
                "role":     "assistant",
                "content":  result["answer"],
                "meta":     {"sources": result["sources"], "chunks": result["chunks"]},
                "is_error": False,
            })
        except PipelineError as exc:
            st.session_state.chat_history.append({
                "role":     "assistant",
                "content":  str(exc),
                "meta":     {},
                "is_error": True,
            })
        except Exception as exc:
            st.session_state.chat_history.append({
                "role":     "assistant",
                "content":  f"Unexpected error: {exc}",
                "meta":     {},
                "is_error": True,
            })

    # Re-render with updated history
    st.rerun()

