"""
Medical RAG Assistant - Streamlit chat application.
Run from project root: streamlit run medical_rag_assistant/app.py
Or from this directory: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure package root is on path when running as script
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import streamlit as st
from langchain_core.documents import Document

from rag.document_loader import load_document, load_documents_from_directory
from rag.rag_pipeline import index_documents, index_data_folder, run_rag_stream
from utils.config import DATA_DIR, SAMPLE_DOCS_DIR, validate_config

UPLOADS_DIR = DATA_DIR / "uploads"

# Page config
st.set_page_config(page_title="Medical AI Assistant", page_icon="🩺", layout="wide")

# Medical disclaimer at top
st.markdown(
    """
    **⚠️ Disclaimer:** This tool provides medical information from documents and should not replace professional medical advice.
    Always consult a qualified healthcare provider for medical decisions.
    """
)
st.divider()

st.title("🩺 Medical AI Assistant")
st.caption("Ask questions based on the loaded medical documents. Answers are grounded in your knowledge base.")

# Sidebar
with st.sidebar:
    st.header("Options")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Indexed documents")
    def _list_docs(dir_path: Path) -> list[str]:
        if not dir_path.is_dir():
            return []
        return sorted(
            p.name for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
        )
    sample_files = _list_docs(SAMPLE_DOCS_DIR)
    upload_files = _list_docs(UPLOADS_DIR)
    all_files = sample_files + upload_files
    if all_files:
        for f in all_files:
            st.text(f"• {f}")
    else:
        st.info("No PDF/TXT in data folder or uploads yet. Add files to data/sample_medical_docs or upload above.")

    st.subheader("Re-index knowledge base")
    if st.button("Re-index documents", use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                n = index_data_folder()
                st.success(f"Indexed {n} chunks.")
            except Exception as e:
                st.error(str(e))

    st.subheader("Upload new document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="upload")
    if uploaded is not None:
        with st.spinner("Processing and indexing..."):
            try:
                path = Path(uploaded.name)
                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                file_path = UPLOADS_DIR / path.name
                with open(file_path, "wb") as f:
                    f.write(uploaded.getvalue())
                docs = load_document(file_path)
                count = index_documents(docs)
                st.success(f"Indexed {count} chunks from {path.name}.")
            except Exception as e:
                st.error(str(e))

# Ensure messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    page = s.get("page")
                    st.text(f"{s.get('source', 'Unknown')}" + (f" – Page {page}" if page is not None else ""))

# Chat input
if prompt := st.chat_input("Ask a medical question..."):
    # Validate config before running
    errors = validate_config()
    if errors:
        st.error(f"Missing environment variables: {', '.join(errors)}. Please set them in .env.")
        st.stop()

    # Initial index on first query if needed (optional: could index at startup)
    if "indexed" not in st.session_state:
        with st.spinner("Preparing knowledge base..."):
            try:
                index_data_folder()
                st.session_state.indexed = True
            except Exception as e:
                st.warning(f"Indexing skipped or failed: {e}")
                st.session_state.indexed = True

    st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        sources_placeholder = st.empty()
        placeholder.markdown("Searching your documents...")
        full = ""
        sources = []
        try:
            for content, srcs in run_rag_stream(prompt):
                if content:
                    full += content
                    placeholder.markdown(full + "▌")
                if srcs is not None:
                    sources = srcs
            placeholder.markdown(full)
            if sources:
                with sources_placeholder.expander("Sources"):
                    for s in sources:
                        page = s.get("page")
                        st.text(f"{s.get('source', 'Unknown')}" + (f" – Page {page}" if page is not None else ""))
        except Exception as e:
            st.error(str(e))
            full = f"Error: {e}"
            sources = []

    st.session_state.messages.append({"role": "assistant", "content": full, "sources": sources})
    st.rerun()
