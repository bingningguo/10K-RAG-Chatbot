"""
Streamlit UI for 10-K RAG Chatbot.
Sidebar: config, Clear Chat, Rebuild Index.
Main: chat loop, retrieved chunks in expander.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    build_faiss_index,
    create_rag_chain,
    get_config,
)

# Default PDF directory
DEFAULT_PDF_DIR = Path(__file__).parent / "10KFiles"


def _ensure_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        st.error(
            "GOOGLE_API_KEY is not set. Please set it in your environment:\n\n"
            "```\nexport GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY\n```\n\n"
            "Windows PowerShell:\n"
            "```\n$env:GOOGLE_API_KEY=\"YOUR_GOOGLE_API_KEY\"\n```"
        )
        st.stop()


def main() -> None:
    st.set_page_config(page_title="10-K RAG Chatbot", layout="wide")
    st.title("10-K RAG Chatbot: Alphabet, Amazon, Microsoft")

    _ensure_api_key()

    # Sidebar: config + controls
    with st.sidebar:
        st.header("Configuration")
        cfg = get_config()
        st.json(cfg)
        st.divider()
        st.header("Controls")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()  # Clear and refresh
        if st.button("Rebuild Index", use_container_width=True):
            if "faiss_index" in st.session_state:
                del st.session_state.faiss_index
            if "rag_chain" in st.session_state:
                del st.session_state.rag_chain
            st.rerun()

    # PDF selection / default paths
    pdf_dir = Path(os.environ.get("PDF_DIR", str(DEFAULT_PDF_DIR)))
    if not pdf_dir.exists():
        pdf_dir = Path(".")
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    if not pdf_files:
        st.warning(f"No PDFs found in {pdf_dir}. Add 10-K PDFs and rebuild.")
    else:
        st.caption(f"Using {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faiss_index" not in st.session_state and pdf_files:
        with st.spinner("Building FAISS index..."):
            try:
                st.session_state.faiss_index = build_faiss_index(pdf_files)
                st.session_state.rag_chain = create_rag_chain(st.session_state.faiss_index)
            except Exception as e:
                st.error(f"Index build failed: {e}")
                st.stop()
    elif not pdf_files:
        if "faiss_index" not in st.session_state:
            st.session_state.faiss_index = None
            st.session_state.rag_chain = None

    # Chat display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("retrieved_docs"):
                with st.expander("🔍 View Retrieved 10-K Evidence"):
                    for d in msg["retrieved_docs"]:
                        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
                        page = d.metadata.get("page", "?")
                        pg = page + 1 if isinstance(page, int) else page
                        st.write(f"**Source:** {src} (Page {pg})")
                        st.caption(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))

    # Chat input
    if prompt := st.chat_input("Ask about 10-K data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.get("rag_chain") is None:
            with st.chat_message("assistant"):
                st.error("Index not ready. Add PDFs and click Rebuild Index.")
            st.stop()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_history = []
                    for m in st.session_state.messages[:-1]:
                        role = "human" if m["role"] == "user" else "ai"
                        chat_history.append((role, m["content"]))
                    result = st.session_state.rag_chain(
                        prompt,
                        chat_history=chat_history,
                    )
                    answer = result["answer"]
                    retrieved = result["retrieved_docs"]
                except Exception as e:
                    st.error(f"RAG chain failed: {e}")
                    answer = "Sorry, an error occurred while generating the answer."
                    retrieved = []

            st.markdown(answer)
            with st.expander("🔍 View Retrieved 10-K Evidence"):
                for d in retrieved:
                    src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
                    page = d.metadata.get("page", "?")
                    pg = page + 1 if isinstance(page, int) else page
                    st.write(f"**Source:** {src} (Page {pg})")
                    st.caption(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "retrieved_docs": retrieved,
        })
        st.rerun()


if __name__ == "__main__":
    main()
