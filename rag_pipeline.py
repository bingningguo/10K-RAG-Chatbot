"""
RAG Pipeline: indexing, retriever, QA chain, and prompt for 10-K analysis.
Retriever uses ONLY the user question (no persona/history pollution).
Chat history is used ONLY in the generation stage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ============== CONFIG ==============
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 4
RETRIEVER_FETCH_K = 20
RETRIEVER_MMR_LAMBDA = 0.5  # 0 = more diverse, 1 = more relevant
MODEL_GEMINI = "gemini-2.5-flash"
MODEL_OPENAI = "gpt-4o"
# Embedding model supported by Google Generative AI
EMBEDDING_MODEL = "models/gemini-embedding-001"
TEMPERATURE = 0.5

RAG_SYSTEM_PROMPT = """You are a strict RAG assistant. You must ONLY use the provided document context. Do NOT use prior knowledge.

RULES:
1. Every factual statement MUST include citation in this exact format: (source: filename, page X)
2. If the answer is not explicitly supported by the context, say: "I don't have enough information to answer this question."
3. Do NOT make up numbers, dates, or facts.

For comparison questions (e.g., comparing Alphabet, Amazon, Microsoft):
- Separate sections by company name
- Provide citations for each company section

For numeric questions (cash, revenue, liquidity, etc.):
- Quote the EXACT number as written in the document
- Include citation immediately after the number

If you cannot cite a source for a claim, do NOT state it.

STRONG FORMAT REQUIREMENT: You MUST support your factual claims by appending a citation tag in this exact format: CIT[Company_Name, Page_X]. If you cannot find the specific number or fact in the context, you MUST reply: "I cannot find evidence in the provided 10-K documents." """


def _ensure_api_key() -> None:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Please set it in your environment:\n"
            "  export GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY\n"
            "  # Windows PowerShell: $env:GOOGLE_API_KEY=\"YOUR_GOOGLE_API_KEY\"\n"
            "  # Windows CMD: setx GOOGLE_API_KEY \"YOUR_GOOGLE_API_KEY\""
        )


def load_and_split_pdfs(pdf_paths: list[Path]) -> list[Document]:
    """Load PDFs and split into chunks."""
    all_docs: list[Document] = []
    for p in pdf_paths:
        if not p.exists():
            continue
        loader = PyPDFLoader(str(p))
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = p.name
        all_docs.extend(docs)
    if not all_docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(all_docs)


def build_faiss_index(
    pdf_paths: list[Path],
) -> FAISS:
    """Build FAISS index from PDF paths."""
    _ensure_api_key()
    chunks = load_and_split_pdfs(pdf_paths)
    if not chunks:
        raise ValueError("No documents loaded. Check PDF paths.")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def create_retriever(vectorstore: FAISS) -> Any:
    """Create MMR retriever; query uses ONLY the question."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_MMR_LAMBDA,
        },
    )


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        page_raw = d.metadata.get("page", 0)
        page = page_raw + 1 if isinstance(page_raw, int) else page_raw
        parts.append(f"[{i + 1}] (source: {src}, page {page})\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def create_rag_chain(vectorstore: FAISS, model: str = "gemini"):
    """Create RAG chain. model: 'gemini' | 'openai'."""
    retriever = create_retriever(vectorstore)
    if model == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set for OpenAI backend.")
        llm = ChatOpenAI(
            model=MODEL_OPENAI,
            temperature=TEMPERATURE,
            openai_api_key=key,
        )
    else:
        _ensure_api_key()
        llm = ChatGoogleGenerativeAI(
            model=MODEL_GEMINI,
            temperature=TEMPERATURE,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def retrieve(query: str) -> tuple[str, list[Document]]:
        docs = retriever.invoke(query)
        return _format_context(docs), docs

    def run(question: str, chat_history: Optional[list] = None) -> dict:
        ctx, docs = retrieve(question)
        messages = []
        if chat_history:
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
        full_prompt = prompt.invoke({
            "context": ctx,
            "chat_history": messages,
            "question": question,
        })
        response = llm.invoke(full_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "answer": answer,
            "retrieved_docs": docs,
            "context": ctx,
        }

    return run


def get_config(model: str = "gemini") -> dict[str, Any]:
    """Return config for display."""
    llm_name = MODEL_OPENAI if model == "openai" else MODEL_GEMINI
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retriever_k": RETRIEVER_K,
        "retriever_fetch_k": RETRIEVER_FETCH_K,
        "embedding_model": EMBEDDING_MODEL,
        "model": model,
        "model_name": llm_name,
        "temperature": TEMPERATURE,
    }
