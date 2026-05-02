"""
rag_pipeline.py
---------------
Complete RAG pipeline: PDF → chunks → embeddings → FAISS → LLM answer.

Design rules:
  - Zero Streamlit imports
  - Zero print() statements  
  - All failures raise PipelineError (user-friendly message)
  - Returns structured dicts — UI decides how to display them
  - Module-level caching for the embedding model (expensive to load)

Public API:
    process_pdf(uploaded_file) -> FAISSVectorStore  |  raises PipelineError
    ask(vectorstore, question) -> dict              |  raises PipelineError
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

# ── LangChain imports — using the CORRECT modern split-package structure ───────
# text_splitter moved to langchain_text_splitters in LangChain ≥ 0.2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# document loaders live in langchain_community
from langchain_community.document_loaders import PyPDFLoader

# FAISS vectorstore — langchain_community
from langchain_community.vectorstores import Chroma

# Core types
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Groq LLM — langchain_groq
from langchain_groq import ChatGroq

# HuggingFace embeddings — langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings

from config import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────

class PipelineError(Exception):
    """
    Raised for ALL recoverable, user-facing pipeline failures.
    The UI catches this and displays the message directly — keep them readable.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model  (module-level cache — load once per process)
# ─────────────────────────────────────────────────────────────────────────────

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Lazy-load the sentence-transformers embedding model and cache it.

    Loading takes ~5 seconds on first call; subsequent calls are instant.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", config.embedding_model)
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            raise PipelineError(
                f"Could not load embedding model '{config.embedding_model}'.\n"
                f"Make sure sentence-transformers is installed: pip install sentence-transformers\n"
                f"Detail: {exc}"
            ) from exc
    return _embeddings


# ─────────────────────────────────────────────────────────────────────────────
# LLM  (new instance each call — picks up runtime key changes)
# ─────────────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    """
    Build a ChatGroq instance from config values.

    Model is ALWAYS config.llm_model — never a hardcoded string.
    """
    if not config.groq_api_key:
        raise PipelineError(
            "Groq API key is missing.\n"
            "  → Enter it in the sidebar, or add GROQ_API_KEY=gsk_... to your .env file."
        )
    return ChatGroq(
        model=config.llm_model,            # ← config only, NEVER hardcoded
        api_key=config.groq_api_key,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PDF loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Parse a PDF from raw bytes into a list of LangChain Documents.

    PyPDFLoader requires a real file path, so we write to a temp file
    and clean it up in the finally block.

    Args:
        file_bytes:  Raw bytes from the uploaded file.
        filename:    Original filename (stored in document metadata).

    Returns:
        List of Documents, one per page.

    Raises:
        PipelineError: Empty file, unreadable PDF, or no extractable text.
    """
    if not file_bytes:
        raise PipelineError("The uploaded file appears to be empty.")

    tmp_path: str | None = None
    try:
        # Write bytes to a named temp file — delete=False so PyPDFLoader can open it
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            f"Could not read '{filename}'. Make sure it is a valid, non-corrupted PDF.\n"
            f"Detail: {exc}"
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not docs:
        raise PipelineError(
            f"No text was extracted from '{filename}'.\n"
            "The file may be a scanned/image-only PDF. Try a text-based PDF."
        )

    # Tag every page with the original filename for source attribution
    for doc in docs:
        doc.metadata["source"] = filename

    logger.info("Loaded %d page(s) from '%s'", len(docs), filename)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split full-page documents into smaller overlapping chunks.

    Chunk size / overlap come from config — never hardcoded.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise PipelineError(
            "The document produced no text chunks after splitting.\n"
            "The PDF may contain only images or non-extractable content."
        )

    logger.info("Split into %d chunk(s) (size=%d, overlap=%d)",
                len(chunks), config.chunk_size, config.chunk_overlap)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public: process_pdf
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(uploaded_file: Any) -> Chroma:
    """
    Full ingestion pipeline: upload → parse → chunk → embed → FAISS index.

    Args:
        uploaded_file:  Streamlit UploadedFile object (.read(), .name).

    Returns:
        FAISS vectorstore ready for similarity search.

    Raises:
        PipelineError: Any step-level failure with a clear user message.
    """
    filename = getattr(uploaded_file, "name", "document.pdf")
    file_bytes = uploaded_file.read()

    docs   = _load_pdf(file_bytes, filename)
    chunks = _chunk_documents(docs)
    emb    = _get_embeddings()
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


    logger.info("Building chroma index from %d chunks…", len(docs))
    try:
        vectorstore = Chroma.from_documents(docs, embeddings)
    except Exception as exc:
        raise PipelineError(
            f"Failed to build the vector search index.\nDetail: {exc}"
        ) from exc

    logger.info("Chroma vector store ready")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Public: ask
# ─────────────────────────────────────────────────────────────────────────────

def ask(vectorstore, question: str) -> dict:
    """
    Retrieve relevant chunks and generate an LLM-grounded answer.

    Args:
        vectorstore:  FAISS index from process_pdf().
        question:     The user's question string.

    Returns:
        {
            "answer":  str,          # LLM response
            "sources": list[str],    # Filenames of contributing chunks
            "chunks":  list[str],    # Trimmed text of retrieved chunks
        }

    Raises:
        PipelineError: Retrieval failure or LLM error, with a clear message.
    """
    question = question.strip()
    if not question:
        raise PipelineError("Please enter a question before submitting.")

    # ── Retrieve ────────────────────────────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.retrieval_k},
    )

    try:
        retrieved_docs = retriever.invoke(question)
    except Exception as exc:
        raise PipelineError(f"Vector search failed.\nDetail: {exc}") from exc

    if not retrieved_docs:
        return {
            "answer":  (
                "I couldn't find any relevant content in the document "
                "to answer your question. Try rephrasing."
            ),
            "sources": [],
            "chunks":  [],
        }

    # ── Build context ────────────────────────────────────────────────────────
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = sorted({doc.metadata.get("source", "document") for doc in retrieved_docs})
    chunks  = [doc.page_content[:350].strip() + "…" for doc in retrieved_docs]

    # ── Prompt ───────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", config.system_prompt),
        ("human", "{question}"),
    ])

    # ── LLM chain (LCEL) ─────────────────────────────────────────────────────
    # context is captured in closure; question flows through RunnablePassthrough
    llm = _get_llm()

    chain = (
        {
            "context":  RunnableLambda(lambda _: context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # ── Invoke with targeted error handling ──────────────────────────────────
    try:
        answer = chain.invoke(question)
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "authentication" in msg or "api_key" in msg:
            raise PipelineError(
                "Groq API authentication failed.\n"
                "Check that your GROQ_API_KEY is correct and active."
            ) from exc
        if "429" in msg or "rate_limit" in msg:
            raise PipelineError(
                "Groq API rate limit hit. Wait a few seconds and try again."
            ) from exc
        if "model" in msg and ("not found" in msg or "does not exist" in msg):
            raise PipelineError(
                f"Model '{config.llm_model}' was not found on Groq.\n"
                "Check config.py → llm_model and verify the model name at console.groq.com."
            ) from exc
        raise PipelineError(f"LLM call failed.\nDetail: {exc}") from exc

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }
