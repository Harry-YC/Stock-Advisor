"""
Ingestion Package for Literature Review

Provides document ingestion with vector DB storage for RAG:
- LocalEmbedder: Local embeddings using sentence-transformers
- VectorDB: Qdrant interface for vector storage
- Pipeline: Document ingestion orchestrator
- TableExtractor: PDF/DOCX table extraction
"""

from .embedder import LocalEmbedder
from .vector_store import VectorDB
from .pipeline import run_ingestion, chunk_text_parent_child
from .table_extractor import extract_pdf_with_tables, extract_docx_with_tables

__all__ = [
    "LocalEmbedder",
    "VectorDB",
    "run_ingestion",
    "chunk_text_parent_child",
    "extract_pdf_with_tables",
    "extract_docx_with_tables",
]
