"""
Document Library Service

Provides persistent document storage with vector search.
Uploaded documents are chunked, embedded, and stored in Qdrant
for retrieval across sessions.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, BinaryIO
from dataclasses import dataclass

from core.document_ingestion import Document, ingest_file, DocumentIngestion
from core.ingestion.vector_store import VectorDB
from core.ingestion.embedder import LocalEmbedder

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document for embedding."""
    content: str
    source: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any]


class DocumentLibrary:
    """
    Persistent document storage with vector search.

    Uploads are:
    1. Ingested (PDF/DOCX/TXT/URL → Document)
    2. Chunked (split into ~500 char pieces)
    3. Embedded (sentence-transformers)
    4. Stored (Qdrant)

    Then retrievable via semantic search.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for shared state."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize document library.

        Args:
            project_id: Optional project ID for filtering
        """
        if self._initialized:
            return

        self.project_id = project_id or "default"

        # Initialize components
        try:
            self.embedder = LocalEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, faster model
                device="auto"
            )

            self.vector_db = VectorDB({
                "collection_name": "document_library",
                "vector_size": 384,  # MiniLM dimension
                "storage_path": "./data/document_library"
            })

            self.ingestion = DocumentIngestion()
            self._initialized = True
            logger.info("DocumentLibrary initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DocumentLibrary: {e}")
            self._initialized = False
            raise

    def add_document(
        self,
        file: BinaryIO,
        filename: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a document to the library.

        Args:
            file: File-like object
            filename: Original filename
            project_id: Optional project to associate with

        Returns:
            Dict with doc_id, chunk_count, status
        """
        project = project_id or self.project_id

        try:
            # 1. Ingest document
            doc = ingest_file(file, filename)
            logger.info(f"Ingested {filename}: {len(doc.content)} chars")

            # 2. Chunk document
            chunks = self._chunk_document(doc)
            logger.info(f"Created {len(chunks)} chunks from {filename}")

            if not chunks:
                return {
                    "doc_id": doc.id,
                    "chunk_count": 0,
                    "status": "empty",
                    "message": "No content extracted"
                }

            # 3. Embed chunks
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_documents(texts, show_progress=False)

            # 4. Store in vector DB
            ids = [f"{doc.id}_{c.chunk_index}" for c in chunks]
            payloads = [
                {
                    "content": c.content,
                    "source": c.source,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "project_id": project,
                    "title": doc.title,
                    **c.metadata
                }
                for c in chunks
            ]

            success, error = self.vector_db.upsert_batch(ids, embeddings, payloads)

            if success:
                logger.info(f"Stored {len(chunks)} chunks for {filename}")
                return {
                    "doc_id": doc.id,
                    "chunk_count": len(chunks),
                    "status": "success",
                    "title": doc.title
                }
            else:
                return {
                    "doc_id": doc.id,
                    "chunk_count": 0,
                    "status": "error",
                    "message": error
                }

        except Exception as e:
            logger.error(f"Failed to add document {filename}: {e}")
            return {
                "doc_id": "",
                "chunk_count": 0,
                "status": "error",
                "message": str(e)
            }

    def add_text(
        self,
        text: str,
        title: str = "Pasted Text",
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add pasted text to the library."""
        project = project_id or self.project_id

        try:
            doc = self.ingestion.ingest_text(text, title)
            chunks = self._chunk_document(doc)

            if not chunks:
                return {"doc_id": doc.id, "chunk_count": 0, "status": "empty"}

            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_documents(texts, show_progress=False)

            ids = [f"{doc.id}_{c.chunk_index}" for c in chunks]
            payloads = [
                {
                    "content": c.content,
                    "source": c.source,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "project_id": project,
                    "title": doc.title
                }
                for c in chunks
            ]

            success, _ = self.vector_db.upsert_batch(ids, embeddings, payloads)

            return {
                "doc_id": doc.id,
                "chunk_count": len(chunks) if success else 0,
                "status": "success" if success else "error",
                "title": doc.title
            }

        except Exception as e:
            logger.error(f"Failed to add text: {e}")
            return {"doc_id": "", "chunk_count": 0, "status": "error", "message": str(e)}

    def search(
        self,
        query: str,
        top_k: int = 10,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of results
            project_id: Optional project filter

        Returns:
            List of matching chunks with scores
        """
        project = project_id or self.project_id

        try:
            results = self.vector_db.search_by_text(
                query_text=query,
                embedder=self.embedder,
                top_k=top_k,
                project_filter=project
            )

            # Format results
            formatted = []
            for r in results:
                payload = r.get("payload", {})
                formatted.append({
                    "content": payload.get("content", ""),
                    "source": payload.get("source", "Unknown"),
                    "title": payload.get("title", ""),
                    "score": r.get("score", 0),
                    "doc_id": payload.get("doc_id", "")
                })

            return formatted

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def list_documents(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all documents in the library."""
        project = project_id or self.project_id
        return self.vector_db.list_documents(project_filter=project)

    def delete_document(self, source: str) -> bool:
        """Delete a document by source name."""
        return self.vector_db.delete_by_source(source)

    def clear_all(self) -> bool:
        """Clear all documents from the library."""
        return self.vector_db.clear_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        info = self.vector_db.get_collection_info()
        docs = self.list_documents()
        return {
            "total_chunks": info.get("point_count", 0),
            "document_count": len(docs),
            "is_cloud": info.get("is_cloud", False)
        }

    def _chunk_document(
        self,
        doc: Document,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[DocumentChunk]:
        """Split document into overlapping chunks."""
        text = doc.content
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence/paragraph boundary
            if end < len(text):
                # Look for good break points
                for sep in ["\n\n", ". ", "\n", " "]:
                    break_point = text.rfind(sep, start + chunk_size // 2, end)
                    if break_point > start:
                        end = break_point + len(sep)
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source=doc.metadata.get("filename", doc.title),
                    doc_id=doc.id,
                    chunk_index=chunk_index,
                    metadata={
                        "source_type": doc.source_type,
                        "year": doc.year,
                        "authors": doc.authors[:3] if doc.authors else []
                    }
                ))
                chunk_index += 1

            start = end - overlap

        return chunks


def get_document_library(project_id: Optional[str] = None) -> Optional[DocumentLibrary]:
    """
    Get document library instance, with graceful failure.

    Returns None if dependencies not available.
    """
    try:
        return DocumentLibrary(project_id)
    except Exception as e:
        logger.warning(f"Document library not available: {e}")
        return None
