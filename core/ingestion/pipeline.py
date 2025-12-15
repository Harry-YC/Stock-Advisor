"""
Document Ingestion Pipeline

Orchestrates the full ingestion workflow:
1. Extract text from documents (PDF, DOCX, PPTX, TXT)
2. Chunk with parent-child strategy for optimal retrieval
3. Generate embeddings
4. Store in vector database

Usage:
    from core.ingestion import run_ingestion

    result = run_ingestion(
        file_path="document.pdf",
        project_id="my_project"
    )
    if result['success']:
        print(f"Indexed {result['chunks']} chunks")
"""

import hashlib
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO

from .embedder import LocalEmbedder
from .vector_store import VectorDB
from .table_extractor import (
    extract_pdf_with_tables,
    extract_docx_with_tables,
    extract_pptx_with_tables,
    ExtractionResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# Chunking
# ============================================================================

def chunk_text_parent_child(
    text: str,
    parent_size: int = 2000,
    child_size: int = 200,
    child_overlap: int = 50,
    tables: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Create parent-child chunk structure for enhanced retrieval.

    Parent chunks are large (2000 chars) for context preservation.
    Child chunks are small (200 chars) for precise embedding matching.
    When a child matches, the parent is returned to the LLM.

    Args:
        text: Document text
        parent_size: Size of parent chunks (chars)
        child_size: Size of child chunks (chars)
        child_overlap: Overlap between child chunks
        tables: Extracted tables (optional, for metadata)

    Returns:
        List of child chunk dicts with parent reference
    """
    if not text.strip():
        return []

    logger.info(
        f"[CHUNKING] Creating chunks with parent_size={parent_size}, "
        f"child_size={child_size}"
    )

    # Step 1: Create parent chunks
    parent_chunks = _create_parent_chunks(text, parent_size)
    logger.info(f"[CHUNKING] Created {len(parent_chunks)} parent chunks")

    # Step 2: Split each parent into children
    all_children = []
    for parent_idx, parent in enumerate(parent_chunks):
        parent_content = parent['content']
        parent_id = hashlib.md5(parent_content.encode()).hexdigest()[:16]

        # Create children from this parent
        children = _split_into_children(parent_content, child_size, child_overlap)

        for child_idx, child_content in enumerate(children):
            all_children.append({
                'content': child_content,
                'parent_content': parent_content,
                'parent_id': parent_id,
                'parent_index': parent_idx,
                'child_index': child_idx,
                'chunk_type': 'child',
                'has_table': '|' in child_content and '---' in child_content
            })

    logger.info(
        f"[CHUNKING] Created {len(all_children)} child chunks from "
        f"{len(parent_chunks)} parents"
    )
    return all_children


def _create_parent_chunks(text: str, parent_size: int) -> List[Dict[str, Any]]:
    """Create parent chunks respecting sentence boundaries."""
    import re

    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    parents = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence exceeds parent size, flush current chunk
        if current_size + sentence_len > parent_size and current_chunk:
            parent_content = ' '.join(current_chunk)
            parents.append({
                'content': parent_content,
                'has_table': '|' in parent_content and '---' in parent_content
            })
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_len + 1  # +1 for space

    # Flush remaining
    if current_chunk:
        parent_content = ' '.join(current_chunk)
        parents.append({
            'content': parent_content,
            'has_table': '|' in parent_content and '---' in parent_content
        })

    return parents


def _split_into_children(text: str, child_size: int, overlap: int) -> List[str]:
    """Split parent text into overlapping child chunks."""
    if len(text) <= child_size:
        return [text]

    children = []
    stride = max(1, child_size - overlap)

    for i in range(0, len(text), stride):
        child = text[i:i + child_size].strip()
        if len(child) > 20:  # Minimum viable chunk
            children.append(child)

        # Stop if we've passed the end
        if i + child_size >= len(text):
            break

    return children


def chunk_text_simple(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Simple fixed-size chunking (fallback).

    Args:
        text: Document text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks

    Returns:
        List of chunk dicts
    """
    if not text.strip():
        return []

    chunks = []
    stride = max(1, chunk_size - overlap)

    for i in range(0, len(text), stride):
        chunk_text = text[i:i + chunk_size].strip()
        if len(chunk_text) > 20:
            chunks.append({
                'content': chunk_text,
                'chunk_type': 'simple',
                'has_table': '|' in chunk_text and '---' in chunk_text
            })

        if i + chunk_size >= len(text):
            break

    return chunks


# ============================================================================
# Pipeline
# ============================================================================

def run_ingestion(
    file_input: Union[str, Path, BytesIO, BinaryIO],
    config: Optional[Dict[str, Any]] = None,
    embedder: Optional[LocalEmbedder] = None,
    vector_db: Optional[VectorDB] = None,
    project_id: str = "default",
    filename: Optional[str] = None,
    use_parent_child: bool = True,
    parent_size: int = 2000,
    child_size: int = 200
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline for a single file.

    Args:
        file_input: Path to document or file-like object
        config: Configuration dict (optional)
        embedder: Existing embedder instance (optional)
        vector_db: Existing vector DB instance (optional)
        project_id: Project identifier for filtering
        filename: Filename (required if file_input is BytesIO)
        use_parent_child: Use parent-child chunking strategy
        parent_size: Parent chunk size in characters
        child_size: Child chunk size in characters

    Returns:
        Dict with ingestion results:
        - success: bool
        - filename: str
        - chunks: int
        - tables: int
        - doc_type: str
        - extraction_method: str
        - error: str (if failed)
    """
    # Determine filename
    if filename is None:
        if isinstance(file_input, (str, Path)):
            filename = Path(file_input).name
        else:
            raise ValueError("filename is required when file_input is a file-like object")

    logger.info(f"=== Starting Ingestion for {filename} ===")

    # 1. Get file extension
    file_ext = Path(filename).suffix.lower()

    # 2. Extract text based on file type
    logger.info("Extracting text...")
    extraction: ExtractionResult

    try:
        if file_ext == '.pdf':
            extraction = extract_pdf_with_tables(file_input)
        elif file_ext in ('.docx', '.doc'):
            extraction = extract_docx_with_tables(file_input)
        elif file_ext in ('.pptx', '.ppt'):
            extraction = extract_pptx_with_tables(file_input)
        elif file_ext in ('.txt', '.md'):
            if isinstance(file_input, (str, Path)):
                with open(file_input, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            else:
                text = file_input.read()
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='ignore')
            extraction = ExtractionResult(
                text=text,
                tables=[],
                page_count=1,
                extraction_method="plaintext"
            )
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            return {'success': False, 'error': f'Unsupported file type: {file_ext}'}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {'success': False, 'error': f'Extraction failed: {str(e)}'}

    text = extraction.text
    tables = extraction.tables
    extraction_method = extraction.extraction_method

    if not text.strip():
        logger.warning("No text extracted.")
        return {'success': False, 'error': 'No text extracted'}

    logger.info(f"Extracted {len(text)} characters using {extraction_method}")
    if tables:
        logger.info(f"Found {len(tables)} tables")

    # 3. Chunk text
    logger.info("Chunking text...")

    if use_parent_child:
        chunks = chunk_text_parent_child(
            text,
            parent_size=parent_size,
            child_size=child_size,
            tables=tables
        )
    else:
        chunks = chunk_text_simple(text, chunk_size=512, overlap=50)

    if not chunks:
        logger.warning("No chunks created.")
        return {'success': False, 'error': 'No chunks created'}

    logger.info(f"Created {len(chunks)} chunks")

    # 4. Generate embeddings
    logger.info("Generating embeddings...")

    if embedder is None:
        from config import settings
        embedder = LocalEmbedder(
            model_name=getattr(settings, 'EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5'),
            device=getattr(settings, 'EMBEDDING_DEVICE', 'auto'),
            batch_size=getattr(settings, 'EMBEDDING_BATCH_SIZE', 32),
        )

    # Extract content strings for embedding
    chunk_contents = [c['content'] for c in chunks]

    embeddings = embedder.embed_documents(chunk_contents)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # 5. Prepare payloads
    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "content": chunk['content'],
            "source": filename,
            "project_id": project_id,
            "chunk_index": i,
            "extraction_method": extraction_method,
            "has_table": chunk.get('has_table', False),
        }

        # Add parent-child metadata
        if 'parent_content' in chunk:
            payload['parent_content'] = chunk['parent_content']
        if 'parent_id' in chunk:
            payload['parent_id'] = chunk['parent_id']
        if 'chunk_type' in chunk:
            payload['chunk_type'] = chunk['chunk_type']

        payloads.append(payload)

    # 6. Generate IDs (deterministic to avoid dupes)
    ids = [
        hashlib.md5(f"{filename}:{c}".encode()).hexdigest()
        for c in chunk_contents
    ]

    # 7. Upsert to VectorDB
    logger.info("Storing in Qdrant...")

    if vector_db is None:
        from config import settings
        vector_db_config = {
            "storage_path": str(getattr(settings, 'QDRANT_STORAGE_PATH', './data/vector_storage')),
            "collection_name": getattr(settings, 'QDRANT_COLLECTION_NAME', 'literature_review_v1'),
            "vector_size": embedder.get_embedding_dimension(),
            "distance": "Cosine"
        }
        vector_db = VectorDB(vector_db_config)

    success, error_msg = vector_db.upsert_batch(ids, embeddings, payloads)

    if success:
        logger.info(f"Successfully indexed {filename}: {len(chunks)} chunks")
        return {
            'success': True,
            'filename': filename,
            'chunks': len(chunks),
            'tables': len(tables),
            'extraction_method': extraction_method,
            'project_id': project_id
        }
    else:
        logger.error(f"Failed to index {filename}: {error_msg}")
        return {'success': False, 'error': f'Failed to store in vector DB: {error_msg}'}


def delete_document(
    filename: str,
    vector_db: Optional[VectorDB] = None
) -> bool:
    """
    Delete a document from the vector database.

    Args:
        filename: The source filename to delete
        vector_db: Optional existing VectorDB instance

    Returns:
        True if successful
    """
    if vector_db is None:
        from config import settings
        vector_db_config = {
            "storage_path": str(getattr(settings, 'QDRANT_STORAGE_PATH', './data/vector_storage')),
            "collection_name": getattr(settings, 'QDRANT_COLLECTION_NAME', 'literature_review_v1'),
            "vector_size": 768,
            "distance": "Cosine"
        }
        vector_db = VectorDB(vector_db_config)

    return vector_db.delete_by_source(filename)


def list_documents(
    project_id: Optional[str] = None,
    vector_db: Optional[VectorDB] = None
) -> List[Dict[str, Any]]:
    """
    List all indexed documents.

    Args:
        project_id: Optional project filter
        vector_db: Optional existing VectorDB instance

    Returns:
        List of documents with source and chunk_count
    """
    if vector_db is None:
        from config import settings
        vector_db_config = {
            "storage_path": str(getattr(settings, 'QDRANT_STORAGE_PATH', './data/vector_storage')),
            "collection_name": getattr(settings, 'QDRANT_COLLECTION_NAME', 'literature_review_v1'),
            "vector_size": 768,
            "distance": "Cosine"
        }
        vector_db = VectorDB(vector_db_config)

    return vector_db.list_documents(project_filter=project_id)


class IngestionPipeline:
    """
    Wrapper class for run_ingestion function to provide an object-oriented interface.
    Used by services like ZoteroSyncService.
    """
    def __init__(self, settings_config: Optional[Any] = None):
        self.settings = settings_config

    def run(self, documents: List[Any], **kwargs):
        """
        Run ingestion for a list of document objects (from core.document_ingestion.Document).
        
        Args:
            documents: List of Document objects
        """
        results = []
        for doc in documents:
            # Convert Document object content to a file-like stream for ingestion
            # We use BytesIO to simulate a file
            from io import BytesIO
            
            # Use content from the Document object
            file_stream = BytesIO(doc.content.encode('utf-8'))
            filename = doc.metadata.get('filename', f"{doc.title}.txt")
            
            # Run ingestion
            res = run_ingestion(
                file_input=file_stream,
                filename=filename,
                project_id="default",  # Default project for Zotero sync
                use_parent_child=True
            )
            results.append(res)
        return results

