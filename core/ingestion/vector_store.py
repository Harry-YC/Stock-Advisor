"""
Vector Database Interface for Literature Review

Provides Qdrant interface for vector storage with support for:
- Local embedded mode (no server required)
- Cloud mode (Qdrant Cloud)
- Program/project filtering
- Document management (list, delete, clear)
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional qdrant import - graceful degradation if not installed
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None
    Distance = None
    VectorParams = None
    PointStruct = None
    logger.warning("qdrant_client not installed - vector database features disabled")


class VectorDB:
    """
    Qdrant vector database interface.

    Supports both embedded mode (local file storage) and cloud mode.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant connection.

        Args:
            config: Dictionary containing:
                - storage_path: Path for local storage (embedded mode)
                - collection_name: Name of the collection
                - vector_size: Embedding dimension
                - distance: Distance metric (Cosine, Euclidean, Dot)

                Environment variables (override config):
                - QDRANT_URL: Cloud endpoint URL
                - QDRANT_API_KEY: Cloud API key
        """
        self.config = config
        self.collection_name = config.get("collection_name", "literature_review_v1")
        self.vector_size = config.get("vector_size", 768)
        self.distance = config.get("distance", "Cosine")
        self.storage_path = config.get("storage_path", "./data/vector_storage")

        # Check for Qdrant Cloud configuration (env vars take precedence)
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            logger.info(f"Connecting to Qdrant Cloud at '{qdrant_url}'")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            self._is_cloud = True
        else:
            logger.info(f"Using Qdrant Embedded at '{self.storage_path}'")
            os.makedirs(self.storage_path, exist_ok=True)
            self.client = QdrantClient(path=self.storage_path)
            self._is_cloud = False

        # Ensure collection exists
        self.ensure_collection()

    def ensure_collection(self):
        """
        Check if collection exists, create if not, and validate schema.
        """
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        # Map string distance to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        desired_distance = distance_map.get(self.distance, Distance.COSINE)

        def create_collection():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=desired_distance
                )
            )
            logger.info(f"Created collection '{self.collection_name}'")

        if self.collection_name not in collection_names:
            create_collection()
            # Create payload indexes for filtering (required for cloud mode)
            self._ensure_payload_indexes()
            return

        # Validate existing collection schema
        try:
            info = self.client.get_collection(self.collection_name)
            current_size = info.config.params.vectors.size
            current_dist = info.config.params.vectors.distance

            if current_size != self.vector_size or current_dist != desired_distance:
                logger.warning(
                    f"Collection '{self.collection_name}' schema mismatch "
                    f"(size {current_size} vs {self.vector_size}, "
                    f"distance {current_dist} vs {desired_distance}). "
                    "Recreating collection."
                )
                self.client.delete_collection(self.collection_name)
                create_collection()
            else:
                logger.info(
                    f"Collection '{self.collection_name}' exists with matching schema."
                )
        except Exception as e:
            logger.error(f"Failed to validate collection schema: {e}")

        # Ensure indexes exist (for existing collections too)
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        """
        Create payload indexes required for filtering.

        Qdrant Cloud requires explicit payload indexes for filter operations.
        This creates keyword indexes on commonly filtered fields.
        """
        if not self._is_cloud:
            # Local embedded mode doesn't require explicit indexes
            return

        index_fields = ["project_id", "source"]

        for field_name in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.info(f"Created payload index on '{field_name}'")
            except Exception as e:
                # Index may already exist - this is fine
                if "already exists" not in str(e).lower():
                    logger.warning(f"Could not create index on '{field_name}': {e}")

    def upsert_batch(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Upsert a batch of vectors.

        Args:
            ids: List of unique IDs (if None, UUIDs will be generated).
            vectors: List of embedding vectors.
            payloads: List of metadata dictionaries.

        Returns:
            Tuple (success, error_message).
        """
        if not vectors:
            return False, "No vectors provided"

        points = []
        for i, vector in enumerate(vectors):
            # Use provided ID or generate UUID
            point_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
            payload = payloads[i] if payloads and i < len(payloads) else {}

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} points.")
            return True, ""
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")
            return False, str(e)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_filter: Optional[str] = None,
        project_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector.
            top_k: Number of results.
            source_filter: Optional source filename to filter by.
            project_filter: Optional project name to filter by.
        """
        if len(query_vector) != self.vector_size:
            logger.warning(
                f"Query vector dimension {len(query_vector)} does not match "
                f"collection dimension {self.vector_size}"
            )

        try:
            # Build filter conditions
            must_conditions = []

            if source_filter:
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter)
                    )
                )

            if project_filter:
                must_conditions.append(
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=project_filter)
                    )
                )

            query_filter = None
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )

            formatted_results = []
            for hit in results:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        project_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search by text using a provided embedder.

        Args:
            query_text: Query text to embed and search.
            embedder: Embedder instance for generating query vector.
            top_k: Number of results.
            source_filter: Optional source filename to filter by.
            project_filter: Optional project name to filter by.
        """
        query_vector = embedder.embed_query(query_text)
        return self.search(
            query_vector,
            top_k,
            source_filter=source_filter,
            project_filter=project_filter
        )

    def list_documents(self, project_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all unique documents (sources) in the collection.

        Args:
            project_filter: Optional project to filter by.

        Returns:
            List of dicts with 'source' and 'chunk_count'.
        """
        try:
            sources = {}
            offset = None

            # Build filter if project specified
            query_filter = None
            if project_filter:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_filter)
                        )
                    ]
                )

            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=query_filter
                )

                for point in results:
                    source = point.payload.get('source', 'Unknown')
                    sources[source] = sources.get(source, 0) + 1

                if offset is None:
                    break

            return [
                {"source": src, "chunk_count": count}
                for src, count in sources.items()
            ]

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def delete_by_source(self, source: str) -> bool:
        """
        Delete all chunks from a specific source document.

        Args:
            source: The source filename to delete.

        Returns:
            True if successful.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted all chunks from source: {source}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete source '{source}': {e}")
            return False

    def clear_all(self) -> bool:
        """
        Delete all points in the collection.

        Returns:
            True if successful.
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.ensure_collection()
            logger.info("Cleared all documents from collection")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dict with 'point_count', 'status', 'is_cloud'.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "point_count": info.points_count,
                "status": str(info.status),
                "is_cloud": self._is_cloud
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"point_count": 0, "status": "unknown", "is_cloud": self._is_cloud}

    def scroll_all_documents(
        self,
        project_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Scroll through all documents in the collection.

        Used for building BM25 index and other bulk operations.

        Args:
            project_filter: Optional project name to filter by.

        Returns:
            List of all document payloads with content and metadata.
        """
        try:
            all_docs = []
            offset = None

            # Build filter if project specified
            query_filter = None
            if project_filter:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="project_id",
                            match=models.MatchValue(value=project_filter)
                        )
                    ]
                )

            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                    scroll_filter=query_filter
                )

                for point in results:
                    all_docs.append(point.payload)

                if offset is None:
                    break

            logger.info(
                f"Scrolled {len(all_docs)} documents" +
                (f" (filter: {project_filter})" if project_filter else "")
            )
            return all_docs

        except Exception as e:
            logger.error(f"Failed to scroll documents: {e}")
            return []
