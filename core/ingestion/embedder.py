"""
Local Embedder for Literature Review RAG

Uses sentence-transformers for local embedding generation.
Supports nomic-embed-text-v1.5 with proper query/document prefixes.
"""

import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """
    Local embedding model using sentence-transformers.

    Features:
    - Singleton pattern to avoid model reloading
    - Automatic device detection (CUDA, MPS, CPU)
    - Support for nomic-embed-text-v1.5 with prefixes
    - Batched embedding for efficiency
    """

    _instance = None
    _model = None
    _model_name = None
    _device = None
    _batch_size = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalEmbedder, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str = "auto",
        batch_size: Optional[int] = 32,
    ):
        """
        Initialize the local embedder.

        Args:
            model_name: Model to use. Default is nomic-embed-text-v1.5.
            device: Device to use ('auto', 'cpu', 'cuda', 'mps').
            batch_size: Batch size for encoding.
        """
        # Reload model if model_name changed
        if self._model is None or self._model_name != model_name:
            self._device = self._get_device(device)
            self._batch_size = batch_size or 32
            self._model_name = model_name

            logger.info(
                f"Initializing LocalEmbedder with model '{model_name}' "
                f"on device '{self._device}' (batch_size={self._batch_size})"
            )
            try:
                self._model = SentenceTransformer(
                    model_name,
                    device=self._device,
                    trust_remote_code=True
                )
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise e
        else:
            # Allow updating batch size without recreating the model
            if batch_size:
                self._batch_size = batch_size

    def _get_device(self, device_request: str) -> str:
        """Determine the best available device."""
        if device_request != "auto":
            return device_request

        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _is_nomic_model(self) -> bool:
        """Check if current model is a nomic model requiring prefixes."""
        return self._model_name and "nomic" in self._model_name.lower()

    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for documents.

        For nomic models, automatically adds 'search_document: ' prefix.

        Args:
            texts: List of document texts to embed.
            batch_size: Optional batch size override.
            show_progress: Show progress bar for large batches.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # Add document prefix for nomic models
        if self._is_nomic_model():
            texts = [f"search_document: {t}" for t in texts]

        effective_batch_size = batch_size or self._batch_size or 32
        all_embeddings = []

        iterator = range(0, len(texts), effective_batch_size)
        if show_progress and len(texts) > effective_batch_size:
            iterator = tqdm(iterator, desc="Embedding documents")

        for i in iterator:
            batch = texts[i:i + effective_batch_size]

            embeddings = self._model.encode(
                batch,
                batch_size=min(effective_batch_size, len(batch)),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for emb in embeddings:
                all_embeddings.append(emb.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        For nomic models, automatically adds 'search_query: ' prefix.

        Args:
            text: Query text to embed.

        Returns:
            Query embedding vector.
        """
        # Add query prefix for nomic models
        if self._is_nomic_model():
            text = f"search_query: {text}"

        embedding = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0].tolist()

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self._model.get_sentence_embedding_dimension()

    @classmethod
    def reset(cls):
        """Reset the singleton instance (useful for testing or model changes)."""
        cls._instance = None
        cls._model = None
        cls._model_name = None
        cls._device = None
        cls._batch_size = None
