"""Embedding service for memory similarity search.

This module provides a local embedding service using sentence-transformers
for generating vector embeddings of memory content.
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Model configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingService:
    """Local embedding service using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model which produces 384-dimensional
    embeddings and is optimized for semantic similarity.

    Example:
        service = EmbeddingService()
        embedding = await service.embed("AAPL reported strong earnings")
        similarities = await service.similarity(query_emb, [emb1, emb2])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        self._model_name = model_name
        self._model: Any = None
        self._logger = logger.bind(component="embedding_service")

    def _get_model(self) -> Any:
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._logger.info("embedding_model_loaded", model=self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            384-dimensional embedding vector.
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    async def similarity(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
    ) -> list[float]:
        """Compute cosine similarity between query and candidates.

        Args:
            query_embedding: The query vector.
            candidate_embeddings: List of candidate vectors to compare.

        Returns:
            List of similarity scores (0-1) for each candidate.
        """
        if not candidate_embeddings:
            return []

        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        candidates_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(candidates_norm, query_norm)

        return similarities.tolist()

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIM
