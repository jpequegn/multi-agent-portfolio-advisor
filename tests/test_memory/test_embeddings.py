"""Tests for the embedding service."""

import pytest

from src.memory.embeddings import EMBEDDING_DIM, EmbeddingService


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    @pytest.fixture
    def service(self) -> EmbeddingService:
        """Create embedding service for tests."""
        return EmbeddingService()

    def test_embedding_dim(self, service: EmbeddingService) -> None:
        """Test embedding dimension is correct."""
        assert service.embedding_dim == EMBEDDING_DIM
        assert service.embedding_dim == 384

    @pytest.mark.asyncio
    async def test_embed_single_text(self, service: EmbeddingService) -> None:
        """Test embedding a single text."""
        embedding = await service.embed("Apple reported strong earnings")

        assert isinstance(embedding, list)
        assert len(embedding) == EMBEDDING_DIM
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_returns_normalized_vector(self, service: EmbeddingService) -> None:
        """Test that embeddings are normalized (approximately unit length)."""
        import numpy as np

        embedding = await service.embed("Test text for embedding")
        norm = np.linalg.norm(embedding)

        # sentence-transformers returns normalized embeddings
        assert 0.99 < norm < 1.01

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, service: EmbeddingService) -> None:
        """Test embedding empty list returns empty list."""
        embeddings = await service.embed_batch([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_batch_multiple(self, service: EmbeddingService) -> None:
        """Test embedding multiple texts."""
        texts = [
            "Apple stock analysis",
            "Google earnings report",
            "Tesla quarterly results",
        ]
        embeddings = await service.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == EMBEDDING_DIM for emb in embeddings)

    @pytest.mark.asyncio
    async def test_similar_texts_have_high_similarity(self, service: EmbeddingService) -> None:
        """Test that semantically similar texts have high cosine similarity."""
        emb1 = await service.embed("Apple reported record quarterly revenue")
        emb2 = await service.embed("Apple announced strong financial results")
        emb3 = await service.embed("The weather is sunny today")

        similarities = await service.similarity(emb1, [emb2, emb3])

        # Similar financial texts should have higher similarity
        assert similarities[0] > similarities[1]
        assert similarities[0] > 0.5  # Reasonably high similarity

    @pytest.mark.asyncio
    async def test_similarity_empty_candidates(self, service: EmbeddingService) -> None:
        """Test similarity with empty candidates returns empty list."""
        query = await service.embed("Test query")
        similarities = await service.similarity(query, [])
        assert similarities == []

    @pytest.mark.asyncio
    async def test_similarity_single_candidate(self, service: EmbeddingService) -> None:
        """Test similarity with single candidate."""
        query = await service.embed("Stock market analysis")
        candidate = await service.embed("Stock market analysis")

        similarities = await service.similarity(query, [candidate])

        assert len(similarities) == 1
        # Same text should have very high similarity
        assert similarities[0] > 0.99

    @pytest.mark.asyncio
    async def test_similarity_range(self, service: EmbeddingService) -> None:
        """Test that similarity scores are in valid range."""
        query = await service.embed("Financial analysis")
        candidates = await service.embed_batch([
            "Stock analysis report",
            "Weather forecast for tomorrow",
            "Cooking recipe instructions",
        ])

        similarities = await service.similarity(query, candidates)

        # Cosine similarity should be between -1 and 1
        # For normalized vectors of natural language, typically 0 to 1
        for sim in similarities:
            assert -1.0 <= sim <= 1.0

    @pytest.mark.asyncio
    async def test_model_lazy_loading(self) -> None:
        """Test that model is lazily loaded."""
        service = EmbeddingService()

        # Model should not be loaded yet
        assert service._model is None

        # After embedding, model should be loaded
        await service.embed("Test")
        assert service._model is not None
