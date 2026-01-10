"""Tests for the memory manager."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.embeddings import EmbeddingService
from src.memory.manager import MemoryManager, get_memory_manager, reset_memory_manager
from src.memory.models import Memory, MemoryType
from src.memory.store import MemoryStore


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    async def manager(self) -> MemoryManager:
        """Create a temporary memory manager for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memories.db")
            store = MemoryStore(db_path=db_path)
            manager = MemoryManager(store=store)
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_initialize(self, manager: MemoryManager) -> None:
        """Test that manager initializes successfully."""
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_store_memory(self, manager: MemoryManager) -> None:
        """Test storing a memory."""
        memory = await manager.store(
            content="AAPL reported strong earnings",
            memory_type=MemoryType.EPISODIC,
            agent="research",
            metadata={"symbol": "AAPL"},
        )

        assert memory.id is not None
        assert memory.content == "AAPL reported strong earnings"
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.agent == "research"
        assert len(memory.embedding) == 384  # Auto-generated

    @pytest.mark.asyncio
    async def test_store_with_user_id(self, manager: MemoryManager) -> None:
        """Test storing a memory with user ID."""
        memory = await manager.store(
            content="User prefers dividend stocks",
            memory_type=MemoryType.PREFERENCE,
            agent="recommendation",
            user_id="user_123",
        )

        assert memory.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_retrieve_similar(self, manager: MemoryManager) -> None:
        """Test retrieving similar memories."""
        # Store some memories
        await manager.store(
            content="Apple reported record revenue",
            memory_type=MemoryType.EPISODIC,
            agent="research",
        )
        await manager.store(
            content="Google announced new AI products",
            memory_type=MemoryType.EPISODIC,
            agent="research",
        )

        # Retrieve similar to Apple query
        results = await manager.retrieve(
            query="Apple financial results",
            limit=2,
        )

        assert len(results) >= 1
        # Apple memory should be most similar
        assert "Apple" in results[0].memory.content

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, manager: MemoryManager) -> None:
        """Test retrieving memories with filters."""
        await manager.store(
            content="Research: AAPL analysis",
            memory_type=MemoryType.EPISODIC,
            agent="research",
        )
        await manager.store(
            content="Analysis: AAPL valuation",
            memory_type=MemoryType.EPISODIC,
            agent="analysis",
        )

        # Filter by agent
        results = await manager.retrieve(
            query="AAPL",
            agent="research",
        )

        assert len(results) >= 1
        assert all(r.memory.agent == "research" for r in results)

    @pytest.mark.asyncio
    async def test_get_context_for_agent(self, manager: MemoryManager) -> None:
        """Test getting context for agent prompt."""
        # Store various types of memories
        await manager.store(
            content="Previously analyzed AAPL: P/E was 28",
            memory_type=MemoryType.EPISODIC,
            agent="research",
            metadata={"symbol": "AAPL"},
        )
        await manager.store(
            content="AAPL typically beats earnings estimates",
            memory_type=MemoryType.SEMANTIC,
            agent="system",
            metadata={"symbol": "AAPL"},
        )

        context = await manager.get_context_for_agent(
            agent="research",
            symbols=["AAPL"],
        )

        assert "## Memory Context" in context
        assert "AAPL" in context

    @pytest.mark.asyncio
    async def test_get_context_with_preferences(self, manager: MemoryManager) -> None:
        """Test getting context includes user preferences."""
        await manager.store(
            content="User prefers dividend stocks",
            memory_type=MemoryType.PREFERENCE,
            agent="system",
            user_id="user_123",
        )

        context = await manager.get_context_for_agent(
            agent="research",
            symbols=["AAPL"],
            user_id="user_123",
        )

        assert "User Preferences" in context
        assert "dividend" in context

    @pytest.mark.asyncio
    async def test_get_context_empty(self, manager: MemoryManager) -> None:
        """Test getting context when no memories exist."""
        context = await manager.get_context_for_agent(
            agent="research",
            symbols=["XYZ"],
        )

        # Should return empty string when no relevant memories
        assert context == ""

    @pytest.mark.asyncio
    async def test_store_analysis_result(self, manager: MemoryManager) -> None:
        """Test convenience method for storing analysis results."""
        memory = await manager.store_analysis_result(
            agent="analysis",
            symbol="AAPL",
            summary="P/E ratio at 28.5, slightly overvalued",
            action="HOLD",
            metrics={"pe_ratio": 28.5},
        )

        assert "AAPL" in memory.content
        assert memory.metadata["symbol"] == "AAPL"
        assert memory.metadata["action"] == "HOLD"
        assert memory.metadata["metrics"]["pe_ratio"] == 28.5

    @pytest.mark.asyncio
    async def test_store_fact(self, manager: MemoryManager) -> None:
        """Test convenience method for storing facts."""
        memory = await manager.store_fact(
            content="NVDA has beaten earnings 8 of last 10 quarters",
            symbol="NVDA",
            fact_type="earnings",
        )

        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.metadata["symbol"] == "NVDA"
        assert memory.metadata["fact_type"] == "earnings"

    @pytest.mark.asyncio
    async def test_store_preference(self, manager: MemoryManager) -> None:
        """Test convenience method for storing preferences."""
        memory = await manager.store_preference(
            user_id="user_456",
            content="Prefers low-risk investments",
            preference_type="risk_tolerance",
        )

        assert memory.memory_type == MemoryType.PREFERENCE
        assert memory.user_id == "user_456"
        assert memory.metadata["preference_type"] == "risk_tolerance"

    @pytest.mark.asyncio
    async def test_run_maintenance(self, manager: MemoryManager) -> None:
        """Test running maintenance tasks."""
        # Store a memory
        await manager.store(
            content="Test memory",
            memory_type=MemoryType.EPISODIC,
            agent="research",
            importance=0.05,  # Low importance
        )

        result = await manager.run_maintenance()

        assert "decayed" in result
        assert "deleted" in result


class TestGlobalMemoryManager:
    """Tests for global memory manager functions."""

    @pytest.fixture(autouse=True)
    async def reset_global(self) -> None:
        """Reset global manager before each test."""
        await reset_memory_manager()
        yield
        await reset_memory_manager()

    @pytest.mark.asyncio
    async def test_get_memory_manager_creates_instance(self) -> None:
        """Test that get_memory_manager creates instance on first call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            with patch.dict("os.environ", {"MEMORY_DB_PATH": db_path}):
                manager = await get_memory_manager()
                assert manager is not None
                assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_get_memory_manager_returns_same_instance(self) -> None:
        """Test that get_memory_manager returns same instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            with patch.dict("os.environ", {"MEMORY_DB_PATH": db_path}):
                manager1 = await get_memory_manager()
                manager2 = await get_memory_manager()
                assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_reset_memory_manager(self) -> None:
        """Test that reset_memory_manager clears global instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            with patch.dict("os.environ", {"MEMORY_DB_PATH": db_path}):
                manager1 = await get_memory_manager()
                await reset_memory_manager()
                manager2 = await get_memory_manager()
                assert manager1 is not manager2


class TestMemoryManagerWithMocks:
    """Tests for MemoryManager with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_store_generates_embedding(self) -> None:
        """Test that storing memory generates embedding."""
        mock_store = AsyncMock(spec=MemoryStore)
        mock_embedding = MagicMock(spec=EmbeddingService)
        mock_embedding.embed = AsyncMock(return_value=[0.1] * 384)

        manager = MemoryManager(store=mock_store, embedding_service=mock_embedding)
        manager._initialized = True

        await manager.store(
            content="Test content",
            memory_type=MemoryType.EPISODIC,
            agent="research",
        )

        mock_embedding.embed.assert_called_once_with("Test content")
        mock_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_updates_accessed_at(self) -> None:
        """Test that retrieving memories updates accessed_at."""
        mock_store = AsyncMock(spec=MemoryStore)
        mock_embedding = MagicMock(spec=EmbeddingService)
        mock_embedding.embed = AsyncMock(return_value=[0.1] * 384)

        # Mock search to return a memory
        mock_memory = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Test",
            embedding=[0.1] * 384,
        )
        from src.memory.models import MemorySearchResult
        mock_store.search_similar = AsyncMock(return_value=[
            MemorySearchResult(memory=mock_memory, similarity=0.9)
        ])

        manager = MemoryManager(store=mock_store, embedding_service=mock_embedding)
        manager._initialized = True

        await manager.retrieve(query="test query")

        mock_store.update_accessed_at.assert_called_once_with(mock_memory.id)

    @pytest.mark.asyncio
    async def test_store_triggers_eviction(self) -> None:
        """Test that storing with user_id triggers LRU eviction check."""
        mock_store = AsyncMock(spec=MemoryStore)
        mock_embedding = MagicMock(spec=EmbeddingService)
        mock_embedding.embed = AsyncMock(return_value=[0.1] * 384)

        manager = MemoryManager(store=mock_store, embedding_service=mock_embedding)
        manager._initialized = True

        await manager.store(
            content="Test",
            memory_type=MemoryType.PREFERENCE,
            agent="system",
            user_id="user_123",
        )

        mock_store.evict_lru.assert_called_once_with("user_123")
