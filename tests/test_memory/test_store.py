"""Tests for the memory store."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.memory.models import Memory, MemoryType
from src.memory.store import MemoryStore


class TestMemoryStore:
    """Tests for MemoryStore."""

    @pytest.fixture
    async def store(self) -> MemoryStore:
        """Create a temporary memory store for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_memories.db")
            store = MemoryStore(db_path=db_path)
            await store.initialize()
            yield store
            await store.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, store: MemoryStore) -> None:
        """Test that initialization creates required tables."""
        # If we got here, tables were created successfully
        assert store._connection is not None

    @pytest.mark.asyncio
    async def test_save_and_get(self, store: MemoryStore) -> None:
        """Test saving and retrieving a memory."""
        memory = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Analyzed AAPL",
            embedding=[0.1] * 384,
            metadata={"symbol": "AAPL"},
        )

        saved = await store.save(memory)
        assert saved.id == memory.id

        retrieved = await store.get(memory.id)
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == "Analyzed AAPL"
        assert retrieved.embedding == [0.1] * 384

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store: MemoryStore) -> None:
        """Test getting a nonexistent memory returns None."""
        result = await store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store: MemoryStore) -> None:
        """Test deleting a memory."""
        memory = Memory(
            memory_type=MemoryType.SEMANTIC,
            agent="analysis",
            content="Test memory",
            embedding=[0.2] * 384,
        )
        await store.save(memory)

        deleted = await store.delete(memory.id)
        assert deleted is True

        # Verify it's gone
        retrieved = await store.get(memory.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store: MemoryStore) -> None:
        """Test deleting nonexistent memory returns False."""
        deleted = await store.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_memories_all(self, store: MemoryStore) -> None:
        """Test listing all memories."""
        for i in range(5):
            await store.save(Memory(
                memory_type=MemoryType.EPISODIC,
                agent="research",
                content=f"Memory {i}",
                embedding=[0.1 * i] * 384,
            ))

        memories = await store.list_memories()
        assert len(memories) == 5

    @pytest.mark.asyncio
    async def test_list_memories_by_type(self, store: MemoryStore) -> None:
        """Test listing memories filtered by type."""
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Episodic 1",
            embedding=[0.1] * 384,
        ))
        await store.save(Memory(
            memory_type=MemoryType.SEMANTIC,
            agent="research",
            content="Semantic 1",
            embedding=[0.2] * 384,
        ))

        episodic = await store.list_memories(memory_type=MemoryType.EPISODIC)
        assert len(episodic) == 1
        assert episodic[0].memory_type == MemoryType.EPISODIC

        semantic = await store.list_memories(memory_type=MemoryType.SEMANTIC)
        assert len(semantic) == 1
        assert semantic[0].memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_list_memories_by_agent(self, store: MemoryStore) -> None:
        """Test listing memories filtered by agent."""
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Research memory",
            embedding=[0.1] * 384,
        ))
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="analysis",
            content="Analysis memory",
            embedding=[0.2] * 384,
        ))

        research = await store.list_memories(agent="research")
        assert len(research) == 1
        assert research[0].agent == "research"

    @pytest.mark.asyncio
    async def test_list_memories_by_user(self, store: MemoryStore) -> None:
        """Test listing memories filtered by user."""
        await store.save(Memory(
            memory_type=MemoryType.PREFERENCE,
            agent="recommendation",
            content="User 1 preference",
            user_id="user_1",
            embedding=[0.1] * 384,
        ))
        await store.save(Memory(
            memory_type=MemoryType.PREFERENCE,
            agent="recommendation",
            content="User 2 preference",
            user_id="user_2",
            embedding=[0.2] * 384,
        ))

        user1 = await store.list_memories(user_id="user_1")
        assert len(user1) == 1
        assert user1[0].user_id == "user_1"

    @pytest.mark.asyncio
    async def test_list_memories_limit_offset(self, store: MemoryStore) -> None:
        """Test listing memories with limit and offset."""
        for i in range(10):
            await store.save(Memory(
                memory_type=MemoryType.EPISODIC,
                agent="research",
                content=f"Memory {i}",
                embedding=[0.1 * i] * 384,
                importance=i / 10,  # Different importance for ordering
            ))

        # Get first 3
        first_3 = await store.list_memories(limit=3, offset=0)
        assert len(first_3) == 3

        # Get next 3
        next_3 = await store.list_memories(limit=3, offset=3)
        assert len(next_3) == 3

        # They should be different
        first_ids = {m.id for m in first_3}
        next_ids = {m.id for m in next_3}
        assert first_ids.isdisjoint(next_ids)

    @pytest.mark.asyncio
    async def test_search_similar(self, store: MemoryStore) -> None:
        """Test similarity search."""
        # Create memories with different embeddings
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Apple analysis",
            embedding=[1.0] + [0.0] * 383,  # Pointing in first dimension
        ))
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Google analysis",
            embedding=[0.0, 1.0] + [0.0] * 382,  # Pointing in second dimension
        ))

        # Search with query similar to Apple
        query_embedding = [0.9] + [0.1] * 383
        results = await store.search_similar(query_embedding, limit=2)

        assert len(results) == 2
        # Apple should be more similar
        assert results[0].memory.content == "Apple analysis"
        assert results[0].similarity > results[1].similarity

    @pytest.mark.asyncio
    async def test_search_similar_with_filters(self, store: MemoryStore) -> None:
        """Test similarity search with filters."""
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Research memory",
            embedding=[1.0] + [0.0] * 383,
        ))
        await store.save(Memory(
            memory_type=MemoryType.SEMANTIC,
            agent="analysis",
            content="Semantic memory",
            embedding=[0.9] + [0.1] * 383,  # Similar embedding
        ))

        query_embedding = [1.0] + [0.0] * 383

        # Filter by type
        episodic_results = await store.search_similar(
            query_embedding,
            memory_type=MemoryType.EPISODIC,
        )
        assert len(episodic_results) == 1
        assert episodic_results[0].memory.memory_type == MemoryType.EPISODIC

    @pytest.mark.asyncio
    async def test_search_similar_min_similarity(self, store: MemoryStore) -> None:
        """Test similarity search respects min_similarity threshold."""
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="High similarity",
            embedding=[1.0] + [0.0] * 383,
        ))
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Low similarity",
            embedding=[0.0, 1.0] + [0.0] * 382,  # Orthogonal
        ))

        query_embedding = [1.0] + [0.0] * 383

        # High threshold should filter out orthogonal
        results = await store.search_similar(query_embedding, min_similarity=0.5)
        assert len(results) == 1
        assert results[0].memory.content == "High similarity"

    @pytest.mark.asyncio
    async def test_update_accessed_at(self, store: MemoryStore) -> None:
        """Test updating accessed_at timestamp."""
        memory = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Test",
            embedding=[0.1] * 384,
        )
        await store.save(memory)
        original_accessed = memory.accessed_at

        # Wait a tiny bit and update
        import asyncio
        await asyncio.sleep(0.01)
        await store.update_accessed_at(memory.id)

        retrieved = await store.get(memory.id)
        assert retrieved is not None
        assert retrieved.accessed_at > original_accessed

    @pytest.mark.asyncio
    async def test_count(self, store: MemoryStore) -> None:
        """Test counting memories."""
        assert await store.count() == 0

        for i in range(5):
            await store.save(Memory(
                memory_type=MemoryType.EPISODIC,
                agent="research",
                content=f"Memory {i}",
                user_id="user_1" if i < 3 else "user_2",
                embedding=[0.1] * 384,
            ))

        assert await store.count() == 5
        assert await store.count(user_id="user_1") == 3
        assert await store.count(user_id="user_2") == 2

    @pytest.mark.asyncio
    async def test_evict_lru(self, store: MemoryStore) -> None:
        """Test LRU eviction."""
        # Create 5 memories for user_1
        for i in range(5):
            await store.save(Memory(
                memory_type=MemoryType.EPISODIC,
                agent="research",
                content=f"Memory {i}",
                user_id="user_1",
                embedding=[0.1] * 384,
            ))

        # Evict to max 3
        evicted = await store.evict_lru("user_1", max_memories=3)
        assert evicted == 2

        # Verify count
        assert await store.count(user_id="user_1") == 3

    @pytest.mark.asyncio
    async def test_evict_lru_under_limit(self, store: MemoryStore) -> None:
        """Test LRU eviction when under limit does nothing."""
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Memory",
            user_id="user_1",
            embedding=[0.1] * 384,
        ))

        evicted = await store.evict_lru("user_1", max_memories=10)
        assert evicted == 0

    @pytest.mark.asyncio
    async def test_delete_low_importance(self, store: MemoryStore) -> None:
        """Test deleting low importance memories."""
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="High importance",
            importance=0.8,
            embedding=[0.1] * 384,
        ))
        await store.save(Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Low importance",
            importance=0.05,
            embedding=[0.2] * 384,
        ))

        deleted = await store.delete_low_importance(threshold=0.1)
        assert deleted == 1

        memories = await store.list_memories()
        assert len(memories) == 1
        assert memories[0].content == "High importance"
