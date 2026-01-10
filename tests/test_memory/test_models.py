"""Tests for memory models."""

from datetime import datetime

import pytest

from src.memory.models import Memory, MemorySearchResult, MemoryType


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_episodic_value(self) -> None:
        """Test EPISODIC type has correct value."""
        assert MemoryType.EPISODIC.value == "episodic"

    def test_semantic_value(self) -> None:
        """Test SEMANTIC type has correct value."""
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_preference_value(self) -> None:
        """Test PREFERENCE type has correct value."""
        assert MemoryType.PREFERENCE.value == "preference"


class TestMemory:
    """Tests for Memory dataclass."""

    def test_required_fields(self) -> None:
        """Test creating memory with required fields."""
        memory = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Test memory content",
        )
        assert memory.memory_type == MemoryType.EPISODIC
        assert memory.agent == "research"
        assert memory.content == "Test memory content"
        assert memory.id is not None  # Auto-generated
        assert memory.importance == 0.5  # Default

    def test_optional_fields(self) -> None:
        """Test memory with optional fields."""
        memory = Memory(
            memory_type=MemoryType.SEMANTIC,
            agent="analysis",
            content="AAPL has beaten earnings 8 times",
            user_id="user_123",
            embedding=[0.1, 0.2, 0.3],
            metadata={"symbol": "AAPL"},
            importance=0.8,
        )
        assert memory.user_id == "user_123"
        assert memory.embedding == [0.1, 0.2, 0.3]
        assert memory.metadata == {"symbol": "AAPL"}
        assert memory.importance == 0.8

    def test_timestamps_auto_generated(self) -> None:
        """Test that timestamps are auto-generated."""
        before = datetime.now()
        memory = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Test",
        )
        after = datetime.now()

        assert before <= memory.created_at <= after
        assert before <= memory.accessed_at <= after

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        memory = Memory(
            memory_type=MemoryType.PREFERENCE,
            agent="recommendation",
            content="User prefers dividend stocks",
            user_id="user_456",
            metadata={"preference_type": "investment_style"},
            importance=0.7,
        )
        data = memory.to_dict()

        assert data["memory_type"] == "preference"
        assert data["agent"] == "recommendation"
        assert data["content"] == "User prefers dividend stocks"
        assert data["user_id"] == "user_456"
        assert data["importance"] == 0.7
        assert "created_at" in data
        assert "accessed_at" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        now = datetime.now()
        data = {
            "id": "test-id",
            "memory_type": "semantic",
            "agent": "analysis",
            "user_id": None,
            "content": "NVDA is a semiconductor company",
            "embedding": [0.5, 0.6],
            "metadata": {"symbol": "NVDA"},
            "importance": 0.6,
            "created_at": now.isoformat(),
            "accessed_at": now.isoformat(),
        }
        memory = Memory.from_dict(data)

        assert memory.id == "test-id"
        assert memory.memory_type == MemoryType.SEMANTIC
        assert memory.agent == "analysis"
        assert memory.content == "NVDA is a semiconductor company"
        assert memory.embedding == [0.5, 0.6]

    def test_round_trip_serialization(self) -> None:
        """Test that to_dict and from_dict are inverses."""
        original = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Analyzed AAPL",
            user_id="user_789",
            embedding=[0.1] * 10,
            metadata={"symbol": "AAPL", "action": "HOLD"},
            importance=0.9,
        )
        data = original.to_dict()
        restored = Memory.from_dict(data)

        assert restored.id == original.id
        assert restored.memory_type == original.memory_type
        assert restored.agent == original.agent
        assert restored.content == original.content
        assert restored.user_id == original.user_id
        assert restored.embedding == original.embedding
        assert restored.metadata == original.metadata
        assert restored.importance == original.importance


class TestMemorySearchResult:
    """Tests for MemorySearchResult dataclass."""

    def test_required_fields(self) -> None:
        """Test creating search result with required fields."""
        memory = Memory(
            memory_type=MemoryType.EPISODIC,
            agent="research",
            content="Test",
        )
        result = MemorySearchResult(memory=memory, similarity=0.85)

        assert result.memory == memory
        assert result.similarity == 0.85

    def test_similarity_range(self) -> None:
        """Test similarity can be any float (validation at usage)."""
        memory = Memory(
            memory_type=MemoryType.SEMANTIC,
            agent="analysis",
            content="Test",
        )
        result = MemorySearchResult(memory=memory, similarity=0.0)
        assert result.similarity == 0.0

        result = MemorySearchResult(memory=memory, similarity=1.0)
        assert result.similarity == 1.0
