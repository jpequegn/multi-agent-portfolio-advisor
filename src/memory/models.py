"""Memory data models for agent long-term memory.

This module defines the core data structures for storing and retrieving
agent memories, including episodic memories (past analyses), semantic
memories (learned facts), and user preferences.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class MemoryType(Enum):
    """Types of memories stored by agents."""

    EPISODIC = "episodic"  # Past analysis sessions and outcomes
    SEMANTIC = "semantic"  # Learned facts about symbols/sectors
    PREFERENCE = "preference"  # User investment preferences


@dataclass
class Memory:
    """A single memory stored by an agent.

    Attributes:
        id: Unique identifier for the memory.
        memory_type: Category of memory (episodic, semantic, preference).
        agent: Name of the agent that created this memory.
        user_id: Optional user ID for user-specific memories.
        content: The text content of the memory.
        embedding: Vector embedding for similarity search (384 dims).
        metadata: Additional structured data (symbol, sector, etc.).
        importance: Score from 0-1 affecting retrieval ranking.
        created_at: When the memory was created.
        accessed_at: Last time the memory was retrieved (for LRU).
    """

    memory_type: MemoryType
    agent: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "agent": self.agent,
            "user_id": self.user_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Create memory from dictionary."""
        return cls(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            agent=data["agent"],
            user_id=data.get("user_id"),
            content=data["content"],
            embedding=data.get("embedding", []),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
        )


@dataclass
class MemorySearchResult:
    """Result from a memory similarity search.

    Attributes:
        memory: The retrieved memory.
        similarity: Cosine similarity score (0-1).
    """

    memory: Memory
    similarity: float
