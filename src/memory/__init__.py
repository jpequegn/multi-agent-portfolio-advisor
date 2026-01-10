"""Agent memory and learning system.

This module provides long-term memory for agents to learn from past
analyses and improve recommendations over time.

Features:
- Episodic memory: Past analysis sessions and outcomes
- Semantic memory: Learned facts about symbols/sectors
- User preferences: Individual user investment preferences
- Similarity search: Retrieve relevant memories using embeddings
- Memory lifecycle: Decay and eviction for old memories
"""

from src.memory.embeddings import EmbeddingService
from src.memory.manager import (
    MemoryManager,
    get_memory_manager,
    reset_memory_manager,
)
from src.memory.models import Memory, MemorySearchResult, MemoryType
from src.memory.store import MemoryStore

__all__ = [
    "EmbeddingService",
    "Memory",
    "MemoryManager",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryType",
    "get_memory_manager",
    "reset_memory_manager",
]
