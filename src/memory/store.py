"""SQLite-based memory store for agent memories.

This module provides persistent storage for agent memories using SQLite,
with support for vector similarity search using NumPy.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np
import structlog

from src.memory.models import Memory, MemorySearchResult, MemoryType

logger = structlog.get_logger(__name__)

# Default database path
DEFAULT_DB_PATH = "./data/memories.db"


class MemoryStore:
    """SQLite-based storage for agent memories.

    Provides CRUD operations and similarity search for memories.
    Embeddings are stored as JSON arrays and similarity is computed
    in Python using NumPy.

    Example:
        store = MemoryStore()
        await store.initialize()
        await store.save(memory)
        results = await store.search_similar(query_embedding, limit=5)
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the memory store.

        Args:
            db_path: Path to SQLite database file.
                    Defaults to MEMORY_DB_PATH env var or ./data/memories.db
        """
        self._db_path = db_path or os.environ.get("MEMORY_DB_PATH", DEFAULT_DB_PATH)
        self._connection: aiosqlite.Connection | None = None
        self._logger = logger.bind(component="memory_store")

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        # Ensure directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row

        await self._create_tables()
        self._logger.info("memory_store_initialized", db_path=self._db_path)

    async def _create_tables(self) -> None:
        """Create the memories table if it doesn't exist."""
        assert self._connection is not None

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                agent TEXT NOT NULL,
                user_id TEXT,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL
            )
        """)

        # Create indexes for common queries
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)
        """)

        await self._connection.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def save(self, memory: Memory) -> Memory:
        """Save a memory to the store.

        Args:
            memory: Memory to save.

        Returns:
            The saved memory.
        """
        assert self._connection is not None

        await self._connection.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, memory_type, agent, user_id, content, embedding, metadata,
             importance, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id,
                memory.memory_type.value,
                memory.agent,
                memory.user_id,
                memory.content,
                json.dumps(memory.embedding),
                json.dumps(memory.metadata),
                memory.importance,
                memory.created_at.isoformat(),
                memory.accessed_at.isoformat(),
            ),
        )
        await self._connection.commit()

        self._logger.debug(
            "memory_saved",
            memory_id=memory.id,
            memory_type=memory.memory_type.value,
            agent=memory.agent,
        )
        return memory

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            The memory if found, None otherwise.
        """
        assert self._connection is not None

        cursor = await self._connection.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory ID.

        Returns:
            True if deleted, False if not found.
        """
        assert self._connection is not None

        cursor = await self._connection.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,),
        )
        await self._connection.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            self._logger.debug("memory_deleted", memory_id=memory_id)
        return deleted

    async def list_memories(
        self,
        memory_type: MemoryType | None = None,
        agent: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """List memories with optional filters.

        Args:
            memory_type: Filter by memory type.
            agent: Filter by agent name.
            user_id: Filter by user ID.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching memories.
        """
        assert self._connection is not None

        conditions: list[str] = []
        params: list[Any] = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY importance DESC, accessed_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def search_similar(
        self,
        query_embedding: list[float],
        memory_type: MemoryType | None = None,
        agent: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> list[MemorySearchResult]:
        """Search for similar memories using vector similarity.

        Args:
            query_embedding: The query vector.
            memory_type: Filter by memory type.
            agent: Filter by agent name.
            user_id: Filter by user ID.
            limit: Maximum number of results.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of search results ordered by similarity.
        """
        assert self._connection is not None

        # Get all candidate memories
        conditions: list[str] = []
        params: list[Any] = []

        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)
        if agent:
            conditions.append("agent = ?")
            params.append(agent)
        if user_id:
            conditions.append("(user_id = ? OR user_id IS NULL)")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM memories WHERE {where_clause}"

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        if not rows:
            return []

        # Compute similarities
        query_vec = np.array(query_embedding)
        query_norm = query_vec / np.linalg.norm(query_vec)

        results: list[MemorySearchResult] = []
        for row in rows:
            memory = self._row_to_memory(row)
            if not memory.embedding:
                continue

            candidate_vec = np.array(memory.embedding)
            candidate_norm = candidate_vec / np.linalg.norm(candidate_vec)
            similarity = float(np.dot(query_norm, candidate_norm))

            # Apply importance weighting (slight boost for important memories)
            weighted_similarity = similarity * (0.9 + 0.1 * memory.importance)

            if weighted_similarity >= min_similarity:
                results.append(MemorySearchResult(memory=memory, similarity=weighted_similarity))

        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    async def update_accessed_at(self, memory_id: str) -> None:
        """Update the accessed_at timestamp for a memory.

        Args:
            memory_id: The memory ID.
        """
        assert self._connection is not None

        await self._connection.execute(
            "UPDATE memories SET accessed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), memory_id),
        )
        await self._connection.commit()

    async def decay_importance(
        self,
        days_threshold: int = 30,
        decay_amount: float = 0.1,
    ) -> int:
        """Decay importance of old, unaccessed memories.

        Args:
            days_threshold: Days since last access to trigger decay.
            decay_amount: Amount to reduce importance by.

        Returns:
            Number of memories affected.
        """
        assert self._connection is not None

        cutoff = datetime.now().isoformat()
        # Calculate cutoff date (days_threshold days ago)
        from datetime import timedelta

        cutoff_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()

        cursor = await self._connection.execute(
            """
            UPDATE memories
            SET importance = MAX(0, importance - ?)
            WHERE accessed_at < ?
            """,
            (decay_amount, cutoff_date),
        )
        await self._connection.commit()

        affected = cursor.rowcount
        if affected > 0:
            self._logger.info(
                "memories_decayed",
                count=affected,
                decay_amount=decay_amount,
                days_threshold=days_threshold,
            )
        return affected

    async def delete_low_importance(self, threshold: float = 0.1) -> int:
        """Delete memories with very low importance.

        Args:
            threshold: Importance threshold below which to delete.

        Returns:
            Number of memories deleted.
        """
        assert self._connection is not None

        cursor = await self._connection.execute(
            "DELETE FROM memories WHERE importance < ?",
            (threshold,),
        )
        await self._connection.commit()

        deleted = cursor.rowcount
        if deleted > 0:
            self._logger.info("low_importance_memories_deleted", count=deleted)
        return deleted

    async def count(
        self,
        user_id: str | None = None,
    ) -> int:
        """Count memories, optionally filtered by user.

        Args:
            user_id: Optional user ID filter.

        Returns:
            Number of memories.
        """
        assert self._connection is not None

        if user_id:
            cursor = await self._connection.execute(
                "SELECT COUNT(*) FROM memories WHERE user_id = ?",
                (user_id,),
            )
        else:
            cursor = await self._connection.execute("SELECT COUNT(*) FROM memories")

        row = await cursor.fetchone()
        return row[0] if row else 0

    async def evict_lru(self, user_id: str, max_memories: int = 1000) -> int:
        """Evict least recently used memories if over limit.

        Args:
            user_id: User ID to check.
            max_memories: Maximum memories allowed per user.

        Returns:
            Number of memories evicted.
        """
        assert self._connection is not None

        count = await self.count(user_id)
        if count <= max_memories:
            return 0

        to_delete = count - max_memories

        # Find IDs of least recently accessed memories
        cursor = await self._connection.execute(
            """
            SELECT id FROM memories
            WHERE user_id = ?
            ORDER BY accessed_at ASC
            LIMIT ?
            """,
            (user_id, to_delete),
        )
        rows = await cursor.fetchall()
        ids_to_delete = [row[0] for row in rows]

        if ids_to_delete:
            placeholders = ",".join("?" * len(ids_to_delete))
            await self._connection.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                ids_to_delete,
            )
            await self._connection.commit()

        self._logger.info("lru_eviction", user_id=user_id, evicted=len(ids_to_delete))
        return len(ids_to_delete)

    def _row_to_memory(self, row: aiosqlite.Row) -> Memory:
        """Convert a database row to a Memory object."""
        return Memory(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            agent=row["agent"],
            user_id=row["user_id"],
            content=row["content"],
            embedding=json.loads(row["embedding"]),
            metadata=json.loads(row["metadata"]),
            importance=row["importance"],
            created_at=datetime.fromisoformat(row["created_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
        )
