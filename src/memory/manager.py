"""Memory manager for agent long-term memory.

This module provides the high-level API for storing and retrieving
agent memories, including context generation for agent prompts.
"""

from datetime import datetime
from typing import Any

import structlog

from src.memory.embeddings import EmbeddingService
from src.memory.models import Memory, MemorySearchResult, MemoryType
from src.memory.store import MemoryStore

logger = structlog.get_logger(__name__)


class MemoryManager:
    """High-level API for agent memory operations.

    Coordinates between the embedding service and memory store to
    provide a simple interface for storing and retrieving memories.

    Example:
        manager = MemoryManager()
        await manager.initialize()

        # Store a memory
        await manager.store(
            content="AAPL P/E ratio was 28.5, recommended HOLD",
            memory_type=MemoryType.EPISODIC,
            agent="analysis",
            metadata={"symbol": "AAPL", "action": "HOLD"},
        )

        # Retrieve relevant memories
        memories = await manager.retrieve(
            query="What do we know about AAPL?",
            agent="analysis",
        )

        # Get context for agent prompt
        context = await manager.get_context_for_agent(
            agent="analysis",
            symbols=["AAPL"],
        )
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize the memory manager.

        Args:
            store: Memory store instance. Created if not provided.
            embedding_service: Embedding service instance. Created if not provided.
        """
        self._store = store or MemoryStore()
        self._embedding_service = embedding_service or EmbeddingService()
        self._logger = logger.bind(component="memory_manager")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the memory manager and underlying store."""
        if not self._initialized:
            await self._store.initialize()
            self._initialized = True
            self._logger.info("memory_manager_initialized")

    async def close(self) -> None:
        """Close the memory manager and release resources."""
        await self._store.close()
        self._initialized = False

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        agent: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> Memory:
        """Store a new memory with auto-generated embedding.

        Args:
            content: The text content of the memory.
            memory_type: Category of memory.
            agent: Name of the agent creating this memory.
            user_id: Optional user ID for user-specific memories.
            metadata: Additional structured data.
            importance: Score from 0-1 affecting retrieval ranking.

        Returns:
            The stored memory.
        """
        # Generate embedding
        embedding = await self._embedding_service.embed(content)

        # Create memory
        memory = Memory(
            memory_type=memory_type,
            agent=agent,
            content=content,
            user_id=user_id,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance,
        )

        # Save to store
        await self._store.save(memory)

        # Check if we need to evict old memories
        if user_id:
            await self._store.evict_lru(user_id)

        self._logger.debug(
            "memory_stored",
            memory_id=memory.id,
            memory_type=memory_type.value,
            agent=agent,
            content_length=len(content),
        )

        return memory

    async def retrieve(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        agent: str | None = None,
        user_id: str | None = None,
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> list[MemorySearchResult]:
        """Retrieve relevant memories using semantic similarity.

        Args:
            query: The query text to find similar memories for.
            memory_type: Optional filter by memory type.
            agent: Optional filter by agent name.
            user_id: Optional filter by user ID.
            limit: Maximum number of results.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of search results ordered by similarity.
        """
        # Generate query embedding
        query_embedding = await self._embedding_service.embed(query)

        # Search for similar memories
        results = await self._store.search_similar(
            query_embedding=query_embedding,
            memory_type=memory_type,
            agent=agent,
            user_id=user_id,
            limit=limit,
            min_similarity=min_similarity,
        )

        # Update accessed_at for retrieved memories
        for result in results:
            await self._store.update_accessed_at(result.memory.id)

        self._logger.debug(
            "memories_retrieved",
            query_length=len(query),
            results_count=len(results),
            top_similarity=results[0].similarity if results else 0,
        )

        return results

    async def get_context_for_agent(
        self,
        agent: str,
        symbols: list[str] | None = None,
        user_id: str | None = None,
        max_memories: int = 5,
    ) -> str:
        """Build context string for agent prompt injection.

        Retrieves relevant memories and formats them as a context
        block that can be injected into agent system prompts.

        Args:
            agent: Name of the agent requesting context.
            symbols: Optional list of symbols to include in query.
            user_id: Optional user ID for user-specific memories.
            max_memories: Maximum memories to include.

        Returns:
            Formatted context string, or empty string if no relevant memories.
        """
        # Build query from symbols
        if symbols:
            query = f"Analysis context for: {', '.join(symbols)}"
        else:
            query = f"Recent {agent} context and insights"

        # Retrieve episodic memories (past analyses)
        episodic = await self.retrieve(
            query=query,
            memory_type=MemoryType.EPISODIC,
            agent=agent,
            user_id=user_id,
            limit=max_memories,
            min_similarity=0.4,
        )

        # Retrieve semantic memories (facts)
        semantic = await self.retrieve(
            query=query,
            memory_type=MemoryType.SEMANTIC,
            user_id=user_id,
            limit=max_memories,
            min_similarity=0.4,
        )

        # Retrieve user preferences
        preferences: list[MemorySearchResult] = []
        if user_id:
            preferences = await self.retrieve(
                query="user investment preferences and style",
                memory_type=MemoryType.PREFERENCE,
                user_id=user_id,
                limit=3,
                min_similarity=0.3,
            )

        # Build context string
        sections: list[str] = []

        if episodic:
            section = "### Past Analyses\n"
            for result in episodic:
                section += f"- {result.memory.content}\n"
            sections.append(section)

        if semantic:
            section = "### Relevant Facts\n"
            for result in semantic:
                section += f"- {result.memory.content}\n"
            sections.append(section)

        if preferences:
            section = "### User Preferences\n"
            for result in preferences:
                section += f"- {result.memory.content}\n"
            sections.append(section)

        if not sections:
            return ""

        context = "## Memory Context\n\n" + "\n".join(sections)

        self._logger.debug(
            "context_generated",
            agent=agent,
            symbols=symbols,
            episodic_count=len(episodic),
            semantic_count=len(semantic),
            preference_count=len(preferences),
        )

        return context

    async def store_analysis_result(
        self,
        agent: str,
        symbol: str,
        summary: str,
        action: str | None = None,
        metrics: dict[str, Any] | None = None,
        user_id: str | None = None,
        importance: float = 0.6,
    ) -> Memory:
        """Store an analysis result as an episodic memory.

        Convenience method for storing analysis outcomes.

        Args:
            agent: Name of the agent that performed the analysis.
            symbol: Stock symbol analyzed.
            summary: Summary of the analysis.
            action: Recommended action (BUY, SELL, HOLD).
            metrics: Key metrics from the analysis.
            user_id: Optional user ID.
            importance: Importance score (0-1).

        Returns:
            The stored memory.
        """
        # Build content
        content = f"Analyzed {symbol}: {summary}"
        if action:
            content += f" Recommended: {action}."

        # Build metadata
        metadata: dict[str, Any] = {
            "symbol": symbol,
            "analyzed_at": datetime.now().isoformat(),
        }
        if action:
            metadata["action"] = action
        if metrics:
            metadata["metrics"] = metrics

        return await self.store(
            content=content,
            memory_type=MemoryType.EPISODIC,
            agent=agent,
            user_id=user_id,
            metadata=metadata,
            importance=importance,
        )

    async def store_fact(
        self,
        content: str,
        symbol: str | None = None,
        sector: str | None = None,
        fact_type: str | None = None,
        importance: float = 0.5,
    ) -> Memory:
        """Store a semantic fact.

        Convenience method for storing learned facts about symbols/sectors.

        Args:
            content: The fact content.
            symbol: Related stock symbol.
            sector: Related sector.
            fact_type: Type of fact (earnings, dividend, etc.).
            importance: Importance score (0-1).

        Returns:
            The stored memory.
        """
        metadata: dict[str, Any] = {}
        if symbol:
            metadata["symbol"] = symbol
        if sector:
            metadata["sector"] = sector
        if fact_type:
            metadata["fact_type"] = fact_type

        return await self.store(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            agent="system",
            metadata=metadata,
            importance=importance,
        )

    async def store_preference(
        self,
        user_id: str,
        content: str,
        preference_type: str | None = None,
        importance: float = 0.7,
    ) -> Memory:
        """Store a user preference.

        Convenience method for storing user investment preferences.

        Args:
            user_id: The user ID.
            content: The preference content.
            preference_type: Type of preference (investment_style, risk, etc.).
            importance: Importance score (0-1).

        Returns:
            The stored memory.
        """
        metadata: dict[str, Any] = {}
        if preference_type:
            metadata["preference_type"] = preference_type

        return await self.store(
            content=content,
            memory_type=MemoryType.PREFERENCE,
            agent="system",
            user_id=user_id,
            metadata=metadata,
            importance=importance,
        )

    async def run_maintenance(self) -> dict[str, int]:
        """Run maintenance tasks: decay importance and evict old memories.

        Returns:
            Dictionary with counts of affected memories.
        """
        decayed = await self._store.decay_importance()
        deleted = await self._store.delete_low_importance()

        self._logger.info(
            "maintenance_complete",
            decayed=decayed,
            deleted=deleted,
        )

        return {"decayed": decayed, "deleted": deleted}


# Global memory manager instance
_memory_manager: MemoryManager | None = None


async def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        await _memory_manager.initialize()
    return _memory_manager


async def reset_memory_manager() -> None:
    """Reset the global memory manager (for testing)."""
    global _memory_manager
    if _memory_manager is not None:
        await _memory_manager.close()
        _memory_manager = None
