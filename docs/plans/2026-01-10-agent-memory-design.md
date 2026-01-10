# Agent Memory and Learning Design

**Issue:** #77
**Date:** 2026-01-10
**Status:** Approved

## Overview

Implement long-term memory for agents to learn from past analyses, improving recommendations over time.

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | SQLite | File-based, no server needed, easy migration to PostgreSQL later |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API keys, 384-dim vectors |
| Vector search | NumPy cosine similarity | Simple, sufficient for <10k memories |

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Agent Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ Research │  │ Analysis │  │    Recommendation    │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘  │
│       └─────────────┼───────────────────┘              │
│                     ▼                                   │
│            ┌─────────────────┐                         │
│            │  MemoryManager  │  ← retrieves relevant   │
│            └────────┬────────┘    memories for prompts │
│                     │                                   │
├─────────────────────┼──────────────────────────────────┤
│            ┌────────▼────────┐                         │
│            │  MemoryStore    │  ← SQLite + embeddings  │
│            │                 │                         │
│            │  ┌───────────┐  │                         │
│            │  │ Episodic  │  │  Past analysis sessions │
│            │  ├───────────┤  │                         │
│            │  │ Semantic  │  │  Facts about symbols    │
│            │  ├───────────┤  │                         │
│            │  │ Preference│  │  User investment style  │
│            │  └───────────┘  │                         │
│            └─────────────────┘                         │
└────────────────────────────────────────────────────────┘
```

## Data Model

```python
class MemoryType(Enum):
    EPISODIC = "episodic"      # Past analysis sessions
    SEMANTIC = "semantic"       # Facts about symbols/sectors
    PREFERENCE = "preference"   # User investment preferences

@dataclass
class Memory:
    id: str                          # UUID
    memory_type: MemoryType
    agent: str                       # research, analysis, recommendation
    user_id: str | None              # For user-specific memories
    content: str                     # The memory text
    embedding: list[float]           # 384-dim vector
    metadata: dict                   # symbol, sector, date, etc.
    importance: float                # 0-1, affects retrieval ranking
    created_at: datetime
    accessed_at: datetime            # Updated on retrieval (LRU)
```

## API

```python
class MemoryManager:
    async def store(content, memory_type, agent, ...) -> Memory
    async def retrieve(query, memory_type, ...) -> list[Memory]
    async def get_context_for_agent(agent, symbols, user_id) -> str
```

## Memory Lifecycle

| Condition | Action |
|-----------|--------|
| Accessed in last 7 days | Keep, boost importance |
| Not accessed 30+ days | Decay importance by 0.1 |
| Importance < 0.1 | Delete |
| Max 1000 memories/user | LRU eviction |

## File Structure

```
src/memory/
├── __init__.py
├── models.py          # Memory, MemoryType dataclasses
├── store.py           # SQLite persistence + vector search
├── embeddings.py      # sentence-transformers wrapper
└── manager.py         # MemoryManager high-level API
```

## Dependencies

- sentence-transformers ^3.0.0
- numpy (existing)
