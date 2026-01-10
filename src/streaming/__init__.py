"""Streaming module for Server-Sent Events (SSE).

This module provides real-time event streaming for portfolio analysis workflows,
allowing clients to receive progress updates as agents execute.
"""

from src.streaming.emitter import (
    WorkflowEventEmitter,
    get_emitter_from_state,
)
from src.streaming.events import (
    StreamEvent,
    StreamEventType,
)

__all__ = [
    "StreamEvent",
    "StreamEventType",
    "WorkflowEventEmitter",
    "get_emitter_from_state",
]
