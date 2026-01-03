"""Observability and tracing with Langfuse.

This module contains:
- Trace configuration
- Custom metrics and metadata
- Cost attribution
- Dashboard utilities
- Nested span hierarchy support
- Tagging system for categorization
"""

from src.observability.metrics import (
    AgentMetadata,
    MetadataBuilder,
    RequestMetadata,
    Tag,
    TagManager,
    ToolMetadata,
    collect_tags,
    create_agent_metadata,
    create_request_metadata,
    create_tool_metadata,
)
from src.observability.tracing import (
    SpanContext,
    SpanLevel,
    SpanMetadata,
    TraceContext,
    agent_span,
    flush_traces,
    generation_span,
    get_langfuse_client,
    planning_span,
    postprocessing_span,
    preprocessing_span,
    request_span,
    shutdown_tracing,
    tool_span,
    traced_agent,
    traced_generation,
    traced_tool,
)

__all__ = [
    # Metadata classes
    "AgentMetadata",
    "MetadataBuilder",
    "RequestMetadata",
    "Tag",
    "TagManager",
    "ToolMetadata",
    # Metadata factory functions
    "collect_tags",
    "create_agent_metadata",
    "create_request_metadata",
    "create_tool_metadata",
    # Span hierarchy classes
    "SpanContext",
    "SpanLevel",
    "SpanMetadata",
    # Convenience span functions
    "agent_span",
    "generation_span",
    "planning_span",
    "postprocessing_span",
    "preprocessing_span",
    "request_span",
    "tool_span",
    # Trace context and decorators
    "TraceContext",
    "traced_agent",
    "traced_generation",
    "traced_tool",
    # Client utilities
    "flush_traces",
    "get_langfuse_client",
    "shutdown_tracing",
]
