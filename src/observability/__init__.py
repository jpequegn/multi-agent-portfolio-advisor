"""Observability and tracing with Langfuse.

This module contains:
- Trace configuration
- Custom metrics
- Cost attribution
- Dashboard utilities
- Nested span hierarchy support
"""

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
