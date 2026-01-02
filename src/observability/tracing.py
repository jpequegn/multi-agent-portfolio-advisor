"""Langfuse tracing integration for observability.

This module provides tracing capabilities using Langfuse for comprehensive
observability of agent workflows, LLM calls, and tool executions.

Span Hierarchy:
- Session: Top-level grouping for user sessions
- Request: Individual API requests within a session
- Agent: Agent execution spans
- Tool: Tool/function call spans
- Generation: LLM API calls
"""

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse import observe as langfuse_observe

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def get_langfuse_client() -> Langfuse:
    """Get or create the Langfuse client.

    Uses environment variables for configuration:
    - LANGFUSE_PUBLIC_KEY: Public API key
    - LANGFUSE_SECRET_KEY: Secret API key
    - LANGFUSE_HOST: Host URL (default: http://localhost:3000)

    Returns:
        Configured Langfuse client instance.
    """
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if public_key and secret_key:
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    # Fall back to get_client which uses env vars automatically
    return get_client()


def traced_agent(
    name: str,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing agent execution.

    Creates a span for the agent with proper metadata and nesting.

    Args:
        name: Name of the agent for the trace.
        capture_input: Whether to capture function inputs.
        capture_output: Whether to capture function outputs.

    Returns:
        Decorated function with tracing.

    Example:
        @traced_agent("research_agent")
        async def research(state: AgentState) -> AgentState:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        @langfuse_observe(
            name=name,
            as_type="span",
            capture_input=capture_input,
            capture_output=capture_output,
        )
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug("agent_trace_start", agent=name)
            result: R = await func(*args, **kwargs)  # type: ignore[misc]
            logger.debug("agent_trace_end", agent=name)
            return result

        @wraps(func)
        @langfuse_observe(
            name=name,
            as_type="span",
            capture_input=capture_input,
            capture_output=capture_output,
        )
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug("agent_trace_start", agent=name)
            result: R = func(*args, **kwargs)
            logger.debug("agent_trace_end", agent=name)
            return result

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator


def traced_tool(
    name: str,
    *,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing tool execution.

    Creates a tool-type span for function calls.

    Args:
        name: Name of the tool for the trace.
        capture_input: Whether to capture function inputs.
        capture_output: Whether to capture function outputs.

    Returns:
        Decorated function with tracing.

    Example:
        @traced_tool("market_data")
        async def get_market_data(symbol: str) -> dict:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        @langfuse_observe(
            name=name,
            as_type="tool",
            capture_input=capture_input,
            capture_output=capture_output,
        )
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug("tool_trace_start", tool=name)
            result: R = await func(*args, **kwargs)  # type: ignore[misc]
            logger.debug("tool_trace_end", tool=name)
            return result

        @wraps(func)
        @langfuse_observe(
            name=name,
            as_type="tool",
            capture_input=capture_input,
            capture_output=capture_output,
        )
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug("tool_trace_start", tool=name)
            result: R = func(*args, **kwargs)
            logger.debug("tool_trace_end", tool=name)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator


def traced_generation(
    name: str,
    *,
    model: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for tracing LLM generation calls.

    Creates a generation-type span for LLM API calls.

    Args:
        name: Name of the generation for the trace.
        model: Model name to record in the trace.
        capture_input: Whether to capture function inputs.
        capture_output: Whether to capture function outputs.

    Returns:
        Decorated function with tracing.

    Example:
        @traced_generation("analyze_portfolio", model="claude-sonnet-4-20250514")
        async def analyze(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        @langfuse_observe(
            name=name,
            as_type="generation",
            capture_input=capture_input,
            capture_output=capture_output,
        )
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug("generation_trace_start", name=name, model=model)
            result: R = await func(*args, **kwargs)  # type: ignore[misc]
            logger.debug("generation_trace_end", name=name)
            return result

        @wraps(func)
        @langfuse_observe(
            name=name,
            as_type="generation",
            capture_input=capture_input,
            capture_output=capture_output,
        )
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug("generation_trace_start", name=name, model=model)
            result: R = func(*args, **kwargs)
            logger.debug("generation_trace_end", name=name)
            return result

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator


class TraceContext:
    """Context manager for creating trace sessions.

    Provides a way to group related operations under a common session
    with propagated attributes.

    Example:
        async with TraceContext(
            session_id="user-123",
            user_id="user-123",
            metadata={"request_type": "portfolio_analysis"}
        ):
            result = await agent(state)
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize trace context.

        Args:
            session_id: Optional session identifier for grouping traces.
            user_id: Optional user identifier.
            metadata: Optional metadata dict to attach to traces.
            tags: Optional list of tags.
        """
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.tags = tags or []
        self._context_manager: Any = None

    async def __aenter__(self) -> "TraceContext":
        """Enter async context and start trace session."""
        self._context_manager = propagate_attributes(
            session_id=self.session_id,
            user_id=self.user_id,
            metadata=self.metadata,
            tags=self.tags,
        )
        self._context_manager.__enter__()
        logger.debug(
            "trace_context_started",
            session_id=self.session_id,
            user_id=self.user_id,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and end trace session."""
        if self._context_manager:
            self._context_manager.__exit__(exc_type, exc_val, exc_tb)
        logger.debug("trace_context_ended", session_id=self.session_id)

    def __enter__(self) -> "TraceContext":
        """Enter sync context and start trace session."""
        self._context_manager = propagate_attributes(
            session_id=self.session_id,
            user_id=self.user_id,
            metadata=self.metadata,
            tags=self.tags,
        )
        self._context_manager.__enter__()
        logger.debug(
            "trace_context_started",
            session_id=self.session_id,
            user_id=self.user_id,
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit sync context and end trace session."""
        if self._context_manager:
            self._context_manager.__exit__(exc_type, exc_val, exc_tb)
        logger.debug("trace_context_ended", session_id=self.session_id)


def flush_traces() -> None:
    """Flush any pending traces to Langfuse.

    Call this in short-lived processes (scripts, serverless) to ensure
    all traces are sent before the process exits.
    """
    try:
        client = get_client()
        client.flush()
        logger.debug("traces_flushed")
    except Exception as e:
        logger.warning("flush_traces_failed", error=str(e))


def shutdown_tracing() -> None:
    """Gracefully shutdown tracing with final flush.

    Call this before process exit to ensure all data is sent.
    """
    try:
        client = get_client()
        client.shutdown()
        logger.debug("tracing_shutdown")
    except Exception as e:
        logger.warning("shutdown_tracing_failed", error=str(e))
