"""Langfuse tracing integration for observability.

This module provides tracing capabilities using Langfuse for comprehensive
observability of agent workflows, LLM calls, and tool executions.

Span Hierarchy:
    Session (user session)
    └── Request (single portfolio analysis request)
        ├── Preprocessing Span
        ├── Research Agent Span
        │   ├── Planning Span (LLM reasoning)
        │   ├── get_market_data Tool Span
        │   │   ├── input: {"symbols": ["AAPL", "GOOGL"]}
        │   │   └── output: {"data": [...]}
        │   ├── search_news Tool Span
        │   └── Response Generation Span
        ├── Analysis Agent Span
        │   └── ...
        ├── Recommendation Agent Span
        │   └── ...
        └── Postprocessing Span

Features:
- Nested span hierarchy with context propagation
- Input/output capture for tool spans
- Reasoning trace support for LLM spans
- Preprocessing/postprocessing spans
- Span level metadata (agent, tool, generation)
"""

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse import observe as langfuse_observe

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


# ============================================================================
# Span Level Enumeration
# ============================================================================


class SpanLevel(Enum):
    """Levels in the span hierarchy."""

    SESSION = "session"
    REQUEST = "request"
    PREPROCESSING = "preprocessing"
    AGENT = "agent"
    PLANNING = "planning"
    TOOL = "tool"
    GENERATION = "generation"
    POSTPROCESSING = "postprocessing"


# ============================================================================
# Span Metadata
# ============================================================================


@dataclass
class SpanMetadata:
    """Metadata for a span in the hierarchy.

    Attributes:
        level: Level in the span hierarchy.
        parent_span_id: ID of the parent span.
        agent_name: Name of the agent (for agent-level spans).
        tool_name: Name of the tool (for tool-level spans).
        model: Model name (for generation spans).
        input_data: Captured input data.
        output_data: Captured output data.
        reasoning: Captured reasoning/planning traces.
        started_at: When the span started.
        ended_at: When the span ended.
        error: Error if the span failed.
    """

    level: SpanLevel
    parent_span_id: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    model: str | None = None
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    reasoning: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Langfuse metadata."""
        return {
            "level": self.level.value,
            "parent_span_id": self.parent_span_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "model": self.model,
            "has_input": self.input_data is not None,
            "has_output": self.output_data is not None,
            "has_reasoning": len(self.reasoning) > 0,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "has_error": self.error is not None,
        }


# ============================================================================
# Span Context Manager
# ============================================================================


class SpanContext:
    """Context manager for creating nested spans.

    Provides explicit control over span hierarchy with support for
    adding reasoning traces and capturing input/output.

    Example:
        async with SpanContext(
            name="research_agent",
            level=SpanLevel.AGENT,
            metadata={"symbols": ["AAPL"]},
        ) as span:
            span.add_reasoning("Analyzing market data for AAPL")
            span.set_input({"symbols": ["AAPL"]})
            result = await do_research()
            span.set_output({"data": result})
    """

    def __init__(
        self,
        name: str,
        level: SpanLevel,
        *,
        parent_span_id: str | None = None,
        agent_name: str | None = None,
        tool_name: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        capture_input: bool = True,
        capture_output: bool = True,
    ) -> None:
        """Initialize span context.

        Args:
            name: Name of the span.
            level: Level in the span hierarchy.
            parent_span_id: Optional parent span ID for nesting.
            agent_name: Agent name (for agent-level spans).
            tool_name: Tool name (for tool-level spans).
            model: Model name (for generation spans).
            metadata: Additional metadata to attach.
            tags: Tags for the span.
            capture_input: Whether to capture input data.
            capture_output: Whether to capture output data.
        """
        self.name = name
        self.level = level
        self.parent_span_id = parent_span_id
        self.metadata = metadata or {}
        self.tags = tags or []
        self.capture_input = capture_input
        self.capture_output = capture_output

        self._span_metadata = SpanMetadata(
            level=level,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            tool_name=tool_name,
            model=model,
        )
        self._langfuse_span: Any = None
        self._span_id: str | None = None

    @property
    def span_id(self) -> str | None:
        """Get the span ID."""
        return self._span_id

    def add_reasoning(self, reasoning: str) -> None:
        """Add a reasoning trace to the span.

        Args:
            reasoning: Reasoning text to capture.
        """
        self._span_metadata.reasoning.append(reasoning)
        logger.debug("span_reasoning_added", span=self.name, reasoning=reasoning[:100])

    def set_input(self, input_data: dict[str, Any]) -> None:
        """Set input data for the span.

        Args:
            input_data: Input data to capture.
        """
        if self.capture_input:
            self._span_metadata.input_data = input_data
            if self._langfuse_span:
                self._langfuse_span.update(input=input_data)

    def set_output(self, output_data: dict[str, Any]) -> None:
        """Set output data for the span.

        Args:
            output_data: Output data to capture.
        """
        if self.capture_output:
            self._span_metadata.output_data = output_data
            if self._langfuse_span:
                self._langfuse_span.update(output=output_data)

    def set_error(self, error: str | Exception) -> None:
        """Mark the span as failed with an error.

        Args:
            error: Error message or exception.
        """
        self._span_metadata.error = str(error)
        if self._langfuse_span:
            self._langfuse_span.update(
                level="ERROR",
                status_message=str(error),
            )

    async def __aenter__(self) -> "SpanContext":
        """Enter async context and start the span."""
        return self._start_span()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and end the span."""
        self._end_span(exc_val)

    def __enter__(self) -> "SpanContext":
        """Enter sync context and start the span."""
        return self._start_span()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit sync context and end the span."""
        self._end_span(exc_val)

    def _start_span(self) -> "SpanContext":
        """Start the Langfuse span."""
        try:
            client = get_client()

            # Determine span type based on level
            span_type_map = {
                SpanLevel.GENERATION: "generation",
                SpanLevel.TOOL: "span",  # Tools are regular spans in Langfuse
            }
            # Default to span for most levels

            # Build metadata
            full_metadata = {
                **self.metadata,
                **self._span_metadata.to_dict(),
            }

            # Create the span
            self._langfuse_span = client.span(
                name=self.name,
                metadata=full_metadata,
            )
            self._span_id = getattr(self._langfuse_span, "id", None)

            logger.debug(
                "span_started",
                name=self.name,
                level=self.level.value,
                span_id=self._span_id,
            )

        except Exception as e:
            logger.warning("span_start_failed", name=self.name, error=str(e))

        return self

    def _end_span(self, error: Exception | None = None) -> None:
        """End the Langfuse span."""
        self._span_metadata.ended_at = datetime.now(UTC)

        if error:
            self.set_error(error)

        try:
            if self._langfuse_span:
                # Update with final metadata including reasoning
                final_metadata = {
                    **self.metadata,
                    **self._span_metadata.to_dict(),
                }
                if self._span_metadata.reasoning:
                    final_metadata["reasoning"] = self._span_metadata.reasoning

                self._langfuse_span.update(
                    metadata=final_metadata,
                    end_time=self._span_metadata.ended_at,
                )
                self._langfuse_span.end()

            logger.debug(
                "span_ended",
                name=self.name,
                level=self.level.value,
                has_error=error is not None,
            )

        except Exception as e:
            logger.warning("span_end_failed", name=self.name, error=str(e))


# ============================================================================
# Convenience Functions for Span Creation
# ============================================================================


def tool_span(
    name: str,
    *,
    input_data: dict[str, Any] | None = None,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create a tool-level span with input/output capture.

    Args:
        name: Name of the tool.
        input_data: Input data to capture.
        parent_span_id: Parent span ID for nesting.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for tool execution.

    Example:
        async with tool_span("get_market_data", input_data={"symbol": "AAPL"}) as span:
            result = await fetch_market_data("AAPL")
            span.set_output({"price": 150.0})
    """
    ctx = SpanContext(
        name=name,
        level=SpanLevel.TOOL,
        tool_name=name,
        parent_span_id=parent_span_id,
        metadata=metadata,
        capture_input=True,
        capture_output=True,
    )
    if input_data:
        ctx._span_metadata.input_data = input_data
    return ctx


def agent_span(
    name: str,
    *,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create an agent-level span.

    Args:
        name: Name of the agent.
        parent_span_id: Parent span ID for nesting.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for agent execution.

    Example:
        async with agent_span("research_agent") as span:
            span.add_reasoning("Starting market research")
            result = await research()
    """
    return SpanContext(
        name=name,
        level=SpanLevel.AGENT,
        agent_name=name,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )


def generation_span(
    name: str,
    *,
    model: str | None = None,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create a generation-level span for LLM calls.

    Args:
        name: Name of the generation.
        model: Model name being used.
        parent_span_id: Parent span ID for nesting.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for LLM generation.

    Example:
        async with generation_span("analyze_portfolio", model="claude-sonnet") as span:
            span.add_reasoning("Analyzing risk factors")
            response = await llm.generate(prompt)
            span.set_output({"response": response})
    """
    return SpanContext(
        name=name,
        level=SpanLevel.GENERATION,
        model=model,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )


def preprocessing_span(
    name: str = "preprocessing",
    *,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create a preprocessing span.

    Args:
        name: Name of the preprocessing step.
        parent_span_id: Parent span ID for nesting.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for preprocessing.

    Example:
        async with preprocessing_span() as span:
            span.set_input({"raw_data": data})
            processed = clean_and_validate(data)
            span.set_output({"processed_data": processed})
    """
    return SpanContext(
        name=name,
        level=SpanLevel.PREPROCESSING,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )


def postprocessing_span(
    name: str = "postprocessing",
    *,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create a postprocessing span.

    Args:
        name: Name of the postprocessing step.
        parent_span_id: Parent span ID for nesting.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for postprocessing.

    Example:
        async with postprocessing_span() as span:
            span.set_input({"raw_result": result})
            formatted = format_response(result)
            span.set_output({"formatted": formatted})
    """
    return SpanContext(
        name=name,
        level=SpanLevel.POSTPROCESSING,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )


def planning_span(
    name: str = "planning",
    *,
    parent_span_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create a planning span for LLM reasoning.

    Args:
        name: Name of the planning step.
        parent_span_id: Parent span ID for nesting.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for planning/reasoning.

    Example:
        async with planning_span() as span:
            span.add_reasoning("Evaluating market conditions")
            span.add_reasoning("Identified 3 risk factors")
            plan = create_analysis_plan()
    """
    return SpanContext(
        name=name,
        level=SpanLevel.PLANNING,
        parent_span_id=parent_span_id,
        metadata=metadata,
    )


def request_span(
    name: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SpanContext:
    """Create a request-level span.

    Args:
        name: Name of the request.
        session_id: Session identifier.
        user_id: User identifier.
        metadata: Additional metadata.

    Returns:
        SpanContext configured for a request.

    Example:
        async with request_span("portfolio_analysis", session_id="sess-123") as span:
            span.set_input({"symbols": ["AAPL", "GOOGL"]})
            result = await process_request()
            span.set_output({"status": "completed"})
    """
    full_metadata = metadata or {}
    if session_id:
        full_metadata["session_id"] = session_id
    if user_id:
        full_metadata["user_id"] = user_id

    return SpanContext(
        name=name,
        level=SpanLevel.REQUEST,
        metadata=full_metadata,
    )


# ============================================================================
# Langfuse Client
# ============================================================================


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
