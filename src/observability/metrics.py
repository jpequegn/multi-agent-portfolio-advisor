"""Custom attributes and metadata for observability.

This module provides rich metadata schemas for traces to enable
better debugging, filtering, and analysis in Langfuse.

Metadata Levels:
- Request: Portfolio-level attributes (size, value, user tier)
- Agent: Agent execution attributes (tools used, reasoning steps)
- Tool: Tool execution attributes (API source, cache status)

Tagging System:
- Automatic tags based on conditions (high-value, error-recovered)
- Custom tags for categorization and filtering
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Tag Definitions
# ============================================================================


class Tag(Enum):
    """Standard tags for trace categorization."""

    # Analysis type tags
    PORTFOLIO_ANALYSIS = "portfolio-analysis"
    RISK_ASSESSMENT = "risk-assessment"
    REBALANCING = "rebalancing"

    # Portfolio characteristics
    HIGH_VALUE_PORTFOLIO = "high-value-portfolio"
    LARGE_PORTFOLIO = "large-portfolio"
    CONCENTRATED_PORTFOLIO = "concentrated-portfolio"

    # Execution characteristics
    MOCK_DATA_USED = "mock-data-used"
    CACHE_HIT = "cache-hit"
    API_FALLBACK = "api-fallback"

    # Error handling
    ERROR_RECOVERED = "error-recovered"
    PARTIAL_RESULT = "partial-result"
    RETRY_SUCCEEDED = "retry-succeeded"

    # User tiers
    USER_PREMIUM = "user-premium"
    USER_STANDARD = "user-standard"

    # Performance
    SLOW_REQUEST = "slow-request"
    HIGH_TOKEN_USAGE = "high-token-usage"


# ============================================================================
# Request-Level Metadata
# ============================================================================


@dataclass
class RequestMetadata:
    """Metadata for request-level traces.

    Captures portfolio and request characteristics for filtering
    and analysis in Langfuse.

    Attributes:
        portfolio_size: Number of positions in the portfolio.
        portfolio_value: Total value of the portfolio.
        symbols: List of symbols being analyzed.
        user_id: Optional user identifier.
        user_tier: User subscription tier.
        request_type: Type of analysis request.
        workflow_id: Unique workflow identifier.
        started_at: When the request started.
        custom: Additional custom metadata.
    """

    portfolio_size: int = 0
    portfolio_value: float = 0.0
    symbols: list[str] = field(default_factory=list)
    user_id: str | None = None
    user_tier: str = "standard"
    request_type: str = "full_analysis"
    workflow_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Langfuse metadata."""
        return {
            "portfolio_size": self.portfolio_size,
            "portfolio_value": self.portfolio_value,
            "symbols": self.symbols,
            "symbols_count": len(self.symbols),
            "user_id": self.user_id,
            "user_tier": self.user_tier,
            "request_type": self.request_type,
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            **self.custom,
        }

    def get_auto_tags(self) -> list[str]:
        """Generate automatic tags based on metadata values."""
        tags: list[str] = []

        # Portfolio analysis tag
        tags.append(Tag.PORTFOLIO_ANALYSIS.value)

        # High-value portfolio (> $100k)
        if self.portfolio_value > 100_000:
            tags.append(Tag.HIGH_VALUE_PORTFOLIO.value)

        # Large portfolio (> 20 positions)
        if self.portfolio_size > 20:
            tags.append(Tag.LARGE_PORTFOLIO.value)

        # User tier tags
        if self.user_tier == "premium":
            tags.append(Tag.USER_PREMIUM.value)
        else:
            tags.append(Tag.USER_STANDARD.value)

        return tags

    @classmethod
    def from_portfolio_state(cls, state: dict[str, Any]) -> "RequestMetadata":
        """Create metadata from a PortfolioState dict.

        Args:
            state: Portfolio state dictionary.

        Returns:
            RequestMetadata instance.
        """
        portfolio = state.get("portfolio", {})
        positions = portfolio.get("positions", [])

        # Calculate portfolio value
        total_value = sum(
            pos.get("quantity", 0) * pos.get("current_price", 0)
            for pos in positions
        )

        return cls(
            portfolio_size=len(positions),
            portfolio_value=total_value,
            symbols=state.get("symbols", []),
            user_id=state.get("user_id"),
            user_tier=state.get("user_tier", "standard"),
            request_type=state.get("request_type", "full_analysis"),
            workflow_id=state.get("workflow_id"),
        )


# ============================================================================
# Agent-Level Metadata
# ============================================================================


@dataclass
class AgentMetadata:
    """Metadata for agent-level traces.

    Captures agent execution characteristics for debugging
    and performance analysis.

    Attributes:
        agent_name: Name of the agent.
        tools_used: List of tools the agent called.
        tool_calls_count: Total number of tool calls.
        reasoning_steps: Number of reasoning/planning steps.
        input_tokens: Tokens used for input.
        output_tokens: Tokens used for output.
        duration_ms: Execution duration in milliseconds.
        errors: List of errors encountered.
        retries: Number of retry attempts.
        custom: Additional custom metadata.
    """

    agent_name: str = ""
    tools_used: list[str] = field(default_factory=list)
    tool_calls_count: int = 0
    reasoning_steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    retries: int = 0
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Langfuse metadata."""
        return {
            "agent": self.agent_name,
            "tools_used": self.tools_used,
            "tools_used_count": len(self.tools_used),
            "tool_calls_count": self.tool_calls_count,
            "reasoning_steps": self.reasoning_steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "duration_ms": self.duration_ms,
            "has_errors": len(self.errors) > 0,
            "error_count": len(self.errors),
            "retries": self.retries,
            **self.custom,
        }

    def add_tool_call(self, tool_name: str) -> None:
        """Record a tool call.

        Args:
            tool_name: Name of the tool that was called.
        """
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        self.tool_calls_count += 1

    def add_reasoning_step(self) -> None:
        """Record a reasoning step."""
        self.reasoning_steps += 1

    def add_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error message.
        """
        self.errors.append(error)

    def get_auto_tags(self) -> list[str]:
        """Generate automatic tags based on metadata values."""
        tags: list[str] = []

        # Error handling tags
        if self.errors and self.retries > 0:
            tags.append(Tag.ERROR_RECOVERED.value)
        if self.retries > 0:
            tags.append(Tag.RETRY_SUCCEEDED.value)

        # High token usage (> 10k tokens)
        if (self.input_tokens + self.output_tokens) > 10_000:
            tags.append(Tag.HIGH_TOKEN_USAGE.value)

        # Slow execution (> 30 seconds)
        if self.duration_ms > 30_000:
            tags.append(Tag.SLOW_REQUEST.value)

        return tags


# ============================================================================
# Tool-Level Metadata
# ============================================================================


@dataclass
class ToolMetadata:
    """Metadata for tool-level traces.

    Captures tool execution details for debugging and
    performance analysis.

    Attributes:
        tool_name: Name of the tool.
        api_source: Source of data (real, mock, cache).
        symbols_requested: Number of symbols in the request.
        cache_hit: Whether the result was from cache.
        cache_key: Cache key if applicable.
        api_latency_ms: API call latency in milliseconds.
        response_size: Size of response data.
        error: Error message if failed.
        retry_count: Number of retries.
        custom: Additional custom metadata.
    """

    tool_name: str = ""
    api_source: str = "real"  # real, mock, cache
    symbols_requested: int = 0
    cache_hit: bool = False
    cache_key: str | None = None
    api_latency_ms: float = 0.0
    response_size: int = 0
    error: str | None = None
    retry_count: int = 0
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Langfuse metadata."""
        return {
            "tool": self.tool_name,
            "api_source": self.api_source,
            "symbols_requested": self.symbols_requested,
            "cache_hit": self.cache_hit,
            "cache_key": self.cache_key,
            "api_latency_ms": self.api_latency_ms,
            "response_size": self.response_size,
            "has_error": self.error is not None,
            "error": self.error,
            "retry_count": self.retry_count,
            **self.custom,
        }

    def get_auto_tags(self) -> list[str]:
        """Generate automatic tags based on metadata values."""
        tags: list[str] = []

        # Data source tags
        if self.api_source == "mock":
            tags.append(Tag.MOCK_DATA_USED.value)
        if self.cache_hit:
            tags.append(Tag.CACHE_HIT.value)
        if self.api_source == "fallback":
            tags.append(Tag.API_FALLBACK.value)

        return tags


# ============================================================================
# Tag Manager
# ============================================================================


class TagManager:
    """Manages tags for traces and spans.

    Provides utilities for collecting, deduplicating, and
    formatting tags from multiple sources.
    """

    def __init__(self) -> None:
        """Initialize the tag manager."""
        self._tags: set[str] = set()

    @property
    def tags(self) -> list[str]:
        """Get all tags as a sorted list."""
        return sorted(self._tags)

    def add(self, tag: str | Tag) -> None:
        """Add a tag.

        Args:
            tag: Tag to add (string or Tag enum).
        """
        if isinstance(tag, Tag):
            self._tags.add(tag.value)
        else:
            self._tags.add(tag)

    def add_many(self, tags: list[str | Tag]) -> None:
        """Add multiple tags.

        Args:
            tags: List of tags to add.
        """
        for tag in tags:
            self.add(tag)

    def add_from_metadata(
        self,
        *metadata: RequestMetadata | AgentMetadata | ToolMetadata,
    ) -> None:
        """Add auto-generated tags from metadata objects.

        Args:
            metadata: Metadata objects to extract tags from.
        """
        for m in metadata:
            self.add_many(m.get_auto_tags())

    def remove(self, tag: str | Tag) -> None:
        """Remove a tag.

        Args:
            tag: Tag to remove.
        """
        tag_value = tag.value if isinstance(tag, Tag) else tag
        self._tags.discard(tag_value)

    def has(self, tag: str | Tag) -> bool:
        """Check if a tag exists.

        Args:
            tag: Tag to check.

        Returns:
            True if tag exists.
        """
        tag_value = tag.value if isinstance(tag, Tag) else tag
        return tag_value in self._tags

    def clear(self) -> None:
        """Remove all tags."""
        self._tags.clear()


# ============================================================================
# Metadata Builder
# ============================================================================


class MetadataBuilder:
    """Builder for constructing trace metadata.

    Provides a fluent interface for building metadata from
    multiple sources and combining with tags.

    Example:
        metadata = (
            MetadataBuilder()
            .with_request(request_meta)
            .with_agent(agent_meta)
            .with_tags(["custom-tag"])
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the metadata builder."""
        self._metadata: dict[str, Any] = {}
        self._tags = TagManager()

    def with_request(self, metadata: RequestMetadata) -> "MetadataBuilder":
        """Add request-level metadata.

        Args:
            metadata: Request metadata.

        Returns:
            Self for chaining.
        """
        self._metadata["request"] = metadata.to_dict()
        self._tags.add_from_metadata(metadata)
        return self

    def with_agent(self, metadata: AgentMetadata) -> "MetadataBuilder":
        """Add agent-level metadata.

        Args:
            metadata: Agent metadata.

        Returns:
            Self for chaining.
        """
        self._metadata["agent"] = metadata.to_dict()
        self._tags.add_from_metadata(metadata)
        return self

    def with_tool(self, metadata: ToolMetadata) -> "MetadataBuilder":
        """Add tool-level metadata.

        Args:
            metadata: Tool metadata.

        Returns:
            Self for chaining.
        """
        self._metadata["tool"] = metadata.to_dict()
        self._tags.add_from_metadata(metadata)
        return self

    def with_tags(self, tags: list[str | Tag]) -> "MetadataBuilder":
        """Add custom tags.

        Args:
            tags: Tags to add.

        Returns:
            Self for chaining.
        """
        self._tags.add_many(tags)
        return self

    def with_custom(self, key: str, value: Any) -> "MetadataBuilder":
        """Add custom metadata field.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for chaining.
        """
        self._metadata[key] = value
        return self

    def build(self) -> dict[str, Any]:
        """Build the final metadata dictionary.

        Returns:
            Combined metadata dictionary.
        """
        return {
            **self._metadata,
            "tags": self._tags.tags,
        }

    def build_tags(self) -> list[str]:
        """Build just the tags list.

        Returns:
            List of all tags.
        """
        return self._tags.tags


# ============================================================================
# Convenience Functions
# ============================================================================


def create_request_metadata(
    portfolio_size: int,
    portfolio_value: float,
    symbols: list[str],
    *,
    user_id: str | None = None,
    user_tier: str = "standard",
    request_type: str = "full_analysis",
    workflow_id: str | None = None,
    **custom: Any,
) -> RequestMetadata:
    """Create request metadata with common fields.

    Args:
        portfolio_size: Number of positions.
        portfolio_value: Total portfolio value.
        symbols: List of symbols.
        user_id: Optional user ID.
        user_tier: User subscription tier.
        request_type: Type of request.
        workflow_id: Workflow identifier.
        **custom: Additional custom fields.

    Returns:
        RequestMetadata instance.
    """
    return RequestMetadata(
        portfolio_size=portfolio_size,
        portfolio_value=portfolio_value,
        symbols=symbols,
        user_id=user_id,
        user_tier=user_tier,
        request_type=request_type,
        workflow_id=workflow_id,
        custom=custom,
    )


def create_agent_metadata(
    agent_name: str,
    *,
    tools_used: list[str] | None = None,
    tool_calls_count: int = 0,
    reasoning_steps: int = 0,
    duration_ms: float = 0.0,
    **custom: Any,
) -> AgentMetadata:
    """Create agent metadata with common fields.

    Args:
        agent_name: Name of the agent.
        tools_used: List of tools used.
        tool_calls_count: Number of tool calls.
        reasoning_steps: Number of reasoning steps.
        duration_ms: Execution duration.
        **custom: Additional custom fields.

    Returns:
        AgentMetadata instance.
    """
    return AgentMetadata(
        agent_name=agent_name,
        tools_used=tools_used or [],
        tool_calls_count=tool_calls_count,
        reasoning_steps=reasoning_steps,
        duration_ms=duration_ms,
        custom=custom,
    )


def create_tool_metadata(
    tool_name: str,
    *,
    api_source: str = "real",
    symbols_requested: int = 0,
    cache_hit: bool = False,
    api_latency_ms: float = 0.0,
    **custom: Any,
) -> ToolMetadata:
    """Create tool metadata with common fields.

    Args:
        tool_name: Name of the tool.
        api_source: Data source (real, mock, cache).
        symbols_requested: Number of symbols requested.
        cache_hit: Whether result was cached.
        api_latency_ms: API latency in ms.
        **custom: Additional custom fields.

    Returns:
        ToolMetadata instance.
    """
    return ToolMetadata(
        tool_name=tool_name,
        api_source=api_source,
        symbols_requested=symbols_requested,
        cache_hit=cache_hit,
        api_latency_ms=api_latency_ms,
        custom=custom,
    )


def collect_tags(*sources: RequestMetadata | AgentMetadata | ToolMetadata) -> list[str]:
    """Collect and deduplicate tags from multiple metadata sources.

    Args:
        sources: Metadata objects to collect tags from.

    Returns:
        Sorted, deduplicated list of tags.
    """
    manager = TagManager()
    manager.add_from_metadata(*sources)
    return manager.tags
