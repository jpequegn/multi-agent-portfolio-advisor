"""Cache metrics tracking for observability.

This module provides tracking and analysis of prompt cache performance,
including hit rates, cost savings, and integration with Langfuse.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog
from langfuse import get_client

logger = structlog.get_logger(__name__)


# Cost multipliers for cache operations (relative to base input token cost)
CACHE_WRITE_COST_MULTIPLIER = 1.25  # 25% more than base input tokens
CACHE_READ_COST_MULTIPLIER = 0.10  # 10% of base input token price


@dataclass
class CacheMetrics:
    """Metrics for a single API call with caching.

    Attributes:
        input_tokens: Regular input tokens processed.
        output_tokens: Output tokens generated.
        cache_creation_tokens: Tokens written to cache (first call).
        cache_read_tokens: Tokens read from cache (subsequent calls).
        model: Model used for the call.
        agent_name: Name of the agent making the call.
        timestamp: When the call was made.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = ""
    agent_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cached.

        Returns:
            Sum of regular and cached input tokens.
        """
        return self.input_tokens + self.cache_read_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Returns:
            Percentage of input tokens read from cache (0-1).
        """
        total = self.total_input_tokens
        if total == 0:
            return 0.0
        return self.cache_read_tokens / total

    @property
    def estimated_savings_percent(self) -> float:
        """Estimate cost savings from caching.

        Calculates the percentage cost reduction compared to
        processing all tokens without caching.

        Returns:
            Estimated savings as a percentage (0-100).
        """
        if self.cache_read_tokens == 0:
            return 0.0

        # Cost without caching: all tokens at full price
        uncached_cost = self.total_input_tokens

        # Cost with caching: read tokens at 10% + creation tokens at 125%
        cached_cost = (
            self.input_tokens
            + (self.cache_read_tokens * CACHE_READ_COST_MULTIPLIER)
            + (self.cache_creation_tokens * CACHE_WRITE_COST_MULTIPLIER)
        )

        if uncached_cost == 0:
            return 0.0

        savings = (uncached_cost - cached_cost) / uncached_cost * 100
        return max(0.0, savings)  # Ensure non-negative

    @property
    def is_cache_hit(self) -> bool:
        """Check if this call had a cache hit.

        Returns:
            True if any tokens were read from cache.
        """
        return self.cache_read_tokens > 0

    @property
    def is_cache_creation(self) -> bool:
        """Check if this call created a cache entry.

        Returns:
            True if tokens were written to cache.
        """
        return self.cache_creation_tokens > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_input_tokens": self.total_input_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "estimated_savings_percent": round(self.estimated_savings_percent, 2),
            "is_cache_hit": self.is_cache_hit,
            "is_cache_creation": self.is_cache_creation,
            "model": self.model,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_response(
        cls,
        response: Any,
        *,
        model: str = "",
        agent_name: str = "",
    ) -> "CacheMetrics":
        """Create metrics from an Anthropic API response.

        Args:
            response: Anthropic API response object.
            model: Model name for tracking.
            agent_name: Agent name for tracking.

        Returns:
            CacheMetrics populated from response usage data.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return cls(model=model, agent_name=agent_name)

        return cls(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
            model=model,
            agent_name=agent_name,
        )

    @classmethod
    def from_usage_dict(
        cls,
        usage: dict[str, int],
        *,
        model: str = "",
        agent_name: str = "",
    ) -> "CacheMetrics":
        """Create metrics from a usage dictionary.

        Args:
            usage: Dictionary with token counts.
            model: Model name for tracking.
            agent_name: Agent name for tracking.

        Returns:
            CacheMetrics populated from usage dict.
        """
        return cls(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            model=model,
            agent_name=agent_name,
        )


class CacheMetricsCollector:
    """Collector for aggregating cache metrics across multiple calls.

    Tracks cache performance over time and can report to Langfuse
    for observability dashboards.

    Example:
        collector = CacheMetricsCollector(workflow_id="wf-123")

        # After each API call
        metrics = CacheMetrics.from_response(response, agent_name="research")
        collector.record(metrics)

        # At end of workflow
        summary = collector.get_summary()
        collector.report_to_langfuse()
    """

    def __init__(
        self,
        workflow_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the collector.

        Args:
            workflow_id: Optional workflow ID for grouping.
            session_id: Optional session ID for Langfuse.
        """
        self.workflow_id = workflow_id
        self.session_id = session_id
        self._metrics: list[CacheMetrics] = []
        self._start_time = datetime.now(UTC)

    def record(self, metrics: CacheMetrics) -> None:
        """Record metrics from an API call.

        Args:
            metrics: Cache metrics from the call.
        """
        self._metrics.append(metrics)
        logger.debug(
            "cache_metrics_recorded",
            workflow_id=self.workflow_id,
            agent_name=metrics.agent_name,
            cache_hit=metrics.is_cache_hit,
            cache_creation=metrics.is_cache_creation,
            hit_rate=f"{metrics.cache_hit_rate:.1%}",
        )

    def get_summary(self) -> dict[str, Any]:
        """Get aggregated summary of all recorded metrics.

        Returns:
            Dictionary with aggregate statistics.
        """
        if not self._metrics:
            return {
                "call_count": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_creation_tokens": 0,
                "total_cache_read_tokens": 0,
                "overall_cache_hit_rate": 0.0,
                "overall_savings_percent": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
            }

        total_input = sum(m.input_tokens for m in self._metrics)
        total_output = sum(m.output_tokens for m in self._metrics)
        total_cache_creation = sum(m.cache_creation_tokens for m in self._metrics)
        total_cache_read = sum(m.cache_read_tokens for m in self._metrics)

        # Overall hit rate across all calls
        total_all_input = total_input + total_cache_read
        overall_hit_rate = total_cache_read / total_all_input if total_all_input > 0 else 0.0

        # Overall savings
        uncached_cost = total_all_input
        cached_cost = (
            total_input
            + (total_cache_read * CACHE_READ_COST_MULTIPLIER)
            + (total_cache_creation * CACHE_WRITE_COST_MULTIPLIER)
        )
        overall_savings = (
            (uncached_cost - cached_cost) / uncached_cost * 100
            if uncached_cost > 0
            else 0.0
        )

        cache_hits = sum(1 for m in self._metrics if m.is_cache_hit)
        cache_misses = len(self._metrics) - cache_hits

        return {
            "call_count": len(self._metrics),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_creation_tokens": total_cache_creation,
            "total_cache_read_tokens": total_cache_read,
            "overall_cache_hit_rate": round(overall_hit_rate, 4),
            "overall_savings_percent": round(max(0.0, overall_savings), 2),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "by_agent": self._get_by_agent_summary(),
            "workflow_id": self.workflow_id,
            "duration_seconds": (datetime.now(UTC) - self._start_time).total_seconds(),
        }

    def _get_by_agent_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary broken down by agent.

        Returns:
            Dictionary mapping agent names to their metrics.
        """
        by_agent: dict[str, list[CacheMetrics]] = {}
        for m in self._metrics:
            agent = m.agent_name or "unknown"
            if agent not in by_agent:
                by_agent[agent] = []
            by_agent[agent].append(m)

        result = {}
        for agent, metrics in by_agent.items():
            total_input = sum(m.input_tokens for m in metrics)
            total_cache_read = sum(m.cache_read_tokens for m in metrics)
            total_all = total_input + total_cache_read

            result[agent] = {
                "call_count": len(metrics),
                "total_input_tokens": total_input,
                "cache_read_tokens": total_cache_read,
                "cache_hit_rate": round(total_cache_read / total_all, 4) if total_all > 0 else 0.0,
                "cache_hits": sum(1 for m in metrics if m.is_cache_hit),
            }

        return result

    def report_to_langfuse(self) -> None:
        """Report aggregated metrics to Langfuse.

        Creates a score in Langfuse with cache performance metrics.
        """
        try:
            summary = self.get_summary()

            client = get_client()

            # Report cache hit rate as a score
            if self.session_id:
                client.score(
                    trace_id=self.session_id,
                    name="cache_hit_rate",
                    value=summary["overall_cache_hit_rate"],
                    comment=f"Cache hits: {summary['cache_hits']}/{summary['call_count']}",
                )

                client.score(
                    trace_id=self.session_id,
                    name="cache_savings_percent",
                    value=summary["overall_savings_percent"],
                    comment="Estimated cost savings from prompt caching",
                )

            logger.info(
                "cache_metrics_reported_to_langfuse",
                workflow_id=self.workflow_id,
                session_id=self.session_id,
                hit_rate=f"{summary['overall_cache_hit_rate']:.1%}",
                savings=f"{summary['overall_savings_percent']:.1f}%",
            )

        except Exception as e:
            logger.warning(
                "langfuse_report_failed",
                error=str(e),
                workflow_id=self.workflow_id,
            )

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._metrics = []
        self._start_time = datetime.now(UTC)
        logger.debug("cache_metrics_cleared", workflow_id=self.workflow_id)
