"""Observability integration for prompt caching.

This module provides integration between cache metrics and Langfuse
for monitoring cache performance in production.
"""

from dataclasses import dataclass
from typing import Any

import structlog
from langfuse import get_client

from src.caching.metrics import CacheMetrics, CacheMetricsCollector

logger = structlog.get_logger(__name__)


@dataclass
class CacheObservabilityConfig:
    """Configuration for cache observability.

    Attributes:
        alert_on_low_hit_rate: Trigger alert when hit rate falls below threshold.
        low_hit_rate_threshold: Minimum acceptable hit rate (0-1).
        track_by_agent: Whether to track metrics per agent.
        track_by_model: Whether to track metrics per model.
    """

    alert_on_low_hit_rate: bool = True
    low_hit_rate_threshold: float = 0.8
    track_by_agent: bool = True
    track_by_model: bool = True


class CacheObservabilityReporter:
    """Reports cache metrics to Langfuse for observability.

    Integrates with Langfuse to provide:
    - Cache hit rate tracking per request
    - Cost savings estimation
    - Alerts for low cache hit rates
    - Per-agent cache performance breakdown

    Example:
        reporter = CacheObservabilityReporter(trace_id="trace-123")

        # During workflow execution
        metrics = CacheMetrics.from_response(response, agent_name="research")
        reporter.record(metrics)

        # At end of workflow
        reporter.report_summary()
    """

    def __init__(
        self,
        trace_id: str,
        *,
        session_id: str | None = None,
        config: CacheObservabilityConfig | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            trace_id: Langfuse trace ID for the current request.
            session_id: Optional session ID for grouping.
            config: Optional configuration overrides.
        """
        self.trace_id = trace_id
        self.session_id = session_id
        self.config = config or CacheObservabilityConfig()
        self._collector = CacheMetricsCollector(
            workflow_id=trace_id,
            session_id=session_id,
        )

    def record(self, metrics: CacheMetrics) -> None:
        """Record metrics from an API call.

        Args:
            metrics: Cache metrics from the call.
        """
        self._collector.record(metrics)

        # Report individual call metrics to Langfuse
        self._report_call_metrics(metrics)

    def _report_call_metrics(self, metrics: CacheMetrics) -> None:
        """Report metrics from a single API call to Langfuse.

        Args:
            metrics: Cache metrics to report.
        """
        try:
            client = get_client()

            # Create a span for the cache metrics
            span = client.span(
                trace_id=self.trace_id,
                name=f"cache_metrics_{metrics.agent_name}",
                metadata={
                    "cache_hit": metrics.is_cache_hit,
                    "cache_creation": metrics.is_cache_creation,
                    "input_tokens": metrics.input_tokens,
                    "cache_read_tokens": metrics.cache_read_tokens,
                    "cache_creation_tokens": metrics.cache_creation_tokens,
                    "cache_hit_rate": metrics.cache_hit_rate,
                    "estimated_savings_percent": metrics.estimated_savings_percent,
                    "agent_name": metrics.agent_name,
                    "model": metrics.model,
                },
            )
            span.end()

            logger.debug(
                "cache_call_metrics_reported",
                trace_id=self.trace_id,
                agent=metrics.agent_name,
                hit_rate=f"{metrics.cache_hit_rate:.1%}",
            )

        except Exception as e:
            logger.warning(
                "cache_call_metrics_report_failed",
                trace_id=self.trace_id,
                error=str(e),
            )

    def report_summary(self) -> dict[str, Any]:
        """Report aggregated metrics summary to Langfuse.

        Returns:
            Summary dictionary of cache metrics.
        """
        summary = self._collector.get_summary()

        try:
            client = get_client()

            # Report cache hit rate as a score
            client.score(
                trace_id=self.trace_id,
                name="cache_hit_rate",
                value=summary["overall_cache_hit_rate"],
                comment=f"Cache hits: {summary['cache_hits']}/{summary['call_count']} calls",
            )

            # Report cost savings as a score
            client.score(
                trace_id=self.trace_id,
                name="cache_cost_savings",
                value=summary["overall_savings_percent"] / 100,  # Normalize to 0-1
                comment=f"Estimated {summary['overall_savings_percent']:.1f}% cost reduction",
            )

            # Check for low hit rate alert
            if (
                self.config.alert_on_low_hit_rate
                and summary["overall_cache_hit_rate"] < self.config.low_hit_rate_threshold
            ):
                logger.warning(
                    "low_cache_hit_rate_alert",
                    trace_id=self.trace_id,
                    hit_rate=summary["overall_cache_hit_rate"],
                    threshold=self.config.low_hit_rate_threshold,
                )

                # Add event for alert
                client.trace_update(
                    trace_id=self.trace_id,
                    metadata={
                        "alert": "low_cache_hit_rate",
                        "cache_hit_rate": summary["overall_cache_hit_rate"],
                        "threshold": self.config.low_hit_rate_threshold,
                    },
                )

            # Report per-agent breakdown if configured
            if self.config.track_by_agent and summary.get("by_agent"):
                for agent_name, agent_metrics in summary["by_agent"].items():
                    client.score(
                        trace_id=self.trace_id,
                        name=f"cache_hit_rate_{agent_name}",
                        value=agent_metrics["cache_hit_rate"],
                        comment=f"{agent_name}: {agent_metrics['cache_hits']} cache hits",
                    )

            logger.info(
                "cache_summary_reported",
                trace_id=self.trace_id,
                hit_rate=f"{summary['overall_cache_hit_rate']:.1%}",
                savings=f"{summary['overall_savings_percent']:.1f}%",
                calls=summary["call_count"],
            )

        except Exception as e:
            logger.warning(
                "cache_summary_report_failed",
                trace_id=self.trace_id,
                error=str(e),
            )

        return summary

    def get_metrics_for_state(self) -> dict[str, Any]:
        """Get cache metrics suitable for storing in workflow state.

        Returns:
            Dictionary of cache metrics for state storage.
        """
        summary = self._collector.get_summary()
        return {
            "overall_cache_hit_rate": summary["overall_cache_hit_rate"],
            "overall_savings_percent": summary["overall_savings_percent"],
            "total_cache_read_tokens": summary["total_cache_read_tokens"],
            "total_cache_creation_tokens": summary["total_cache_creation_tokens"],
            "cache_hits": summary["cache_hits"],
            "cache_misses": summary["cache_misses"],
            "by_agent": summary.get("by_agent", {}),
        }


def create_cache_reporter(
    trace_id: str,
    session_id: str | None = None,
) -> CacheObservabilityReporter:
    """Create a cache observability reporter.

    Convenience function for creating a reporter with default config.

    Args:
        trace_id: Langfuse trace ID.
        session_id: Optional session ID.

    Returns:
        Configured CacheObservabilityReporter.
    """
    return CacheObservabilityReporter(
        trace_id=trace_id,
        session_id=session_id,
    )
