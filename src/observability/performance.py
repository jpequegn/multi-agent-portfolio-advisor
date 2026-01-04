"""Performance monitoring system for observability.

This module provides comprehensive performance monitoring and bottleneck
identification for the multi-agent system.

Features:
- Latency tracking at all levels (request, agent, tool, LLM)
- Percentile calculations (P50, P90, P95, P99)
- Bottleneck detection and analysis
- Performance regression alerts
- Performance reports and recommendations

Performance Targets (from README):
- End-to-end latency < 30 seconds
- Cost per analysis < $0.50

Example:
    tracker = get_performance_tracker()

    # Record latencies
    tracker.record_request_latency("trace-123", 5000.0)
    tracker.record_agent_latency("trace-123", "research", 2000.0)
    tracker.record_tool_latency("trace-123", "get_market_data", 500.0)

    # Get metrics and detect bottlenecks
    metrics = tracker.get_metrics("trace-123")
    bottlenecks = identify_bottlenecks("trace-123")

    # Generate report
    report = generate_performance_report()
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.observability.costs import AlertSeverity

logger = structlog.get_logger(__name__)


# ============================================================================
# Performance Targets
# ============================================================================

# Performance targets from README
TARGET_END_TO_END_LATENCY_MS = 30_000  # 30 seconds
TARGET_COST_PER_ANALYSIS = 0.50  # $0.50

# Component-level targets (derived)
TARGET_AGENT_LATENCY_MS = 10_000  # 10 seconds per agent
TARGET_TOOL_LATENCY_MS = 2_000  # 2 seconds per tool
TARGET_LLM_LATENCY_MS = 5_000  # 5 seconds per LLM call


# ============================================================================
# Enums
# ============================================================================


class PerformanceLevel(Enum):
    """Performance level classification."""

    EXCELLENT = "excellent"  # Well under target
    GOOD = "good"  # Under target
    ACCEPTABLE = "acceptable"  # At or slightly over target
    DEGRADED = "degraded"  # Significantly over target
    CRITICAL = "critical"  # Far over target


class BottleneckType(Enum):
    """Types of performance bottlenecks."""

    SLOW_AGENT = "slow_agent"
    SLOW_TOOL = "slow_tool"
    SLOW_LLM = "slow_llm"
    HIGH_LATENCY = "high_latency"
    QUEUE_DELAY = "queue_delay"
    RESOURCE_CONTENTION = "resource_contention"
    NETWORK_LATENCY = "network_latency"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a request.

    Attributes:
        trace_id: Trace identifier.
        total_latency_ms: Total request latency.
        time_to_first_agent_ms: Time until first agent started.
        agent_latencies: Latency per agent.
        slowest_agent: Name of the slowest agent.
        tool_latencies: Latency per tool.
        slowest_tool: Name of the slowest tool.
        llm_latencies: List of LLM call latencies.
        tokens_per_second: Token generation rate.
        timestamp: When metrics were recorded.
    """

    trace_id: str
    total_latency_ms: float = 0.0
    time_to_first_agent_ms: float = 0.0

    # Agent level
    agent_latencies: dict[str, float] = field(default_factory=dict)
    slowest_agent: str = ""

    # Tool level
    tool_latencies: dict[str, float] = field(default_factory=dict)
    slowest_tool: str = ""

    # LLM level
    llm_latencies: list[float] = field(default_factory=list)
    tokens_per_second: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "total_latency_ms": self.total_latency_ms,
            "time_to_first_agent_ms": self.time_to_first_agent_ms,
            "agent_latencies": self.agent_latencies,
            "slowest_agent": self.slowest_agent,
            "tool_latencies": self.tool_latencies,
            "slowest_tool": self.slowest_tool,
            "llm_latencies": self.llm_latencies,
            "tokens_per_second": self.tokens_per_second,
            "timestamp": self.timestamp.isoformat(),
        }

    def get_performance_level(self) -> PerformanceLevel:
        """Determine overall performance level."""
        ratio = self.total_latency_ms / TARGET_END_TO_END_LATENCY_MS

        if ratio <= 0.5:
            return PerformanceLevel.EXCELLENT
        elif ratio <= 0.8:
            return PerformanceLevel.GOOD
        elif ratio <= 1.0:
            return PerformanceLevel.ACCEPTABLE
        elif ratio <= 1.5:
            return PerformanceLevel.DEGRADED
        else:
            return PerformanceLevel.CRITICAL


@dataclass
class LatencyRecord:
    """A single latency measurement."""

    trace_id: str
    component: str  # "request", "agent", "tool", "llm"
    name: str  # Component name (e.g., "research_agent")
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "component": self.component,
            "name": self.name,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class PercentileStats:
    """Percentile statistics for a metric.

    Attributes:
        metric_name: Name of the metric.
        count: Number of samples.
        min_value: Minimum value.
        max_value: Maximum value.
        mean: Average value.
        p50: 50th percentile (median).
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        time_window: Time window for these stats.
    """

    metric_name: str
    count: int = 0
    min_value: float = 0.0
    max_value: float = 0.0
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    time_window: str = "1h"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.mean,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "time_window": self.time_window,
        }


@dataclass
class Bottleneck:
    """A detected performance bottleneck.

    Attributes:
        bottleneck_id: Unique identifier.
        trace_id: Associated trace.
        bottleneck_type: Type of bottleneck.
        component: Affected component name.
        latency_ms: Observed latency.
        expected_ms: Expected/target latency.
        impact_score: Impact score (0-1).
        suggestion: Optimization suggestion.
    """

    bottleneck_id: str
    trace_id: str
    bottleneck_type: BottleneckType
    component: str
    latency_ms: float
    expected_ms: float
    impact_score: float
    suggestion: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bottleneck_id": self.bottleneck_id,
            "trace_id": self.trace_id,
            "bottleneck_type": self.bottleneck_type.value,
            "component": self.component,
            "latency_ms": self.latency_ms,
            "expected_ms": self.expected_ms,
            "impact_score": self.impact_score,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PerformanceAlert:
    """A performance regression alert.

    Attributes:
        alert_id: Unique identifier.
        severity: Alert severity.
        metric_name: Affected metric.
        current_value: Current metric value.
        baseline_value: Baseline/expected value.
        deviation_percent: Percentage deviation from baseline.
        message: Alert message.
        triggered_at: When the alert was triggered.
    """

    alert_id: str
    severity: AlertSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    message: str
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "deviation_percent": self.deviation_percent,
            "message": self.message,
            "triggered_at": self.triggered_at.isoformat(),
        }


@dataclass
class PerformanceThresholds:
    """Thresholds for performance alerts.

    Attributes:
        warning_deviation_percent: Deviation % for warning alerts.
        error_deviation_percent: Deviation % for error alerts.
        critical_deviation_percent: Deviation % for critical alerts.
        min_samples: Minimum samples before alerting.
    """

    warning_deviation_percent: float = 20.0
    error_deviation_percent: float = 50.0
    critical_deviation_percent: float = 100.0
    min_samples: int = 10


@dataclass
class PerformanceReport:
    """Performance report with aggregated statistics.

    Attributes:
        report_id: Unique identifier.
        time_window: Time window covered.
        generated_at: When report was generated.
        total_requests: Number of requests.
        request_latency_stats: Request latency percentiles.
        agent_latency_stats: Per-agent latency percentiles.
        tool_latency_stats: Per-tool latency percentiles.
        llm_latency_stats: LLM latency percentiles.
        bottlenecks: Detected bottlenecks.
        alerts: Triggered alerts.
        recommendations: Optimization recommendations.
    """

    report_id: str
    time_window: str
    generated_at: datetime
    total_requests: int
    request_latency_stats: PercentileStats
    agent_latency_stats: dict[str, PercentileStats]
    tool_latency_stats: dict[str, PercentileStats]
    llm_latency_stats: PercentileStats
    bottlenecks: list[Bottleneck]
    alerts: list[PerformanceAlert]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "time_window": self.time_window,
            "generated_at": self.generated_at.isoformat(),
            "total_requests": self.total_requests,
            "request_latency_stats": self.request_latency_stats.to_dict(),
            "agent_latency_stats": {k: v.to_dict() for k, v in self.agent_latency_stats.items()},
            "tool_latency_stats": {k: v.to_dict() for k, v in self.tool_latency_stats.items()},
            "llm_latency_stats": self.llm_latency_stats.to_dict(),
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "alerts": [a.to_dict() for a in self.alerts],
            "recommendations": self.recommendations,
        }


# ============================================================================
# Percentile Calculator
# ============================================================================


class PercentileCalculator:
    """Calculates percentile statistics for latency data."""

    @staticmethod
    def calculate(values: list[float], metric_name: str, time_window: str = "1h") -> PercentileStats:
        """Calculate percentile statistics for a list of values.

        Args:
            values: List of numeric values.
            metric_name: Name of the metric.
            time_window: Time window description.

        Returns:
            PercentileStats with calculated values.
        """
        if not values:
            return PercentileStats(metric_name=metric_name, time_window=time_window)

        sorted_values = sorted(values)
        n = len(sorted_values)

        return PercentileStats(
            metric_name=metric_name,
            count=n,
            min_value=sorted_values[0],
            max_value=sorted_values[-1],
            mean=statistics.mean(sorted_values),
            p50=PercentileCalculator._percentile(sorted_values, 50),
            p90=PercentileCalculator._percentile(sorted_values, 90),
            p95=PercentileCalculator._percentile(sorted_values, 95),
            p99=PercentileCalculator._percentile(sorted_values, 99),
            time_window=time_window,
        )

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        """Calculate a specific percentile from sorted values.

        Args:
            sorted_values: Pre-sorted list of values.
            percentile: Percentile to calculate (0-100).

        Returns:
            Percentile value.
        """
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]

        # Calculate index
        k = (percentile / 100) * (n - 1)
        f = int(k)
        c = f + 1 if f + 1 < n else f

        # Linear interpolation
        if f == c:
            return sorted_values[f]

        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1


# ============================================================================
# Performance Tracker
# ============================================================================


class PerformanceTracker:
    """Tracks performance metrics across requests.

    Collects latency data at all levels and provides aggregated statistics.
    """

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self._latency_records: list[LatencyRecord] = []
        self._metrics_by_trace: dict[str, PerformanceMetrics] = {}
        self._baselines: dict[str, float] = {}
        self._calculator = PercentileCalculator()

    def record_request_latency(
        self,
        trace_id: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record total request latency.

        Args:
            trace_id: Trace identifier.
            latency_ms: Latency in milliseconds.
            metadata: Optional metadata.
        """
        record = LatencyRecord(
            trace_id=trace_id,
            component="request",
            name="total",
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._latency_records.append(record)

        # Update metrics
        metrics = self._get_or_create_metrics(trace_id)
        metrics.total_latency_ms = latency_ms

        logger.debug(
            "request_latency_recorded",
            trace_id=trace_id,
            latency_ms=latency_ms,
        )

    def record_agent_latency(
        self,
        trace_id: str,
        agent_name: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record agent execution latency.

        Args:
            trace_id: Trace identifier.
            agent_name: Name of the agent.
            latency_ms: Latency in milliseconds.
            metadata: Optional metadata.
        """
        record = LatencyRecord(
            trace_id=trace_id,
            component="agent",
            name=agent_name,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._latency_records.append(record)

        # Update metrics
        metrics = self._get_or_create_metrics(trace_id)
        metrics.agent_latencies[agent_name] = latency_ms

        # Update slowest agent
        if not metrics.slowest_agent or latency_ms > metrics.agent_latencies.get(
            metrics.slowest_agent, 0
        ):
            metrics.slowest_agent = agent_name

        logger.debug(
            "agent_latency_recorded",
            trace_id=trace_id,
            agent_name=agent_name,
            latency_ms=latency_ms,
        )

    def record_tool_latency(
        self,
        trace_id: str,
        tool_name: str,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record tool execution latency.

        Args:
            trace_id: Trace identifier.
            tool_name: Name of the tool.
            latency_ms: Latency in milliseconds.
            metadata: Optional metadata.
        """
        record = LatencyRecord(
            trace_id=trace_id,
            component="tool",
            name=tool_name,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._latency_records.append(record)

        # Update metrics
        metrics = self._get_or_create_metrics(trace_id)
        metrics.tool_latencies[tool_name] = latency_ms

        # Update slowest tool
        if not metrics.slowest_tool or latency_ms > metrics.tool_latencies.get(
            metrics.slowest_tool, 0
        ):
            metrics.slowest_tool = tool_name

        logger.debug(
            "tool_latency_recorded",
            trace_id=trace_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
        )

    def record_llm_latency(
        self,
        trace_id: str,
        latency_ms: float,
        tokens_generated: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record LLM call latency.

        Args:
            trace_id: Trace identifier.
            latency_ms: Latency in milliseconds.
            tokens_generated: Number of tokens generated.
            metadata: Optional metadata.
        """
        record = LatencyRecord(
            trace_id=trace_id,
            component="llm",
            name="generation",
            latency_ms=latency_ms,
            metadata={"tokens_generated": tokens_generated, **(metadata or {})},
        )
        self._latency_records.append(record)

        # Update metrics
        metrics = self._get_or_create_metrics(trace_id)
        metrics.llm_latencies.append(latency_ms)

        # Calculate tokens per second
        if latency_ms > 0 and tokens_generated > 0:
            tps = (tokens_generated / latency_ms) * 1000
            # Update as weighted average
            if metrics.tokens_per_second > 0:
                metrics.tokens_per_second = (metrics.tokens_per_second + tps) / 2
            else:
                metrics.tokens_per_second = tps

        logger.debug(
            "llm_latency_recorded",
            trace_id=trace_id,
            latency_ms=latency_ms,
            tokens_generated=tokens_generated,
        )

    def record_time_to_first_agent(
        self,
        trace_id: str,
        latency_ms: float,
    ) -> None:
        """Record time until first agent started.

        Args:
            trace_id: Trace identifier.
            latency_ms: Latency in milliseconds.
        """
        metrics = self._get_or_create_metrics(trace_id)
        metrics.time_to_first_agent_ms = latency_ms

        logger.debug(
            "time_to_first_agent_recorded",
            trace_id=trace_id,
            latency_ms=latency_ms,
        )

    def get_metrics(self, trace_id: str) -> PerformanceMetrics | None:
        """Get performance metrics for a trace.

        Args:
            trace_id: Trace identifier.

        Returns:
            PerformanceMetrics or None if not found.
        """
        return self._metrics_by_trace.get(trace_id)

    def get_percentiles(
        self,
        component: str,
        name: str | None = None,
        time_window: timedelta = timedelta(hours=1),
    ) -> PercentileStats:
        """Get percentile statistics for a component.

        Args:
            component: Component type ("request", "agent", "tool", "llm").
            name: Optional component name for filtering.
            time_window: Time window to consider.

        Returns:
            PercentileStats for the component.
        """
        cutoff = datetime.now(UTC) - time_window

        # Filter records
        values = [
            r.latency_ms
            for r in self._latency_records
            if r.component == component
            and r.timestamp >= cutoff
            and (name is None or r.name == name)
        ]

        metric_name = f"{component}_{name}" if name else component
        window_str = self._format_time_window(time_window)

        return self._calculator.calculate(values, metric_name, window_str)

    def set_baseline(self, metric_name: str, value: float) -> None:
        """Set a baseline value for regression detection.

        Args:
            metric_name: Name of the metric.
            value: Baseline value.
        """
        self._baselines[metric_name] = value
        logger.debug("baseline_set", metric_name=metric_name, value=value)

    def get_baseline(self, metric_name: str) -> float | None:
        """Get baseline value for a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Baseline value or None.
        """
        return self._baselines.get(metric_name)

    def clear_old_records(self, max_age: timedelta = timedelta(days=7)) -> int:
        """Clear records older than max_age.

        Args:
            max_age: Maximum age of records to keep.

        Returns:
            Number of records removed.
        """
        cutoff = datetime.now(UTC) - max_age
        original_count = len(self._latency_records)
        self._latency_records = [r for r in self._latency_records if r.timestamp >= cutoff]
        removed = original_count - len(self._latency_records)

        if removed > 0:
            logger.info("old_records_cleared", removed=removed)

        return removed

    def _get_or_create_metrics(self, trace_id: str) -> PerformanceMetrics:
        """Get or create metrics for a trace."""
        if trace_id not in self._metrics_by_trace:
            self._metrics_by_trace[trace_id] = PerformanceMetrics(trace_id=trace_id)
        return self._metrics_by_trace[trace_id]

    @staticmethod
    def _format_time_window(td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h"
        else:
            return f"{total_seconds // 86400}d"


# ============================================================================
# Bottleneck Detector
# ============================================================================


class BottleneckDetector:
    """Detects performance bottlenecks in traces."""

    def __init__(self, tracker: PerformanceTracker) -> None:
        """Initialize detector.

        Args:
            tracker: Performance tracker to use.
        """
        self._tracker = tracker
        self._bottleneck_counter = 0

    def identify_bottlenecks(self, trace_id: str) -> list[Bottleneck]:
        """Identify performance bottlenecks in a request.

        Args:
            trace_id: Trace identifier.

        Returns:
            List of detected bottlenecks.
        """
        metrics = self._tracker.get_metrics(trace_id)
        if not metrics:
            return []

        bottlenecks: list[Bottleneck] = []

        # Check overall latency
        if metrics.total_latency_ms > TARGET_END_TO_END_LATENCY_MS:
            bottlenecks.append(
                self._create_bottleneck(
                    trace_id=trace_id,
                    bottleneck_type=BottleneckType.HIGH_LATENCY,
                    component="request",
                    latency_ms=metrics.total_latency_ms,
                    expected_ms=TARGET_END_TO_END_LATENCY_MS,
                    suggestion="Consider parallelizing agent execution or optimizing slow components.",
                )
            )

        # Check agent latencies
        for agent_name, latency in metrics.agent_latencies.items():
            if latency > TARGET_AGENT_LATENCY_MS:
                bottlenecks.append(
                    self._create_bottleneck(
                        trace_id=trace_id,
                        bottleneck_type=BottleneckType.SLOW_AGENT,
                        component=agent_name,
                        latency_ms=latency,
                        expected_ms=TARGET_AGENT_LATENCY_MS,
                        suggestion=f"Optimize {agent_name} or reduce tool calls.",
                    )
                )

        # Check tool latencies
        for tool_name, latency in metrics.tool_latencies.items():
            if latency > TARGET_TOOL_LATENCY_MS:
                bottlenecks.append(
                    self._create_bottleneck(
                        trace_id=trace_id,
                        bottleneck_type=BottleneckType.SLOW_TOOL,
                        component=tool_name,
                        latency_ms=latency,
                        expected_ms=TARGET_TOOL_LATENCY_MS,
                        suggestion=f"Add caching for {tool_name} or optimize API calls.",
                    )
                )

        # Check LLM latencies
        for i, latency in enumerate(metrics.llm_latencies):
            if latency > TARGET_LLM_LATENCY_MS:
                bottlenecks.append(
                    self._create_bottleneck(
                        trace_id=trace_id,
                        bottleneck_type=BottleneckType.SLOW_LLM,
                        component=f"llm_call_{i}",
                        latency_ms=latency,
                        expected_ms=TARGET_LLM_LATENCY_MS,
                        suggestion="Consider using a faster model or reducing prompt size.",
                    )
                )

        # Check time to first agent (queue delay)
        if metrics.time_to_first_agent_ms > 1000:  # > 1 second
            bottlenecks.append(
                self._create_bottleneck(
                    trace_id=trace_id,
                    bottleneck_type=BottleneckType.QUEUE_DELAY,
                    component="preprocessing",
                    latency_ms=metrics.time_to_first_agent_ms,
                    expected_ms=1000,
                    suggestion="Optimize preprocessing or increase concurrency.",
                )
            )

        logger.debug(
            "bottlenecks_identified",
            trace_id=trace_id,
            count=len(bottlenecks),
        )

        return bottlenecks

    def _create_bottleneck(
        self,
        trace_id: str,
        bottleneck_type: BottleneckType,
        component: str,
        latency_ms: float,
        expected_ms: float,
        suggestion: str,
    ) -> Bottleneck:
        """Create a bottleneck instance."""
        self._bottleneck_counter += 1
        impact_score = min(1.0, (latency_ms - expected_ms) / expected_ms)

        return Bottleneck(
            bottleneck_id=f"bn-{self._bottleneck_counter:06d}",
            trace_id=trace_id,
            bottleneck_type=bottleneck_type,
            component=component,
            latency_ms=latency_ms,
            expected_ms=expected_ms,
            impact_score=impact_score,
            suggestion=suggestion,
        )


# ============================================================================
# Performance Alert Manager
# ============================================================================


class PerformanceAlertManager:
    """Manages performance regression alerts."""

    def __init__(
        self,
        tracker: PerformanceTracker,
        thresholds: PerformanceThresholds | None = None,
    ) -> None:
        """Initialize alert manager.

        Args:
            tracker: Performance tracker to use.
            thresholds: Alert thresholds.
        """
        self._tracker = tracker
        self._thresholds = thresholds or PerformanceThresholds()
        self._alerts: list[PerformanceAlert] = []
        self._alert_counter = 0
        self._callbacks: list[Callable[[PerformanceAlert], None]] = []

    def register_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Register a callback for alert notifications.

        Args:
            callback: Function to call when alert is triggered.
        """
        self._callbacks.append(callback)

    def check_regression(
        self,
        metric_name: str,
        current_value: float,
    ) -> PerformanceAlert | None:
        """Check for performance regression.

        Args:
            metric_name: Name of the metric.
            current_value: Current metric value.

        Returns:
            PerformanceAlert if regression detected, None otherwise.
        """
        baseline = self._tracker.get_baseline(metric_name)
        if baseline is None or baseline == 0:
            return None

        deviation_percent = ((current_value - baseline) / baseline) * 100

        # Determine severity
        severity: AlertSeverity | None = None
        if deviation_percent >= self._thresholds.critical_deviation_percent:
            severity = AlertSeverity.CRITICAL
        elif deviation_percent >= self._thresholds.error_deviation_percent:
            severity = AlertSeverity.ERROR
        elif deviation_percent >= self._thresholds.warning_deviation_percent:
            severity = AlertSeverity.WARNING

        if severity is None:
            return None

        self._alert_counter += 1
        alert = PerformanceAlert(
            alert_id=f"perf-{self._alert_counter:06d}",
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline,
            deviation_percent=deviation_percent,
            message=f"{metric_name} increased by {deviation_percent:.1f}% "
            f"(baseline: {baseline:.2f}, current: {current_value:.2f})",
        )

        self._alerts.append(alert)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("alert_callback_failed", error=str(e))

        logger.warning(
            "performance_regression_detected",
            alert_id=alert.alert_id,
            metric_name=metric_name,
            severity=severity.value,
            deviation_percent=deviation_percent,
        )

        return alert

    def get_alerts(
        self,
        severity: AlertSeverity | None = None,
        since: datetime | None = None,
    ) -> list[PerformanceAlert]:
        """Get alerts with optional filtering.

        Args:
            severity: Filter by severity.
            since: Filter to alerts after this time.

        Returns:
            List of matching alerts.
        """
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if since:
            alerts = [a for a in alerts if a.triggered_at >= since]

        return alerts

    def clear_alerts(self, before: datetime | None = None) -> int:
        """Clear alerts.

        Args:
            before: Clear alerts before this time.

        Returns:
            Number of alerts cleared.
        """
        if before:
            original_count = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.triggered_at >= before]
            return original_count - len(self._alerts)
        else:
            count = len(self._alerts)
            self._alerts = []
            return count


# ============================================================================
# Report Generator
# ============================================================================


class PerformanceReportGenerator:
    """Generates performance reports."""

    def __init__(
        self,
        tracker: PerformanceTracker,
        detector: BottleneckDetector,
        alert_manager: PerformanceAlertManager,
    ) -> None:
        """Initialize report generator.

        Args:
            tracker: Performance tracker.
            detector: Bottleneck detector.
            alert_manager: Alert manager.
        """
        self._tracker = tracker
        self._detector = detector
        self._alert_manager = alert_manager
        self._report_counter = 0

    def generate_report(
        self,
        time_window: timedelta = timedelta(hours=1),
    ) -> PerformanceReport:
        """Generate a performance report.

        Args:
            time_window: Time window to cover.

        Returns:
            PerformanceReport with aggregated statistics.
        """
        self._report_counter += 1
        window_str = self._format_time_window(time_window)
        cutoff = datetime.now(UTC) - time_window

        # Get request latency stats
        request_stats = self._tracker.get_percentiles("request", time_window=time_window)

        # Get agent latency stats
        agent_names = self._get_unique_names("agent", cutoff)
        agent_stats = {
            name: self._tracker.get_percentiles("agent", name, time_window) for name in agent_names
        }

        # Get tool latency stats
        tool_names = self._get_unique_names("tool", cutoff)
        tool_stats = {
            name: self._tracker.get_percentiles("tool", name, time_window) for name in tool_names
        }

        # Get LLM latency stats
        llm_stats = self._tracker.get_percentiles("llm", time_window=time_window)

        # Collect bottlenecks from recent traces
        bottlenecks: list[Bottleneck] = []
        for trace_id in self._get_recent_trace_ids(cutoff):
            bottlenecks.extend(self._detector.identify_bottlenecks(trace_id))

        # Get recent alerts
        alerts = self._alert_manager.get_alerts(since=cutoff)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            request_stats,
            agent_stats,
            tool_stats,
            bottlenecks,
        )

        report = PerformanceReport(
            report_id=f"perf-report-{self._report_counter:06d}",
            time_window=window_str,
            generated_at=datetime.now(UTC),
            total_requests=request_stats.count,
            request_latency_stats=request_stats,
            agent_latency_stats=agent_stats,
            tool_latency_stats=tool_stats,
            llm_latency_stats=llm_stats,
            bottlenecks=bottlenecks,
            alerts=alerts,
            recommendations=recommendations,
        )

        logger.info(
            "performance_report_generated",
            report_id=report.report_id,
            total_requests=request_stats.count,
            bottleneck_count=len(bottlenecks),
            alert_count=len(alerts),
        )

        return report

    def _get_unique_names(self, component: str, cutoff: datetime) -> set[str]:
        """Get unique component names from recent records."""
        return {
            r.name
            for r in self._tracker._latency_records
            if r.component == component and r.timestamp >= cutoff
        }

    def _get_recent_trace_ids(self, cutoff: datetime) -> set[str]:
        """Get trace IDs from recent records."""
        return {r.trace_id for r in self._tracker._latency_records if r.timestamp >= cutoff}

    def _generate_recommendations(
        self,
        request_stats: PercentileStats,
        agent_stats: dict[str, PercentileStats],
        tool_stats: dict[str, PercentileStats],
        bottlenecks: list[Bottleneck],
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations: list[str] = []

        # Check P95 vs target
        if request_stats.p95 > TARGET_END_TO_END_LATENCY_MS:
            recommendations.append(
                f"P95 latency ({request_stats.p95:.0f}ms) exceeds target "
                f"({TARGET_END_TO_END_LATENCY_MS}ms). Consider caching or parallelization."
            )

        # Find slowest agents
        if agent_stats:
            slowest_agent = max(agent_stats.items(), key=lambda x: x[1].p95)
            if slowest_agent[1].p95 > TARGET_AGENT_LATENCY_MS:
                recommendations.append(
                    f"Agent '{slowest_agent[0]}' is the slowest (P95: {slowest_agent[1].p95:.0f}ms). "
                    f"Optimize this agent for the biggest impact."
                )

        # Find slowest tools
        if tool_stats:
            slowest_tool = max(tool_stats.items(), key=lambda x: x[1].p95)
            if slowest_tool[1].p95 > TARGET_TOOL_LATENCY_MS:
                recommendations.append(
                    f"Tool '{slowest_tool[0]}' is the slowest (P95: {slowest_tool[1].p95:.0f}ms). "
                    f"Consider adding caching or optimizing API calls."
                )

        # Common bottleneck patterns
        bottleneck_types = [b.bottleneck_type for b in bottlenecks]
        if bottleneck_types.count(BottleneckType.SLOW_LLM) > 2:
            recommendations.append(
                "Multiple slow LLM calls detected. Consider using a faster model "
                "or reducing prompt sizes."
            )

        if bottleneck_types.count(BottleneckType.QUEUE_DELAY) > 0:
            recommendations.append(
                "Queue delays detected. Consider increasing worker concurrency."
            )

        if not recommendations:
            recommendations.append("Performance is within acceptable targets. No action needed.")

        return recommendations

    @staticmethod
    def _format_time_window(td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h"
        else:
            return f"{total_seconds // 86400}d"


# ============================================================================
# Module-Level Singletons and Functions
# ============================================================================

_performance_tracker: PerformanceTracker | None = None
_bottleneck_detector: BottleneckDetector | None = None
_alert_manager: PerformanceAlertManager | None = None
_report_generator: PerformanceReportGenerator | None = None


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


def get_bottleneck_detector() -> BottleneckDetector:
    """Get the global bottleneck detector instance."""
    global _bottleneck_detector
    if _bottleneck_detector is None:
        _bottleneck_detector = BottleneckDetector(get_performance_tracker())
    return _bottleneck_detector


def get_alert_manager() -> PerformanceAlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = PerformanceAlertManager(get_performance_tracker())
    return _alert_manager


def get_report_generator() -> PerformanceReportGenerator:
    """Get the global report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = PerformanceReportGenerator(
            get_performance_tracker(),
            get_bottleneck_detector(),
            get_alert_manager(),
        )
    return _report_generator


def reset_performance_tracking() -> None:
    """Reset all performance tracking singletons."""
    global _performance_tracker, _bottleneck_detector, _alert_manager, _report_generator
    _performance_tracker = None
    _bottleneck_detector = None
    _alert_manager = None
    _report_generator = None


def identify_bottlenecks(trace_id: str) -> list[Bottleneck]:
    """Identify performance bottlenecks in a request.

    Args:
        trace_id: Trace identifier.

    Returns:
        List of detected bottlenecks.
    """
    return get_bottleneck_detector().identify_bottlenecks(trace_id)


def generate_performance_report(
    time_window: timedelta = timedelta(hours=1),
) -> PerformanceReport:
    """Generate a performance report.

    Args:
        time_window: Time window to cover.

    Returns:
        PerformanceReport with aggregated statistics.
    """
    return get_report_generator().generate_report(time_window)


def format_performance_report(report: PerformanceReport) -> str:
    """Format a performance report as a human-readable string.

    Args:
        report: Performance report to format.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        f"PERFORMANCE REPORT - {report.report_id}",
        "=" * 60,
        "",
        f"Time Window: {report.time_window}",
        f"Generated: {report.generated_at.isoformat()}",
        f"Total Requests: {report.total_requests}",
        "",
        "-" * 40,
        "REQUEST LATENCY",
        "-" * 40,
        f"  Count: {report.request_latency_stats.count}",
        f"  Mean:  {report.request_latency_stats.mean:.2f}ms",
        f"  P50:   {report.request_latency_stats.p50:.2f}ms",
        f"  P90:   {report.request_latency_stats.p90:.2f}ms",
        f"  P95:   {report.request_latency_stats.p95:.2f}ms",
        f"  P99:   {report.request_latency_stats.p99:.2f}ms",
    ]

    if report.agent_latency_stats:
        lines.extend(
            [
                "",
                "-" * 40,
                "AGENT LATENCY",
                "-" * 40,
            ]
        )
        for agent_name, stats in sorted(report.agent_latency_stats.items()):
            lines.append(f"  {agent_name}:")
            lines.append(f"    P50: {stats.p50:.2f}ms | P95: {stats.p95:.2f}ms | P99: {stats.p99:.2f}ms")

    if report.tool_latency_stats:
        lines.extend(
            [
                "",
                "-" * 40,
                "TOOL LATENCY",
                "-" * 40,
            ]
        )
        for tool_name, stats in sorted(report.tool_latency_stats.items()):
            lines.append(f"  {tool_name}:")
            lines.append(f"    P50: {stats.p50:.2f}ms | P95: {stats.p95:.2f}ms | P99: {stats.p99:.2f}ms")

    if report.bottlenecks:
        lines.extend(
            [
                "",
                "-" * 40,
                f"BOTTLENECKS ({len(report.bottlenecks)})",
                "-" * 40,
            ]
        )
        for bn in report.bottlenecks[:5]:  # Show top 5
            lines.append(f"  • {bn.component}: {bn.latency_ms:.0f}ms ({bn.bottleneck_type.value})")
            lines.append(f"    {bn.suggestion}")
        if len(report.bottlenecks) > 5:
            lines.append(f"  ... and {len(report.bottlenecks) - 5} more")

    if report.alerts:
        lines.extend(
            [
                "",
                "-" * 40,
                f"ALERTS ({len(report.alerts)})",
                "-" * 40,
            ]
        )
        for alert in report.alerts[:5]:
            icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(
                alert.severity.value, "⚪"
            )
            lines.append(f"  {icon} [{alert.severity.value.upper()}] {alert.message}")
        if len(report.alerts) > 5:
            lines.append(f"  ... and {len(report.alerts) - 5} more")

    lines.extend(
        [
            "",
            "-" * 40,
            "RECOMMENDATIONS",
            "-" * 40,
        ]
    )
    for rec in report.recommendations:
        lines.append(f"  • {rec}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
