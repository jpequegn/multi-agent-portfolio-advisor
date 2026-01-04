"""Observability dashboard for monitoring the multi-agent system.

This module provides dashboard configurations and metric aggregation
for visualizing system health, performance, and costs.

Dashboard Panels:
- Overview: System health at a glance
- Agent Performance: Per-agent metrics comparison
- Errors: Error tracking and trends
- Cost Analysis: Cost monitoring and attribution

Features:
- Real-time metric collection
- Historical data aggregation
- Configurable refresh intervals
- Export to Langfuse dashboard API
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.observability.costs import (
    CostBreakdown,
    CostReport,
    CostTracker,
    format_cost,
    generate_cost_report,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Metric Definitions
# ============================================================================


class MetricType(Enum):
    """Types of metrics that can be displayed."""

    COUNTER = "counter"  # Cumulative count
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Rate per time unit


class AggregationType(Enum):
    """How to aggregate metrics over time."""

    SUM = "sum"
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class TimeWindow(Enum):
    """Time windows for metric aggregation."""

    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected and displayed.

    Attributes:
        name: Unique metric name.
        display_name: Human-readable name.
        description: Description of what the metric measures.
        metric_type: Type of metric (counter, gauge, etc.).
        unit: Unit of measurement.
        aggregation: How to aggregate over time.
    """

    name: str
    display_name: str
    description: str
    metric_type: MetricType
    unit: str = ""
    aggregation: AggregationType = AggregationType.SUM


# Standard metric definitions for the multi-agent system
STANDARD_METRICS = {
    # Request metrics
    "requests_total": MetricDefinition(
        name="requests_total",
        display_name="Total Requests",
        description="Total number of portfolio analysis requests",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "requests_successful": MetricDefinition(
        name="requests_successful",
        display_name="Successful Requests",
        description="Number of successfully completed requests",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "requests_failed": MetricDefinition(
        name="requests_failed",
        display_name="Failed Requests",
        description="Number of failed requests",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "success_rate": MetricDefinition(
        name="success_rate",
        display_name="Success Rate",
        description="Percentage of successful requests",
        metric_type=MetricType.GAUGE,
        unit="%",
        aggregation=AggregationType.AVERAGE,
    ),
    # Latency metrics
    "latency_avg": MetricDefinition(
        name="latency_avg",
        display_name="Average Latency",
        description="Average request latency",
        metric_type=MetricType.GAUGE,
        unit="ms",
        aggregation=AggregationType.AVERAGE,
    ),
    "latency_p50": MetricDefinition(
        name="latency_p50",
        display_name="P50 Latency",
        description="50th percentile latency",
        metric_type=MetricType.HISTOGRAM,
        unit="ms",
        aggregation=AggregationType.PERCENTILE_50,
    ),
    "latency_p95": MetricDefinition(
        name="latency_p95",
        display_name="P95 Latency",
        description="95th percentile latency",
        metric_type=MetricType.HISTOGRAM,
        unit="ms",
        aggregation=AggregationType.PERCENTILE_95,
    ),
    "latency_p99": MetricDefinition(
        name="latency_p99",
        display_name="P99 Latency",
        description="99th percentile latency",
        metric_type=MetricType.HISTOGRAM,
        unit="ms",
        aggregation=AggregationType.PERCENTILE_99,
    ),
    # Token metrics
    "tokens_total": MetricDefinition(
        name="tokens_total",
        display_name="Total Tokens",
        description="Total tokens consumed",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "tokens_input": MetricDefinition(
        name="tokens_input",
        display_name="Input Tokens",
        description="Total input tokens consumed",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "tokens_output": MetricDefinition(
        name="tokens_output",
        display_name="Output Tokens",
        description="Total output tokens consumed",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    # Cost metrics
    "cost_total": MetricDefinition(
        name="cost_total",
        display_name="Total Cost",
        description="Total cost in dollars",
        metric_type=MetricType.COUNTER,
        unit="$",
        aggregation=AggregationType.SUM,
    ),
    "cost_per_request": MetricDefinition(
        name="cost_per_request",
        display_name="Cost per Request",
        description="Average cost per request",
        metric_type=MetricType.GAUGE,
        unit="$",
        aggregation=AggregationType.AVERAGE,
    ),
    # Agent metrics
    "agent_calls": MetricDefinition(
        name="agent_calls",
        display_name="Agent Calls",
        description="Number of agent executions",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "agent_errors": MetricDefinition(
        name="agent_errors",
        display_name="Agent Errors",
        description="Number of agent errors",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    # Tool metrics
    "tool_calls": MetricDefinition(
        name="tool_calls",
        display_name="Tool Calls",
        description="Number of tool executions",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "tool_cache_hits": MetricDefinition(
        name="tool_cache_hits",
        display_name="Cache Hits",
        description="Number of tool cache hits",
        metric_type=MetricType.COUNTER,
        aggregation=AggregationType.SUM,
    ),
    "cache_hit_rate": MetricDefinition(
        name="cache_hit_rate",
        display_name="Cache Hit Rate",
        description="Percentage of tool calls served from cache",
        metric_type=MetricType.GAUGE,
        unit="%",
        aggregation=AggregationType.AVERAGE,
    ),
}


# ============================================================================
# Metric Data Point
# ============================================================================


@dataclass
class MetricDataPoint:
    """A single data point for a metric.

    Attributes:
        metric_name: Name of the metric.
        value: Numeric value.
        timestamp: When the metric was recorded.
        labels: Additional labels for filtering.
    """

    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


# ============================================================================
# Dashboard Panel Definitions
# ============================================================================


class PanelType(Enum):
    """Types of dashboard panels."""

    STAT = "stat"  # Single stat display
    LINE_CHART = "line_chart"  # Time series line chart
    BAR_CHART = "bar_chart"  # Bar chart
    PIE_CHART = "pie_chart"  # Pie chart
    TABLE = "table"  # Data table
    HEATMAP = "heatmap"  # Heatmap visualization


@dataclass
class DashboardPanel:
    """Configuration for a dashboard panel.

    Attributes:
        id: Unique panel identifier.
        title: Panel title.
        description: Panel description.
        panel_type: Type of visualization.
        metrics: Metrics to display.
        time_window: Time range for data.
        refresh_interval: How often to refresh (seconds).
        position: Grid position (row, col).
        size: Panel size (width, height).
    """

    id: str
    title: str
    description: str
    panel_type: PanelType
    metrics: list[str]
    time_window: TimeWindow = TimeWindow.DAY
    refresh_interval: int = 60
    position: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (4, 3)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API/export."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.panel_type.value,
            "metrics": self.metrics,
            "time_window": self.time_window.value,
            "refresh_interval": self.refresh_interval,
            "position": {"row": self.position[0], "col": self.position[1]},
            "size": {"width": self.size[0], "height": self.size[1]},
        }


# ============================================================================
# Dashboard Configuration
# ============================================================================


@dataclass
class DashboardConfig:
    """Configuration for a complete dashboard.

    Attributes:
        id: Unique dashboard identifier.
        name: Dashboard name.
        description: Dashboard description.
        panels: List of panels.
        default_time_window: Default time range.
        auto_refresh: Whether to auto-refresh.
        refresh_interval: Global refresh interval.
    """

    id: str
    name: str
    description: str
    panels: list[DashboardPanel] = field(default_factory=list)
    default_time_window: TimeWindow = TimeWindow.DAY
    auto_refresh: bool = True
    refresh_interval: int = 60

    def add_panel(self, panel: DashboardPanel) -> None:
        """Add a panel to the dashboard."""
        self.panels.append(panel)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API/export."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "panels": [p.to_dict() for p in self.panels],
            "default_time_window": self.default_time_window.value,
            "auto_refresh": self.auto_refresh,
            "refresh_interval": self.refresh_interval,
        }


# ============================================================================
# Pre-configured Dashboard Panels
# ============================================================================


# Overview Panel
OVERVIEW_PANELS = [
    DashboardPanel(
        id="total_requests_24h",
        title="Requests (24h)",
        description="Total requests in the last 24 hours",
        panel_type=PanelType.STAT,
        metrics=["requests_total"],
        time_window=TimeWindow.DAY,
        position=(0, 0),
        size=(3, 2),
    ),
    DashboardPanel(
        id="success_rate",
        title="Success Rate",
        description="Percentage of successful requests",
        panel_type=PanelType.STAT,
        metrics=["success_rate"],
        time_window=TimeWindow.DAY,
        position=(0, 3),
        size=(3, 2),
    ),
    DashboardPanel(
        id="avg_latency",
        title="Avg Latency",
        description="Average request latency in milliseconds",
        panel_type=PanelType.STAT,
        metrics=["latency_avg"],
        time_window=TimeWindow.DAY,
        position=(0, 6),
        size=(3, 2),
    ),
    DashboardPanel(
        id="total_cost_24h",
        title="Total Cost (24h)",
        description="Total cost in the last 24 hours",
        panel_type=PanelType.STAT,
        metrics=["cost_total"],
        time_window=TimeWindow.DAY,
        position=(0, 9),
        size=(3, 2),
    ),
    DashboardPanel(
        id="request_volume_trend",
        title="Request Volume Over Time",
        description="Number of requests over time",
        panel_type=PanelType.LINE_CHART,
        metrics=["requests_total"],
        time_window=TimeWindow.WEEK,
        position=(2, 0),
        size=(12, 4),
    ),
]

# Agent Performance Panels
AGENT_PERFORMANCE_PANELS = [
    DashboardPanel(
        id="requests_by_agent",
        title="Requests by Agent",
        description="Request distribution across agents",
        panel_type=PanelType.BAR_CHART,
        metrics=["agent_calls"],
        time_window=TimeWindow.DAY,
        position=(0, 0),
        size=(6, 4),
    ),
    DashboardPanel(
        id="success_rate_by_agent",
        title="Success Rate by Agent",
        description="Success rate per agent",
        panel_type=PanelType.BAR_CHART,
        metrics=["success_rate"],
        time_window=TimeWindow.DAY,
        position=(0, 6),
        size=(6, 4),
    ),
    DashboardPanel(
        id="latency_by_agent",
        title="Latency by Agent",
        description="Average latency per agent",
        panel_type=PanelType.BAR_CHART,
        metrics=["latency_avg"],
        time_window=TimeWindow.DAY,
        position=(4, 0),
        size=(6, 4),
    ),
    DashboardPanel(
        id="cost_by_agent",
        title="Cost by Agent",
        description="Cost distribution across agents",
        panel_type=PanelType.PIE_CHART,
        metrics=["cost_total"],
        time_window=TimeWindow.DAY,
        position=(4, 6),
        size=(6, 4),
    ),
]

# Error Panels
ERROR_PANELS = [
    DashboardPanel(
        id="error_count",
        title="Error Count",
        description="Total errors in the period",
        panel_type=PanelType.STAT,
        metrics=["requests_failed"],
        time_window=TimeWindow.DAY,
        position=(0, 0),
        size=(4, 2),
    ),
    DashboardPanel(
        id="error_rate",
        title="Error Rate",
        description="Percentage of failed requests",
        panel_type=PanelType.STAT,
        metrics=["requests_failed"],
        time_window=TimeWindow.DAY,
        position=(0, 4),
        size=(4, 2),
    ),
    DashboardPanel(
        id="agent_errors",
        title="Agent Errors",
        description="Errors by agent",
        panel_type=PanelType.BAR_CHART,
        metrics=["agent_errors"],
        time_window=TimeWindow.DAY,
        position=(0, 8),
        size=(4, 2),
    ),
    DashboardPanel(
        id="error_trend",
        title="Error Trend",
        description="Errors over time",
        panel_type=PanelType.LINE_CHART,
        metrics=["requests_failed"],
        time_window=TimeWindow.WEEK,
        position=(2, 0),
        size=(12, 4),
    ),
    DashboardPanel(
        id="recent_failures",
        title="Recent Failures",
        description="Table of recent failed requests with trace links",
        panel_type=PanelType.TABLE,
        metrics=["requests_failed"],
        time_window=TimeWindow.DAY,
        position=(6, 0),
        size=(12, 4),
    ),
]

# Cost Analysis Panels
COST_PANELS = [
    DashboardPanel(
        id="cost_trend",
        title="Cost Trend",
        description="Cost over time",
        panel_type=PanelType.LINE_CHART,
        metrics=["cost_total"],
        time_window=TimeWindow.WEEK,
        position=(0, 0),
        size=(12, 4),
    ),
    DashboardPanel(
        id="cost_by_agent_chart",
        title="Cost by Agent",
        description="Cost breakdown by agent",
        panel_type=PanelType.PIE_CHART,
        metrics=["cost_total"],
        time_window=TimeWindow.DAY,
        position=(4, 0),
        size=(6, 4),
    ),
    DashboardPanel(
        id="cost_by_model",
        title="Cost by Model",
        description="Cost breakdown by model",
        panel_type=PanelType.PIE_CHART,
        metrics=["cost_total"],
        time_window=TimeWindow.DAY,
        position=(4, 6),
        size=(6, 4),
    ),
    DashboardPanel(
        id="cost_per_request_trend",
        title="Cost per Request",
        description="Average cost per request over time",
        panel_type=PanelType.LINE_CHART,
        metrics=["cost_per_request"],
        time_window=TimeWindow.WEEK,
        position=(8, 0),
        size=(12, 4),
    ),
]


# ============================================================================
# Metric Collector
# ============================================================================


class MetricCollector:
    """Collects and aggregates metrics for dashboard display.

    Provides methods for recording metrics and querying aggregated data.
    """

    def __init__(self) -> None:
        """Initialize the metric collector."""
        self._data_points: list[MetricDataPoint] = []
        self._request_stats: dict[str, Any] = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "latencies": [],
        }
        self._agent_stats: dict[str, dict[str, Any]] = {}
        self._tool_stats: dict[str, dict[str, Any]] = {}

    def record_request(
        self,
        *,
        success: bool,
        latency_ms: float,
        trace_id: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a request completion.

        Args:
            success: Whether the request succeeded.
            latency_ms: Request latency in milliseconds.
            trace_id: Optional trace identifier.
            labels: Additional labels.
        """
        self._request_stats["total"] += 1
        if success:
            self._request_stats["successful"] += 1
        else:
            self._request_stats["failed"] += 1
        self._request_stats["latencies"].append(latency_ms)

        # Record individual data points
        self._data_points.append(
            MetricDataPoint(
                metric_name="requests_total",
                value=1,
                labels=labels or {},
            )
        )
        self._data_points.append(
            MetricDataPoint(
                metric_name="requests_successful" if success else "requests_failed",
                value=1,
                labels=labels or {},
            )
        )
        self._data_points.append(
            MetricDataPoint(
                metric_name="latency_avg",
                value=latency_ms,
                labels=labels or {},
            )
        )

        logger.debug(
            "request_recorded",
            success=success,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

    def record_agent_execution(
        self,
        agent_name: str,
        *,
        success: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record an agent execution.

        Args:
            agent_name: Name of the agent.
            success: Whether execution succeeded.
            latency_ms: Execution latency in milliseconds.
            error: Optional error message.
        """
        if agent_name not in self._agent_stats:
            self._agent_stats[agent_name] = {
                "calls": 0,
                "successful": 0,
                "failed": 0,
                "latencies": [],
                "errors": [],
            }

        stats = self._agent_stats[agent_name]
        stats["calls"] += 1
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1
            if error:
                stats["errors"].append(error)
        stats["latencies"].append(latency_ms)

        self._data_points.append(
            MetricDataPoint(
                metric_name="agent_calls",
                value=1,
                labels={"agent": agent_name},
            )
        )
        if not success:
            self._data_points.append(
                MetricDataPoint(
                    metric_name="agent_errors",
                    value=1,
                    labels={"agent": agent_name},
                )
            )

    def record_tool_execution(
        self,
        tool_name: str,
        *,
        success: bool,
        latency_ms: float,
        cache_hit: bool = False,
    ) -> None:
        """Record a tool execution.

        Args:
            tool_name: Name of the tool.
            success: Whether execution succeeded.
            latency_ms: Execution latency in milliseconds.
            cache_hit: Whether the result was from cache.
        """
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "successful": 0,
                "failed": 0,
                "cache_hits": 0,
                "latencies": [],
            }

        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1
        if cache_hit:
            stats["cache_hits"] += 1
        stats["latencies"].append(latency_ms)

        self._data_points.append(
            MetricDataPoint(
                metric_name="tool_calls",
                value=1,
                labels={"tool": tool_name},
            )
        )
        if cache_hit:
            self._data_points.append(
                MetricDataPoint(
                    metric_name="tool_cache_hits",
                    value=1,
                    labels={"tool": tool_name},
                )
            )

    def get_success_rate(self) -> float:
        """Get the overall success rate.

        Returns:
            Success rate as a percentage.
        """
        total = self._request_stats["total"]
        if total == 0:
            return 0.0
        return (self._request_stats["successful"] / total) * 100

    def get_average_latency(self) -> float:
        """Get the average request latency.

        Returns:
            Average latency in milliseconds.
        """
        latencies = self._request_stats["latencies"]
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def get_percentile_latency(self, percentile: float) -> float:
        """Get a percentile latency value.

        Args:
            percentile: Percentile (0-100).

        Returns:
            Latency at the given percentile.
        """
        latencies = sorted(self._request_stats["latencies"])
        if not latencies:
            return 0.0
        index = int(len(latencies) * percentile / 100)
        index = min(index, len(latencies) - 1)
        return latencies[index]

    def get_agent_stats(self, agent_name: str) -> dict[str, Any]:
        """Get statistics for a specific agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Agent statistics dictionary.
        """
        stats = self._agent_stats.get(agent_name, {})
        if not stats:
            return {}

        latencies = stats.get("latencies", [])
        return {
            "calls": stats.get("calls", 0),
            "successful": stats.get("successful", 0),
            "failed": stats.get("failed", 0),
            "success_rate": (
                (stats["successful"] / stats["calls"] * 100)
                if stats.get("calls", 0) > 0
                else 0.0
            ),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "error_count": len(stats.get("errors", [])),
        }

    def get_all_agent_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all agents.

        Returns:
            Dictionary of agent name to statistics.
        """
        return {name: self.get_agent_stats(name) for name in self._agent_stats}

    def get_tool_stats(self, tool_name: str) -> dict[str, Any]:
        """Get statistics for a specific tool.

        Args:
            tool_name: Name of the tool.

        Returns:
            Tool statistics dictionary.
        """
        stats = self._tool_stats.get(tool_name, {})
        if not stats:
            return {}

        latencies = stats.get("latencies", [])
        calls = stats.get("calls", 0)
        return {
            "calls": calls,
            "successful": stats.get("successful", 0),
            "failed": stats.get("failed", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "cache_hit_rate": (stats["cache_hits"] / calls * 100) if calls > 0 else 0.0,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        }

    def get_data_points(
        self,
        metric_name: str,
        *,
        since: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[MetricDataPoint]:
        """Get data points for a metric.

        Args:
            metric_name: Name of the metric.
            since: Only return points after this time.
            labels: Filter by labels.

        Returns:
            List of matching data points.
        """
        points = [p for p in self._data_points if p.metric_name == metric_name]

        if since:
            points = [p for p in points if p.timestamp >= since]

        if labels:
            points = [
                p
                for p in points
                if all(p.labels.get(k) == v for k, v in labels.items())
            ]

        return points

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._data_points.clear()
        self._request_stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "latencies": [],
        }
        self._agent_stats.clear()
        self._tool_stats.clear()


# ============================================================================
# Dashboard Snapshot
# ============================================================================


@dataclass
class DashboardSnapshot:
    """A point-in-time snapshot of dashboard data.

    Attributes:
        timestamp: When the snapshot was taken.
        time_window: Time window for the data.
        overview: Overview panel data.
        agent_performance: Agent performance data.
        errors: Error panel data.
        cost_analysis: Cost analysis data.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    time_window: TimeWindow = TimeWindow.DAY
    overview: dict[str, Any] = field(default_factory=dict)
    agent_performance: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    cost_analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "time_window": self.time_window.value,
            "overview": self.overview,
            "agent_performance": self.agent_performance,
            "errors": self.errors,
            "cost_analysis": self.cost_analysis,
        }


# ============================================================================
# Dashboard Service
# ============================================================================


class DashboardService:
    """Service for generating dashboard data.

    Combines metrics from various sources to produce dashboard snapshots.
    """

    def __init__(
        self,
        metric_collector: MetricCollector | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize the dashboard service.

        Args:
            metric_collector: Metric collector instance.
            cost_tracker: Cost tracker instance.
        """
        self.metrics = metric_collector or MetricCollector()
        self.cost_tracker = cost_tracker

    def get_overview_data(
        self,
        time_window: TimeWindow = TimeWindow.DAY,
    ) -> dict[str, Any]:
        """Get overview panel data.

        Args:
            time_window: Time window for data.

        Returns:
            Overview data dictionary.
        """
        cost_report = self._get_cost_report(time_window)

        return {
            "requests_24h": self.metrics._request_stats["total"],
            "success_rate": self.metrics.get_success_rate(),
            "avg_latency_ms": self.metrics.get_average_latency(),
            "p50_latency_ms": self.metrics.get_percentile_latency(50),
            "p95_latency_ms": self.metrics.get_percentile_latency(95),
            "p99_latency_ms": self.metrics.get_percentile_latency(99),
            "total_cost": cost_report.total_cost if cost_report else 0.0,
            "total_cost_formatted": (
                format_cost(cost_report.total_cost) if cost_report else "$0.00"
            ),
            "total_tokens": cost_report.total_tokens if cost_report else 0,
        }

    def get_agent_performance_data(
        self,
        time_window: TimeWindow = TimeWindow.DAY,
    ) -> dict[str, Any]:
        """Get agent performance panel data.

        Args:
            time_window: Time window for data.

        Returns:
            Agent performance data dictionary.
        """
        agent_stats = self.metrics.get_all_agent_stats()
        cost_report = self._get_cost_report(time_window)

        # Merge cost data if available
        if cost_report:
            for agent_name, cost in cost_report.by_agent.items():
                if agent_name in agent_stats:
                    agent_stats[agent_name]["cost"] = cost
                    agent_stats[agent_name]["cost_formatted"] = format_cost(cost)

        return {
            "agents": agent_stats,
            "total_agent_calls": sum(s.get("calls", 0) for s in agent_stats.values()),
            "agents_with_errors": [
                name for name, stats in agent_stats.items() if stats.get("failed", 0) > 0
            ],
        }

    def get_error_data(
        self,
        time_window: TimeWindow = TimeWindow.DAY,
    ) -> dict[str, Any]:
        """Get error panel data.

        Args:
            time_window: Time window for data.

        Returns:
            Error data dictionary.
        """
        total = self.metrics._request_stats["total"]
        failed = self.metrics._request_stats["failed"]

        # Collect errors by agent
        errors_by_agent: dict[str, int] = {}
        for agent_name, stats in self.metrics._agent_stats.items():
            error_count = stats.get("failed", 0)
            if error_count > 0:
                errors_by_agent[agent_name] = error_count

        return {
            "error_count": failed,
            "error_rate": (failed / total * 100) if total > 0 else 0.0,
            "errors_by_agent": errors_by_agent,
            "recent_errors": [],  # Would be populated from trace storage
        }

    def get_cost_analysis_data(
        self,
        time_window: TimeWindow = TimeWindow.DAY,
    ) -> dict[str, Any]:
        """Get cost analysis panel data.

        Args:
            time_window: Time window for data.

        Returns:
            Cost analysis data dictionary.
        """
        cost_report = self._get_cost_report(time_window)

        if not cost_report:
            return {
                "total_cost": 0.0,
                "total_cost_formatted": "$0.00",
                "cost_per_request": 0.0,
                "by_agent": {},
                "by_model": {},
                "by_day": {},
            }

        return {
            "total_cost": cost_report.total_cost,
            "total_cost_formatted": format_cost(cost_report.total_cost),
            "total_requests": cost_report.total_requests,
            "cost_per_request": cost_report.average_cost_per_request,
            "cost_per_request_formatted": format_cost(
                cost_report.average_cost_per_request
            ),
            "by_agent": {
                agent: {"cost": cost, "formatted": format_cost(cost)}
                for agent, cost in cost_report.by_agent.items()
            },
            "by_model": {
                model: {"cost": cost, "formatted": format_cost(cost)}
                for model, cost in cost_report.by_model.items()
            },
            "by_day": cost_report.by_day,
            "alerts_count": cost_report.alerts_count,
        }

    def get_snapshot(
        self,
        time_window: TimeWindow = TimeWindow.DAY,
    ) -> DashboardSnapshot:
        """Get a complete dashboard snapshot.

        Args:
            time_window: Time window for data.

        Returns:
            Dashboard snapshot with all panel data.
        """
        return DashboardSnapshot(
            time_window=time_window,
            overview=self.get_overview_data(time_window),
            agent_performance=self.get_agent_performance_data(time_window),
            errors=self.get_error_data(time_window),
            cost_analysis=self.get_cost_analysis_data(time_window),
        )

    def _get_cost_report(
        self,
        time_window: TimeWindow,
    ) -> CostReport | None:
        """Get cost report for the time window.

        Args:
            time_window: Time window for data.

        Returns:
            Cost report or None if no cost tracker.
        """
        if not self.cost_tracker:
            return None

        breakdowns = self.cost_tracker.completed_breakdowns

        # Filter by time window
        now = datetime.now(UTC)
        window_delta = {
            TimeWindow.HOUR: timedelta(hours=1),
            TimeWindow.DAY: timedelta(days=1),
            TimeWindow.WEEK: timedelta(days=7),
            TimeWindow.MONTH: timedelta(days=30),
        }
        cutoff = now - window_delta[time_window]

        filtered = [b for b in breakdowns if b.started_at >= cutoff]

        return generate_cost_report(
            filtered,
            period_start=cutoff,
            period_end=now,
            alerts=self.cost_tracker.alert_manager.alerts,
        )


# ============================================================================
# Dashboard Factory
# ============================================================================


def create_overview_dashboard() -> DashboardConfig:
    """Create the overview dashboard configuration.

    Returns:
        Configured overview dashboard.
    """
    dashboard = DashboardConfig(
        id="overview",
        name="System Overview",
        description="At-a-glance view of system health and performance",
    )
    for panel in OVERVIEW_PANELS:
        dashboard.add_panel(panel)
    return dashboard


def create_agent_dashboard() -> DashboardConfig:
    """Create the agent performance dashboard configuration.

    Returns:
        Configured agent performance dashboard.
    """
    dashboard = DashboardConfig(
        id="agent_performance",
        name="Agent Performance",
        description="Detailed agent execution metrics and comparisons",
    )
    for panel in AGENT_PERFORMANCE_PANELS:
        dashboard.add_panel(panel)
    return dashboard


def create_error_dashboard() -> DashboardConfig:
    """Create the error analysis dashboard configuration.

    Returns:
        Configured error analysis dashboard.
    """
    dashboard = DashboardConfig(
        id="errors",
        name="Error Analysis",
        description="Error tracking, trends, and recent failures",
    )
    for panel in ERROR_PANELS:
        dashboard.add_panel(panel)
    return dashboard


def create_cost_dashboard() -> DashboardConfig:
    """Create the cost analysis dashboard configuration.

    Returns:
        Configured cost analysis dashboard.
    """
    dashboard = DashboardConfig(
        id="cost_analysis",
        name="Cost Analysis",
        description="Cost monitoring, attribution, and trends",
    )
    for panel in COST_PANELS:
        dashboard.add_panel(panel)
    return dashboard


def create_all_dashboards() -> list[DashboardConfig]:
    """Create all standard dashboards.

    Returns:
        List of all dashboard configurations.
    """
    return [
        create_overview_dashboard(),
        create_agent_dashboard(),
        create_error_dashboard(),
        create_cost_dashboard(),
    ]


# ============================================================================
# Global Instance
# ============================================================================

_global_collector: MetricCollector | None = None
_global_service: DashboardService | None = None


def get_metric_collector() -> MetricCollector:
    """Get the global metric collector instance.

    Returns:
        Global MetricCollector instance.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricCollector()
    return _global_collector


def get_dashboard_service(
    cost_tracker: CostTracker | None = None,
) -> DashboardService:
    """Get the global dashboard service instance.

    Args:
        cost_tracker: Optional cost tracker to use.

    Returns:
        Global DashboardService instance.
    """
    global _global_service
    if _global_service is None:
        _global_service = DashboardService(
            metric_collector=get_metric_collector(),
            cost_tracker=cost_tracker,
        )
    return _global_service


def reset_dashboard_service() -> None:
    """Reset the global dashboard service and collector."""
    global _global_collector, _global_service
    _global_collector = None
    _global_service = None
