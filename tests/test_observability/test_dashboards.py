"""Tests for the observability dashboard module."""

from datetime import UTC, datetime, timedelta

import pytest

from src.observability.costs import CostBreakdown, CostTracker, TokenUsage
from src.observability.dashboards import (
    AGENT_PERFORMANCE_PANELS,
    COST_PANELS,
    ERROR_PANELS,
    OVERVIEW_PANELS,
    STANDARD_METRICS,
    AggregationType,
    DashboardConfig,
    DashboardPanel,
    DashboardService,
    DashboardSnapshot,
    MetricCollector,
    MetricDataPoint,
    MetricDefinition,
    MetricType,
    PanelType,
    TimeWindow,
    create_agent_dashboard,
    create_all_dashboards,
    create_cost_dashboard,
    create_error_dashboard,
    create_overview_dashboard,
    get_dashboard_service,
    get_metric_collector,
    reset_dashboard_service,
)


# ============================================================================
# MetricType and AggregationType Tests
# ============================================================================


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types_exist(self) -> None:
        """Test all metric types are defined."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.RATE.value == "rate"


class TestAggregationType:
    """Tests for AggregationType enum."""

    def test_aggregation_types_exist(self) -> None:
        """Test all aggregation types are defined."""
        assert AggregationType.SUM.value == "sum"
        assert AggregationType.AVERAGE.value == "avg"
        assert AggregationType.MIN.value == "min"
        assert AggregationType.MAX.value == "max"
        assert AggregationType.COUNT.value == "count"
        assert AggregationType.PERCENTILE_50.value == "p50"
        assert AggregationType.PERCENTILE_90.value == "p90"
        assert AggregationType.PERCENTILE_95.value == "p95"
        assert AggregationType.PERCENTILE_99.value == "p99"


class TestTimeWindow:
    """Tests for TimeWindow enum."""

    def test_time_windows_exist(self) -> None:
        """Test all time windows are defined."""
        assert TimeWindow.HOUR.value == "1h"
        assert TimeWindow.DAY.value == "24h"
        assert TimeWindow.WEEK.value == "7d"
        assert TimeWindow.MONTH.value == "30d"


# ============================================================================
# MetricDefinition Tests
# ============================================================================


class TestMetricDefinition:
    """Tests for MetricDefinition dataclass."""

    def test_create_metric_definition(self) -> None:
        """Test creating a metric definition."""
        metric = MetricDefinition(
            name="test_metric",
            display_name="Test Metric",
            description="A test metric",
            metric_type=MetricType.COUNTER,
            unit="count",
            aggregation=AggregationType.SUM,
        )

        assert metric.name == "test_metric"
        assert metric.display_name == "Test Metric"
        assert metric.description == "A test metric"
        assert metric.metric_type == MetricType.COUNTER
        assert metric.unit == "count"
        assert metric.aggregation == AggregationType.SUM

    def test_default_values(self) -> None:
        """Test default values for metric definition."""
        metric = MetricDefinition(
            name="test",
            display_name="Test",
            description="Test",
            metric_type=MetricType.GAUGE,
        )

        assert metric.unit == ""
        assert metric.aggregation == AggregationType.SUM


class TestStandardMetrics:
    """Tests for standard metric definitions."""

    def test_standard_metrics_defined(self) -> None:
        """Test standard metrics are defined."""
        assert "requests_total" in STANDARD_METRICS
        assert "success_rate" in STANDARD_METRICS
        assert "latency_avg" in STANDARD_METRICS
        assert "cost_total" in STANDARD_METRICS
        assert "agent_calls" in STANDARD_METRICS
        assert "tool_calls" in STANDARD_METRICS

    def test_request_metrics(self) -> None:
        """Test request-related metrics."""
        total = STANDARD_METRICS["requests_total"]
        assert total.metric_type == MetricType.COUNTER
        assert total.aggregation == AggregationType.SUM

        success_rate = STANDARD_METRICS["success_rate"]
        assert success_rate.metric_type == MetricType.GAUGE
        assert success_rate.unit == "%"

    def test_latency_metrics(self) -> None:
        """Test latency-related metrics."""
        avg = STANDARD_METRICS["latency_avg"]
        assert avg.unit == "ms"

        p95 = STANDARD_METRICS["latency_p95"]
        assert p95.metric_type == MetricType.HISTOGRAM
        assert p95.aggregation == AggregationType.PERCENTILE_95


# ============================================================================
# MetricDataPoint Tests
# ============================================================================


class TestMetricDataPoint:
    """Tests for MetricDataPoint dataclass."""

    def test_create_data_point(self) -> None:
        """Test creating a metric data point."""
        point = MetricDataPoint(
            metric_name="requests_total",
            value=42.0,
            labels={"agent": "research"},
        )

        assert point.metric_name == "requests_total"
        assert point.value == 42.0
        assert point.labels == {"agent": "research"}
        assert point.timestamp is not None

    def test_to_dict(self) -> None:
        """Test converting data point to dictionary."""
        point = MetricDataPoint(
            metric_name="test",
            value=10.5,
            labels={"key": "value"},
        )

        d = point.to_dict()
        assert d["metric"] == "test"
        assert d["value"] == 10.5
        assert d["labels"] == {"key": "value"}
        assert "timestamp" in d


# ============================================================================
# DashboardPanel Tests
# ============================================================================


class TestDashboardPanel:
    """Tests for DashboardPanel dataclass."""

    def test_create_panel(self) -> None:
        """Test creating a dashboard panel."""
        panel = DashboardPanel(
            id="test_panel",
            title="Test Panel",
            description="A test panel",
            panel_type=PanelType.STAT,
            metrics=["requests_total"],
            time_window=TimeWindow.DAY,
            position=(0, 0),
            size=(4, 3),
        )

        assert panel.id == "test_panel"
        assert panel.title == "Test Panel"
        assert panel.panel_type == PanelType.STAT
        assert panel.metrics == ["requests_total"]
        assert panel.time_window == TimeWindow.DAY

    def test_panel_to_dict(self) -> None:
        """Test converting panel to dictionary."""
        panel = DashboardPanel(
            id="test",
            title="Test",
            description="Test",
            panel_type=PanelType.LINE_CHART,
            metrics=["latency_avg"],
            position=(2, 4),
            size=(6, 4),
        )

        d = panel.to_dict()
        assert d["id"] == "test"
        assert d["type"] == "line_chart"
        assert d["position"] == {"row": 2, "col": 4}
        assert d["size"] == {"width": 6, "height": 4}


class TestPanelType:
    """Tests for PanelType enum."""

    def test_panel_types_exist(self) -> None:
        """Test all panel types are defined."""
        assert PanelType.STAT.value == "stat"
        assert PanelType.LINE_CHART.value == "line_chart"
        assert PanelType.BAR_CHART.value == "bar_chart"
        assert PanelType.PIE_CHART.value == "pie_chart"
        assert PanelType.TABLE.value == "table"
        assert PanelType.HEATMAP.value == "heatmap"


# ============================================================================
# DashboardConfig Tests
# ============================================================================


class TestDashboardConfig:
    """Tests for DashboardConfig dataclass."""

    def test_create_dashboard(self) -> None:
        """Test creating a dashboard configuration."""
        dashboard = DashboardConfig(
            id="test_dashboard",
            name="Test Dashboard",
            description="A test dashboard",
        )

        assert dashboard.id == "test_dashboard"
        assert dashboard.name == "Test Dashboard"
        assert dashboard.panels == []
        assert dashboard.auto_refresh is True
        assert dashboard.refresh_interval == 60

    def test_add_panel(self) -> None:
        """Test adding panels to dashboard."""
        dashboard = DashboardConfig(
            id="test",
            name="Test",
            description="Test",
        )

        panel = DashboardPanel(
            id="panel1",
            title="Panel 1",
            description="Test panel",
            panel_type=PanelType.STAT,
            metrics=["requests_total"],
        )

        dashboard.add_panel(panel)
        assert len(dashboard.panels) == 1
        assert dashboard.panels[0].id == "panel1"

    def test_dashboard_to_dict(self) -> None:
        """Test converting dashboard to dictionary."""
        dashboard = DashboardConfig(
            id="test",
            name="Test",
            description="Test",
            default_time_window=TimeWindow.WEEK,
        )

        d = dashboard.to_dict()
        assert d["id"] == "test"
        assert d["name"] == "Test"
        assert d["default_time_window"] == "7d"
        assert d["panels"] == []


# ============================================================================
# Pre-configured Panels Tests
# ============================================================================


class TestPreconfiguredPanels:
    """Tests for pre-configured dashboard panels."""

    def test_overview_panels_defined(self) -> None:
        """Test overview panels are defined."""
        assert len(OVERVIEW_PANELS) > 0

        # Check for expected panels
        panel_ids = [p.id for p in OVERVIEW_PANELS]
        assert "total_requests_24h" in panel_ids
        assert "success_rate" in panel_ids
        assert "avg_latency" in panel_ids
        assert "total_cost_24h" in panel_ids

    def test_agent_performance_panels_defined(self) -> None:
        """Test agent performance panels are defined."""
        assert len(AGENT_PERFORMANCE_PANELS) > 0

        panel_ids = [p.id for p in AGENT_PERFORMANCE_PANELS]
        assert "requests_by_agent" in panel_ids
        assert "success_rate_by_agent" in panel_ids
        assert "cost_by_agent" in panel_ids

    def test_error_panels_defined(self) -> None:
        """Test error panels are defined."""
        assert len(ERROR_PANELS) > 0

        panel_ids = [p.id for p in ERROR_PANELS]
        assert "error_count" in panel_ids
        assert "error_trend" in panel_ids
        assert "recent_failures" in panel_ids

    def test_cost_panels_defined(self) -> None:
        """Test cost analysis panels are defined."""
        assert len(COST_PANELS) > 0

        panel_ids = [p.id for p in COST_PANELS]
        assert "cost_trend" in panel_ids
        assert "cost_by_agent_chart" in panel_ids
        assert "cost_by_model" in panel_ids


# ============================================================================
# MetricCollector Tests
# ============================================================================


class TestMetricCollector:
    """Tests for MetricCollector class."""

    def test_create_collector(self) -> None:
        """Test creating a metric collector."""
        collector = MetricCollector()
        assert collector._request_stats["total"] == 0
        assert collector._agent_stats == {}
        assert collector._tool_stats == {}

    def test_record_successful_request(self) -> None:
        """Test recording a successful request."""
        collector = MetricCollector()

        collector.record_request(
            success=True,
            latency_ms=100.0,
            trace_id="trace-1",
        )

        assert collector._request_stats["total"] == 1
        assert collector._request_stats["successful"] == 1
        assert collector._request_stats["failed"] == 0
        assert 100.0 in collector._request_stats["latencies"]

    def test_record_failed_request(self) -> None:
        """Test recording a failed request."""
        collector = MetricCollector()

        collector.record_request(
            success=False,
            latency_ms=50.0,
        )

        assert collector._request_stats["total"] == 1
        assert collector._request_stats["successful"] == 0
        assert collector._request_stats["failed"] == 1

    def test_record_agent_execution(self) -> None:
        """Test recording agent execution."""
        collector = MetricCollector()

        collector.record_agent_execution(
            "research_agent",
            success=True,
            latency_ms=200.0,
        )

        assert "research_agent" in collector._agent_stats
        stats = collector._agent_stats["research_agent"]
        assert stats["calls"] == 1
        assert stats["successful"] == 1
        assert stats["failed"] == 0

    def test_record_agent_execution_with_error(self) -> None:
        """Test recording agent execution with error."""
        collector = MetricCollector()

        collector.record_agent_execution(
            "analysis_agent",
            success=False,
            latency_ms=50.0,
            error="API error",
        )

        stats = collector._agent_stats["analysis_agent"]
        assert stats["failed"] == 1
        assert "API error" in stats["errors"]

    def test_record_tool_execution(self) -> None:
        """Test recording tool execution."""
        collector = MetricCollector()

        collector.record_tool_execution(
            "get_market_data",
            success=True,
            latency_ms=150.0,
            cache_hit=False,
        )

        assert "get_market_data" in collector._tool_stats
        stats = collector._tool_stats["get_market_data"]
        assert stats["calls"] == 1
        assert stats["cache_hits"] == 0

    def test_record_tool_execution_cache_hit(self) -> None:
        """Test recording tool execution with cache hit."""
        collector = MetricCollector()

        collector.record_tool_execution(
            "get_market_data",
            success=True,
            latency_ms=5.0,
            cache_hit=True,
        )

        stats = collector._tool_stats["get_market_data"]
        assert stats["cache_hits"] == 1

    def test_get_success_rate(self) -> None:
        """Test calculating success rate."""
        collector = MetricCollector()

        # No requests yet
        assert collector.get_success_rate() == 0.0

        # 3 successes, 1 failure = 75%
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=False, latency_ms=100)

        assert collector.get_success_rate() == 75.0

    def test_get_average_latency(self) -> None:
        """Test calculating average latency."""
        collector = MetricCollector()

        # No requests yet
        assert collector.get_average_latency() == 0.0

        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=True, latency_ms=200)
        collector.record_request(success=True, latency_ms=300)

        assert collector.get_average_latency() == 200.0

    def test_get_percentile_latency(self) -> None:
        """Test calculating percentile latency."""
        collector = MetricCollector()

        # No requests yet
        assert collector.get_percentile_latency(50) == 0.0

        # Add 10 requests with increasing latencies
        for i in range(1, 11):
            collector.record_request(success=True, latency_ms=i * 10.0)

        # P50 should be around 50-60ms
        p50 = collector.get_percentile_latency(50)
        assert 50 <= p50 <= 60

        # P90 should be around 90-100ms
        p90 = collector.get_percentile_latency(90)
        assert 90 <= p90 <= 100

    def test_get_agent_stats(self) -> None:
        """Test getting agent statistics."""
        collector = MetricCollector()

        # No stats yet
        assert collector.get_agent_stats("unknown") == {}

        # Add some executions
        collector.record_agent_execution("test_agent", success=True, latency_ms=100)
        collector.record_agent_execution("test_agent", success=True, latency_ms=200)
        collector.record_agent_execution("test_agent", success=False, latency_ms=50)

        stats = collector.get_agent_stats("test_agent")
        assert stats["calls"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1
        assert stats["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert stats["avg_latency"] == pytest.approx(116.67, rel=0.1)

    def test_get_all_agent_stats(self) -> None:
        """Test getting all agent statistics."""
        collector = MetricCollector()

        collector.record_agent_execution("agent1", success=True, latency_ms=100)
        collector.record_agent_execution("agent2", success=True, latency_ms=150)

        all_stats = collector.get_all_agent_stats()
        assert "agent1" in all_stats
        assert "agent2" in all_stats

    def test_get_tool_stats(self) -> None:
        """Test getting tool statistics."""
        collector = MetricCollector()

        # No stats yet
        assert collector.get_tool_stats("unknown") == {}

        collector.record_tool_execution("tool1", success=True, latency_ms=100, cache_hit=True)
        collector.record_tool_execution("tool1", success=True, latency_ms=50, cache_hit=True)
        collector.record_tool_execution("tool1", success=True, latency_ms=200, cache_hit=False)

        stats = collector.get_tool_stats("tool1")
        assert stats["calls"] == 3
        assert stats["cache_hits"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(66.67, rel=0.1)

    def test_get_data_points(self) -> None:
        """Test getting data points with filters."""
        collector = MetricCollector()

        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=False, latency_ms=50)

        # Get all request points
        points = collector.get_data_points("requests_total")
        assert len(points) == 2

        # Get only successful
        success_points = collector.get_data_points("requests_successful")
        assert len(success_points) == 1

    def test_get_data_points_with_time_filter(self) -> None:
        """Test getting data points with time filter."""
        collector = MetricCollector()

        collector.record_request(success=True, latency_ms=100)

        # Get points since an hour ago (should find the point)
        since = datetime.now(UTC) - timedelta(hours=1)
        points = collector.get_data_points("requests_total", since=since)
        assert len(points) == 1

        # Get points since the future (should find nothing)
        future = datetime.now(UTC) + timedelta(hours=1)
        points = collector.get_data_points("requests_total", since=future)
        assert len(points) == 0

    def test_clear_collector(self) -> None:
        """Test clearing the collector."""
        collector = MetricCollector()

        collector.record_request(success=True, latency_ms=100)
        collector.record_agent_execution("agent1", success=True, latency_ms=100)
        collector.record_tool_execution("tool1", success=True, latency_ms=100)

        collector.clear()

        assert collector._request_stats["total"] == 0
        assert collector._agent_stats == {}
        assert collector._tool_stats == {}
        assert collector._data_points == []


# ============================================================================
# DashboardSnapshot Tests
# ============================================================================


class TestDashboardSnapshot:
    """Tests for DashboardSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        """Test creating a dashboard snapshot."""
        snapshot = DashboardSnapshot(
            time_window=TimeWindow.DAY,
            overview={"requests_24h": 100},
            agent_performance={"agents": {}},
            errors={"error_count": 5},
            cost_analysis={"total_cost": 10.0},
        )

        assert snapshot.time_window == TimeWindow.DAY
        assert snapshot.overview["requests_24h"] == 100
        assert snapshot.errors["error_count"] == 5

    def test_snapshot_to_dict(self) -> None:
        """Test converting snapshot to dictionary."""
        snapshot = DashboardSnapshot(
            overview={"test": 1},
        )

        d = snapshot.to_dict()
        assert "timestamp" in d
        assert d["overview"] == {"test": 1}
        assert d["time_window"] == "24h"


# ============================================================================
# DashboardService Tests
# ============================================================================


class TestDashboardService:
    """Tests for DashboardService class."""

    def test_create_service(self) -> None:
        """Test creating a dashboard service."""
        service = DashboardService()
        assert service.metrics is not None
        assert service.cost_tracker is None

    def test_create_service_with_cost_tracker(self) -> None:
        """Test creating service with cost tracker."""
        tracker = CostTracker()
        service = DashboardService(cost_tracker=tracker)
        assert service.cost_tracker is tracker

    def test_get_overview_data_empty(self) -> None:
        """Test getting overview data with no metrics."""
        service = DashboardService()

        overview = service.get_overview_data()

        assert overview["requests_24h"] == 0
        assert overview["success_rate"] == 0.0
        assert overview["avg_latency_ms"] == 0.0
        assert overview["total_cost"] == 0.0

    def test_get_overview_data_with_metrics(self) -> None:
        """Test getting overview data with metrics."""
        collector = MetricCollector()
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=True, latency_ms=200)
        collector.record_request(success=False, latency_ms=50)

        service = DashboardService(metric_collector=collector)
        overview = service.get_overview_data()

        assert overview["requests_24h"] == 3
        assert overview["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert overview["avg_latency_ms"] == pytest.approx(116.67, rel=0.1)

    def test_get_overview_data_with_costs(self) -> None:
        """Test getting overview data with cost tracking."""
        tracker = CostTracker()
        breakdown = tracker.start_request(trace_id="test")
        breakdown.add_usage(
            TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                model="claude-sonnet-4-20250514",
                agent_name="research",
            )
        )
        tracker.end_request()

        service = DashboardService(cost_tracker=tracker)
        overview = service.get_overview_data()

        assert overview["total_cost"] > 0
        assert overview["total_tokens"] == 1500

    def test_get_agent_performance_data(self) -> None:
        """Test getting agent performance data."""
        collector = MetricCollector()
        collector.record_agent_execution("agent1", success=True, latency_ms=100)
        collector.record_agent_execution("agent1", success=False, latency_ms=50)
        collector.record_agent_execution("agent2", success=True, latency_ms=150)

        service = DashboardService(metric_collector=collector)
        perf = service.get_agent_performance_data()

        assert "agent1" in perf["agents"]
        assert "agent2" in perf["agents"]
        assert perf["total_agent_calls"] == 3
        assert "agent1" in perf["agents_with_errors"]
        assert "agent2" not in perf["agents_with_errors"]

    def test_get_error_data(self) -> None:
        """Test getting error data."""
        collector = MetricCollector()
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=False, latency_ms=50)
        collector.record_agent_execution("agent1", success=False, latency_ms=50)

        service = DashboardService(metric_collector=collector)
        errors = service.get_error_data()

        assert errors["error_count"] == 1
        assert errors["error_rate"] == pytest.approx(33.33, rel=0.1)
        assert "agent1" in errors["errors_by_agent"]

    def test_get_cost_analysis_data_no_tracker(self) -> None:
        """Test getting cost data without tracker."""
        service = DashboardService()
        costs = service.get_cost_analysis_data()

        assert costs["total_cost"] == 0.0
        assert costs["by_agent"] == {}
        assert costs["by_model"] == {}

    def test_get_cost_analysis_data_with_tracker(self) -> None:
        """Test getting cost data with tracker."""
        tracker = CostTracker()

        # Add a request with cost
        breakdown = tracker.start_request(trace_id="test")
        breakdown.add_usage(
            TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                model="claude-sonnet-4-20250514",
                agent_name="research",
            )
        )
        tracker.end_request()

        service = DashboardService(cost_tracker=tracker)
        costs = service.get_cost_analysis_data()

        assert costs["total_cost"] > 0
        assert "research" in costs["by_agent"]
        assert "claude-sonnet-4-20250514" in costs["by_model"]

    def test_get_snapshot(self) -> None:
        """Test getting a complete dashboard snapshot."""
        collector = MetricCollector()
        collector.record_request(success=True, latency_ms=100)

        service = DashboardService(metric_collector=collector)
        snapshot = service.get_snapshot()

        assert snapshot.time_window == TimeWindow.DAY
        assert snapshot.overview["requests_24h"] == 1
        assert "agents" in snapshot.agent_performance
        assert "error_count" in snapshot.errors
        assert "total_cost" in snapshot.cost_analysis

    def test_get_snapshot_different_time_window(self) -> None:
        """Test getting snapshot with different time window."""
        service = DashboardService()
        snapshot = service.get_snapshot(time_window=TimeWindow.WEEK)

        assert snapshot.time_window == TimeWindow.WEEK


# ============================================================================
# Dashboard Factory Tests
# ============================================================================


class TestDashboardFactory:
    """Tests for dashboard factory functions."""

    def test_create_overview_dashboard(self) -> None:
        """Test creating overview dashboard."""
        dashboard = create_overview_dashboard()

        assert dashboard.id == "overview"
        assert dashboard.name == "System Overview"
        assert len(dashboard.panels) > 0

    def test_create_agent_dashboard(self) -> None:
        """Test creating agent dashboard."""
        dashboard = create_agent_dashboard()

        assert dashboard.id == "agent_performance"
        assert dashboard.name == "Agent Performance"
        assert len(dashboard.panels) > 0

    def test_create_error_dashboard(self) -> None:
        """Test creating error dashboard."""
        dashboard = create_error_dashboard()

        assert dashboard.id == "errors"
        assert dashboard.name == "Error Analysis"
        assert len(dashboard.panels) > 0

    def test_create_cost_dashboard(self) -> None:
        """Test creating cost dashboard."""
        dashboard = create_cost_dashboard()

        assert dashboard.id == "cost_analysis"
        assert dashboard.name == "Cost Analysis"
        assert len(dashboard.panels) > 0

    def test_create_all_dashboards(self) -> None:
        """Test creating all dashboards."""
        dashboards = create_all_dashboards()

        assert len(dashboards) == 4
        dashboard_ids = [d.id for d in dashboards]
        assert "overview" in dashboard_ids
        assert "agent_performance" in dashboard_ids
        assert "errors" in dashboard_ids
        assert "cost_analysis" in dashboard_ids


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstances:
    """Tests for global instance management."""

    def setup_method(self) -> None:
        """Reset global instances before each test."""
        reset_dashboard_service()

    def test_get_metric_collector_singleton(self) -> None:
        """Test metric collector is a singleton."""
        collector1 = get_metric_collector()
        collector2 = get_metric_collector()

        assert collector1 is collector2

    def test_get_dashboard_service_singleton(self) -> None:
        """Test dashboard service is a singleton."""
        service1 = get_dashboard_service()
        service2 = get_dashboard_service()

        assert service1 is service2

    def test_get_dashboard_service_uses_global_collector(self) -> None:
        """Test dashboard service uses global collector."""
        collector = get_metric_collector()
        service = get_dashboard_service()

        assert service.metrics is collector

    def test_reset_dashboard_service(self) -> None:
        """Test resetting dashboard service."""
        service1 = get_dashboard_service()
        reset_dashboard_service()
        service2 = get_dashboard_service()

        assert service1 is not service2


# ============================================================================
# Integration Tests
# ============================================================================


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""

    def test_full_dashboard_workflow(self) -> None:
        """Test complete dashboard workflow."""
        # Create service with cost tracker
        tracker = CostTracker()
        collector = MetricCollector()
        service = DashboardService(
            metric_collector=collector,
            cost_tracker=tracker,
        )

        # Record some activity
        collector.record_request(success=True, latency_ms=100)
        collector.record_request(success=True, latency_ms=200)
        collector.record_request(success=False, latency_ms=50)

        collector.record_agent_execution("research", success=True, latency_ms=150)
        collector.record_agent_execution("analysis", success=True, latency_ms=250)

        collector.record_tool_execution("market_data", success=True, latency_ms=80, cache_hit=True)

        # Record costs
        breakdown = tracker.start_request(trace_id="test-1")
        breakdown.add_usage(
            TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                model="claude-sonnet-4-20250514",
                agent_name="research",
            )
        )
        tracker.end_request()

        # Get snapshot
        snapshot = service.get_snapshot()

        # Verify overview
        assert snapshot.overview["requests_24h"] == 3
        assert snapshot.overview["success_rate"] == pytest.approx(66.67, rel=0.1)
        assert snapshot.overview["total_cost"] > 0

        # Verify agent performance
        assert "research" in snapshot.agent_performance["agents"]
        assert "analysis" in snapshot.agent_performance["agents"]

        # Verify errors
        assert snapshot.errors["error_count"] == 1

        # Verify cost analysis
        assert snapshot.cost_analysis["total_cost"] > 0
        assert "research" in snapshot.cost_analysis["by_agent"]

    def test_dashboard_export(self) -> None:
        """Test exporting dashboard configuration."""
        dashboard = create_overview_dashboard()
        exported = dashboard.to_dict()

        # Verify export structure
        assert "id" in exported
        assert "name" in exported
        assert "panels" in exported
        assert isinstance(exported["panels"], list)
        assert all("id" in p for p in exported["panels"])
        assert all("type" in p for p in exported["panels"])

    def test_snapshot_serialization(self) -> None:
        """Test snapshot serialization."""
        collector = MetricCollector()
        collector.record_request(success=True, latency_ms=100)

        service = DashboardService(metric_collector=collector)
        snapshot = service.get_snapshot()
        serialized = snapshot.to_dict()

        # Verify serialization
        assert "timestamp" in serialized
        assert "overview" in serialized
        assert "agent_performance" in serialized
        assert "errors" in serialized
        assert "cost_analysis" in serialized

        # Should be JSON-serializable (no datetime objects, etc.)
        import json

        json_str = json.dumps(serialized)
        assert len(json_str) > 0
