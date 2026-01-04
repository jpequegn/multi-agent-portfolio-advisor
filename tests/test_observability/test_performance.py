"""Tests for performance monitoring system.

Tests cover:
- Performance metrics and data models
- Latency tracking at all levels
- Percentile calculations (P50, P90, P95, P99)
- Bottleneck detection
- Performance regression alerts
- Performance reports
"""

from datetime import datetime, timedelta

import pytest

from src.observability.costs import AlertSeverity
from src.observability.performance import (
    TARGET_AGENT_LATENCY_MS,
    TARGET_END_TO_END_LATENCY_MS,
    TARGET_LLM_LATENCY_MS,
    TARGET_TOOL_LATENCY_MS,
    Bottleneck,
    BottleneckDetector,
    BottleneckType,
    LatencyRecord,
    PercentileCalculator,
    PercentileStats,
    PerformanceAlert,
    PerformanceAlertManager,
    PerformanceLevel,
    PerformanceMetrics,
    PerformanceReportGenerator,
    PerformanceThresholds,
    PerformanceTracker,
    format_performance_report,
    generate_performance_report,
    get_alert_manager,
    get_bottleneck_detector,
    get_performance_tracker,
    get_report_generator,
    identify_bottlenecks,
    reset_performance_tracking,
)

# ============================================================================
# Test Performance Targets
# ============================================================================


class TestPerformanceTargets:
    """Tests for performance target constants."""

    def test_targets_are_positive(self) -> None:
        """Test all targets are positive values."""
        assert TARGET_END_TO_END_LATENCY_MS > 0
        assert TARGET_AGENT_LATENCY_MS > 0
        assert TARGET_TOOL_LATENCY_MS > 0
        assert TARGET_LLM_LATENCY_MS > 0

    def test_target_hierarchy(self) -> None:
        """Test component targets are less than end-to-end target."""
        # Individual components should be faster than total
        assert TARGET_AGENT_LATENCY_MS < TARGET_END_TO_END_LATENCY_MS
        assert TARGET_TOOL_LATENCY_MS < TARGET_END_TO_END_LATENCY_MS
        assert TARGET_LLM_LATENCY_MS < TARGET_END_TO_END_LATENCY_MS


# ============================================================================
# Test Enums
# ============================================================================


class TestPerformanceLevel:
    """Tests for PerformanceLevel enum."""

    def test_performance_levels(self) -> None:
        """Test all performance levels exist."""
        assert PerformanceLevel.EXCELLENT.value == "excellent"
        assert PerformanceLevel.GOOD.value == "good"
        assert PerformanceLevel.ACCEPTABLE.value == "acceptable"
        assert PerformanceLevel.DEGRADED.value == "degraded"
        assert PerformanceLevel.CRITICAL.value == "critical"


class TestBottleneckType:
    """Tests for BottleneckType enum."""

    def test_bottleneck_types(self) -> None:
        """Test all bottleneck types exist."""
        assert BottleneckType.SLOW_AGENT.value == "slow_agent"
        assert BottleneckType.SLOW_TOOL.value == "slow_tool"
        assert BottleneckType.SLOW_LLM.value == "slow_llm"
        assert BottleneckType.HIGH_LATENCY.value == "high_latency"
        assert BottleneckType.QUEUE_DELAY.value == "queue_delay"


# ============================================================================
# Test Data Models
# ============================================================================


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_create_metrics(self) -> None:
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=5000.0,
            time_to_first_agent_ms=100.0,
        )

        assert metrics.trace_id == "trace-123"
        assert metrics.total_latency_ms == 5000.0
        assert metrics.time_to_first_agent_ms == 100.0
        assert metrics.agent_latencies == {}
        assert metrics.tool_latencies == {}
        assert metrics.llm_latencies == []

    def test_metrics_with_agent_latencies(self) -> None:
        """Test metrics with agent latencies."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=10000.0,
            agent_latencies={"research": 3000.0, "analysis": 4000.0},
            slowest_agent="analysis",
        )

        assert metrics.agent_latencies["research"] == 3000.0
        assert metrics.slowest_agent == "analysis"

    def test_metrics_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=5000.0,
            agent_latencies={"research": 3000.0},
        )

        result = metrics.to_dict()

        assert result["trace_id"] == "trace-123"
        assert result["total_latency_ms"] == 5000.0
        assert result["agent_latencies"] == {"research": 3000.0}
        assert "timestamp" in result

    def test_performance_level_excellent(self) -> None:
        """Test excellent performance level (< 50% of target)."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=TARGET_END_TO_END_LATENCY_MS * 0.3,
        )
        assert metrics.get_performance_level() == PerformanceLevel.EXCELLENT

    def test_performance_level_good(self) -> None:
        """Test good performance level (50-80% of target)."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=TARGET_END_TO_END_LATENCY_MS * 0.7,
        )
        assert metrics.get_performance_level() == PerformanceLevel.GOOD

    def test_performance_level_acceptable(self) -> None:
        """Test acceptable performance level (80-100% of target)."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=TARGET_END_TO_END_LATENCY_MS * 0.9,
        )
        assert metrics.get_performance_level() == PerformanceLevel.ACCEPTABLE

    def test_performance_level_degraded(self) -> None:
        """Test degraded performance level (100-150% of target)."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=TARGET_END_TO_END_LATENCY_MS * 1.3,
        )
        assert metrics.get_performance_level() == PerformanceLevel.DEGRADED

    def test_performance_level_critical(self) -> None:
        """Test critical performance level (> 150% of target)."""
        metrics = PerformanceMetrics(
            trace_id="trace-123",
            total_latency_ms=TARGET_END_TO_END_LATENCY_MS * 2.0,
        )
        assert metrics.get_performance_level() == PerformanceLevel.CRITICAL


class TestLatencyRecord:
    """Tests for LatencyRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a latency record."""
        record = LatencyRecord(
            trace_id="trace-123",
            component="agent",
            name="research",
            latency_ms=3000.0,
        )

        assert record.trace_id == "trace-123"
        assert record.component == "agent"
        assert record.name == "research"
        assert record.latency_ms == 3000.0
        assert isinstance(record.timestamp, datetime)

    def test_record_with_metadata(self) -> None:
        """Test record with metadata."""
        record = LatencyRecord(
            trace_id="trace-123",
            component="tool",
            name="get_market_data",
            latency_ms=500.0,
            metadata={"cache_hit": False},
        )

        assert record.metadata["cache_hit"] is False

    def test_record_to_dict(self) -> None:
        """Test converting record to dictionary."""
        record = LatencyRecord(
            trace_id="trace-123",
            component="llm",
            name="generation",
            latency_ms=2000.0,
        )

        result = record.to_dict()

        assert result["trace_id"] == "trace-123"
        assert result["component"] == "llm"
        assert result["latency_ms"] == 2000.0


class TestPercentileStats:
    """Tests for PercentileStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating percentile stats."""
        stats = PercentileStats(
            metric_name="request_latency",
            count=100,
            min_value=100.0,
            max_value=10000.0,
            mean=2500.0,
            p50=2000.0,
            p90=5000.0,
            p95=7000.0,
            p99=9000.0,
        )

        assert stats.metric_name == "request_latency"
        assert stats.count == 100
        assert stats.p50 == 2000.0
        assert stats.p95 == 7000.0

    def test_stats_to_dict(self) -> None:
        """Test converting stats to dictionary."""
        stats = PercentileStats(
            metric_name="request_latency",
            count=50,
            p50=1000.0,
            p99=5000.0,
        )

        result = stats.to_dict()

        assert result["metric_name"] == "request_latency"
        assert result["count"] == 50
        assert result["p50"] == 1000.0


class TestBottleneck:
    """Tests for Bottleneck dataclass."""

    def test_create_bottleneck(self) -> None:
        """Test creating a bottleneck."""
        bottleneck = Bottleneck(
            bottleneck_id="bn-001",
            trace_id="trace-123",
            bottleneck_type=BottleneckType.SLOW_AGENT,
            component="research_agent",
            latency_ms=15000.0,
            expected_ms=TARGET_AGENT_LATENCY_MS,
            impact_score=0.5,
            suggestion="Optimize research agent queries.",
        )

        assert bottleneck.bottleneck_id == "bn-001"
        assert bottleneck.bottleneck_type == BottleneckType.SLOW_AGENT
        assert bottleneck.component == "research_agent"
        assert bottleneck.impact_score == 0.5

    def test_bottleneck_to_dict(self) -> None:
        """Test converting bottleneck to dictionary."""
        bottleneck = Bottleneck(
            bottleneck_id="bn-001",
            trace_id="trace-123",
            bottleneck_type=BottleneckType.SLOW_TOOL,
            component="get_market_data",
            latency_ms=5000.0,
            expected_ms=2000.0,
            impact_score=1.0,
            suggestion="Add caching.",
        )

        result = bottleneck.to_dict()

        assert result["bottleneck_type"] == "slow_tool"
        assert result["impact_score"] == 1.0


class TestPerformanceAlert:
    """Tests for PerformanceAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating a performance alert."""
        alert = PerformanceAlert(
            alert_id="perf-001",
            severity=AlertSeverity.WARNING,
            metric_name="request_latency_p95",
            current_value=35000.0,
            baseline_value=25000.0,
            deviation_percent=40.0,
            message="P95 latency increased by 40%",
        )

        assert alert.alert_id == "perf-001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.deviation_percent == 40.0

    def test_alert_to_dict(self) -> None:
        """Test converting alert to dictionary."""
        alert = PerformanceAlert(
            alert_id="perf-001",
            severity=AlertSeverity.ERROR,
            metric_name="agent_latency",
            current_value=20000.0,
            baseline_value=10000.0,
            deviation_percent=100.0,
            message="Agent latency doubled",
        )

        result = alert.to_dict()

        assert result["severity"] == "error"
        assert result["deviation_percent"] == 100.0


# ============================================================================
# Test Percentile Calculator
# ============================================================================


class TestPercentileCalculator:
    """Tests for PercentileCalculator."""

    def test_calculate_empty_values(self) -> None:
        """Test calculating percentiles for empty list."""
        stats = PercentileCalculator.calculate([], "test_metric")

        assert stats.count == 0
        assert stats.p50 == 0.0
        assert stats.p99 == 0.0

    def test_calculate_single_value(self) -> None:
        """Test calculating percentiles for single value."""
        stats = PercentileCalculator.calculate([100.0], "test_metric")

        assert stats.count == 1
        assert stats.p50 == 100.0
        assert stats.p90 == 100.0
        assert stats.p99 == 100.0

    def test_calculate_percentiles(self) -> None:
        """Test calculating percentiles for multiple values."""
        values = list(range(1, 101))  # 1 to 100
        stats = PercentileCalculator.calculate(values, "test_metric")

        assert stats.count == 100
        assert stats.min_value == 1.0
        assert stats.max_value == 100.0
        assert stats.mean == 50.5
        # P50 should be around 50
        assert 49 <= stats.p50 <= 51
        # P90 should be around 90
        assert 89 <= stats.p90 <= 91
        # P95 should be around 95
        assert 94 <= stats.p95 <= 96
        # P99 should be around 99
        assert 98 <= stats.p99 <= 100

    def test_calculate_unsorted_values(self) -> None:
        """Test that values are sorted before calculation."""
        values = [100.0, 1.0, 50.0, 25.0, 75.0]
        stats = PercentileCalculator.calculate(values, "test_metric")

        assert stats.min_value == 1.0
        assert stats.max_value == 100.0


# ============================================================================
# Test Performance Tracker
# ============================================================================


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    @pytest.fixture
    def tracker(self) -> PerformanceTracker:
        """Create a fresh tracker for each test."""
        return PerformanceTracker()

    def test_record_request_latency(self, tracker: PerformanceTracker) -> None:
        """Test recording request latency."""
        tracker.record_request_latency("trace-123", 5000.0)

        metrics = tracker.get_metrics("trace-123")
        assert metrics is not None
        assert metrics.total_latency_ms == 5000.0

    def test_record_agent_latency(self, tracker: PerformanceTracker) -> None:
        """Test recording agent latency."""
        tracker.record_agent_latency("trace-123", "research", 3000.0)
        tracker.record_agent_latency("trace-123", "analysis", 4000.0)

        metrics = tracker.get_metrics("trace-123")
        assert metrics is not None
        assert metrics.agent_latencies["research"] == 3000.0
        assert metrics.agent_latencies["analysis"] == 4000.0
        assert metrics.slowest_agent == "analysis"

    def test_record_tool_latency(self, tracker: PerformanceTracker) -> None:
        """Test recording tool latency."""
        tracker.record_tool_latency("trace-123", "get_market_data", 500.0)
        tracker.record_tool_latency("trace-123", "get_news", 1000.0)

        metrics = tracker.get_metrics("trace-123")
        assert metrics is not None
        assert metrics.tool_latencies["get_market_data"] == 500.0
        assert metrics.slowest_tool == "get_news"

    def test_record_llm_latency(self, tracker: PerformanceTracker) -> None:
        """Test recording LLM latency."""
        tracker.record_llm_latency("trace-123", 2000.0, tokens_generated=500)
        tracker.record_llm_latency("trace-123", 3000.0, tokens_generated=1000)

        metrics = tracker.get_metrics("trace-123")
        assert metrics is not None
        assert len(metrics.llm_latencies) == 2
        assert metrics.tokens_per_second > 0

    def test_record_time_to_first_agent(self, tracker: PerformanceTracker) -> None:
        """Test recording time to first agent."""
        tracker.record_time_to_first_agent("trace-123", 150.0)

        metrics = tracker.get_metrics("trace-123")
        assert metrics is not None
        assert metrics.time_to_first_agent_ms == 150.0

    def test_get_percentiles(self, tracker: PerformanceTracker) -> None:
        """Test getting percentile statistics."""
        for i in range(100):
            tracker.record_request_latency(f"trace-{i}", float(i * 100 + 100))

        stats = tracker.get_percentiles("request")

        assert stats.count == 100
        assert stats.min_value == 100.0
        assert stats.max_value == 10000.0

    def test_get_percentiles_by_name(self, tracker: PerformanceTracker) -> None:
        """Test getting percentiles for specific component."""
        for i in range(50):
            tracker.record_agent_latency(f"trace-{i}", "research", float(i * 100))
            tracker.record_agent_latency(f"trace-{i}", "analysis", float(i * 200))

        research_stats = tracker.get_percentiles("agent", "research")
        analysis_stats = tracker.get_percentiles("agent", "analysis")

        assert research_stats.count == 50
        assert analysis_stats.count == 50
        assert analysis_stats.mean > research_stats.mean

    def test_set_and_get_baseline(self, tracker: PerformanceTracker) -> None:
        """Test setting and getting baselines."""
        tracker.set_baseline("request_latency_p95", 5000.0)

        baseline = tracker.get_baseline("request_latency_p95")
        assert baseline == 5000.0

        # Non-existent baseline
        assert tracker.get_baseline("unknown") is None

    def test_clear_old_records(self, tracker: PerformanceTracker) -> None:
        """Test clearing old records."""
        # Record some latencies
        for i in range(10):
            tracker.record_request_latency(f"trace-{i}", float(i * 100))

        # Clear records older than 0 days (all of them)
        removed = tracker.clear_old_records(max_age=timedelta(days=0))

        assert removed == 10


# ============================================================================
# Test Bottleneck Detector
# ============================================================================


class TestBottleneckDetector:
    """Tests for BottleneckDetector."""

    @pytest.fixture
    def tracker(self) -> PerformanceTracker:
        """Create a fresh tracker."""
        return PerformanceTracker()

    @pytest.fixture
    def detector(self, tracker: PerformanceTracker) -> BottleneckDetector:
        """Create a bottleneck detector."""
        return BottleneckDetector(tracker)

    def test_no_bottlenecks_for_good_performance(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test no bottlenecks when performance is good."""
        tracker.record_request_latency("trace-123", TARGET_END_TO_END_LATENCY_MS * 0.5)
        tracker.record_agent_latency("trace-123", "research", TARGET_AGENT_LATENCY_MS * 0.5)

        bottlenecks = detector.identify_bottlenecks("trace-123")
        assert len(bottlenecks) == 0

    def test_detect_high_latency_bottleneck(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test detecting high overall latency."""
        tracker.record_request_latency("trace-123", TARGET_END_TO_END_LATENCY_MS * 1.5)

        bottlenecks = detector.identify_bottlenecks("trace-123")

        assert len(bottlenecks) == 1
        assert bottlenecks[0].bottleneck_type == BottleneckType.HIGH_LATENCY

    def test_detect_slow_agent_bottleneck(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test detecting slow agent."""
        tracker.record_agent_latency("trace-123", "research", TARGET_AGENT_LATENCY_MS * 1.5)

        bottlenecks = detector.identify_bottlenecks("trace-123")

        slow_agent = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.SLOW_AGENT]
        assert len(slow_agent) == 1
        assert slow_agent[0].component == "research"

    def test_detect_slow_tool_bottleneck(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test detecting slow tool."""
        tracker.record_tool_latency("trace-123", "get_market_data", TARGET_TOOL_LATENCY_MS * 2)

        bottlenecks = detector.identify_bottlenecks("trace-123")

        slow_tool = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.SLOW_TOOL]
        assert len(slow_tool) == 1
        assert slow_tool[0].component == "get_market_data"

    def test_detect_slow_llm_bottleneck(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test detecting slow LLM call."""
        tracker.record_llm_latency("trace-123", TARGET_LLM_LATENCY_MS * 2)

        bottlenecks = detector.identify_bottlenecks("trace-123")

        slow_llm = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.SLOW_LLM]
        assert len(slow_llm) == 1

    def test_detect_queue_delay_bottleneck(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test detecting queue delay."""
        tracker.record_time_to_first_agent("trace-123", 2000.0)  # > 1 second

        bottlenecks = detector.identify_bottlenecks("trace-123")

        queue_delay = [b for b in bottlenecks if b.bottleneck_type == BottleneckType.QUEUE_DELAY]
        assert len(queue_delay) == 1

    def test_bottleneck_impact_score(
        self, tracker: PerformanceTracker, detector: BottleneckDetector
    ) -> None:
        """Test bottleneck impact score calculation."""
        # 50% over target -> impact score of 0.5
        tracker.record_agent_latency("trace-123", "research", TARGET_AGENT_LATENCY_MS * 1.5)

        bottlenecks = detector.identify_bottlenecks("trace-123")

        assert len(bottlenecks) == 1
        assert 0.4 <= bottlenecks[0].impact_score <= 0.6

    def test_no_metrics_returns_empty(self, detector: BottleneckDetector) -> None:
        """Test that non-existent trace returns empty list."""
        bottlenecks = detector.identify_bottlenecks("non-existent-trace")
        assert bottlenecks == []


# ============================================================================
# Test Performance Alert Manager
# ============================================================================


class TestPerformanceAlertManager:
    """Tests for PerformanceAlertManager."""

    @pytest.fixture
    def tracker(self) -> PerformanceTracker:
        """Create a fresh tracker."""
        return PerformanceTracker()

    @pytest.fixture
    def alert_manager(self, tracker: PerformanceTracker) -> PerformanceAlertManager:
        """Create an alert manager."""
        return PerformanceAlertManager(tracker)

    def test_no_alert_without_baseline(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test no alert when baseline is not set."""
        # Record some data but don't set baseline
        tracker.record_request_latency("trace-123", 10000.0)
        alert = alert_manager.check_regression("request_latency", 10000.0)
        assert alert is None

    def test_no_alert_within_threshold(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test no alert when within threshold."""
        tracker.set_baseline("request_latency", 10000.0)

        # 10% increase is below warning threshold (20%)
        alert = alert_manager.check_regression("request_latency", 11000.0)
        assert alert is None

    def test_warning_alert(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test warning alert at 20-50% deviation."""
        tracker.set_baseline("request_latency", 10000.0)

        # 30% increase
        alert = alert_manager.check_regression("request_latency", 13000.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert 25 <= alert.deviation_percent <= 35

    def test_error_alert(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test error alert at 50-100% deviation."""
        tracker.set_baseline("request_latency", 10000.0)

        # 75% increase
        alert = alert_manager.check_regression("request_latency", 17500.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.ERROR

    def test_critical_alert(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test critical alert at > 100% deviation."""
        tracker.set_baseline("request_latency", 10000.0)

        # 150% increase
        alert = alert_manager.check_regression("request_latency", 25000.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_custom_thresholds(self, tracker: PerformanceTracker) -> None:
        """Test custom alert thresholds."""
        custom_thresholds = PerformanceThresholds(
            warning_deviation_percent=10.0,
            error_deviation_percent=25.0,
            critical_deviation_percent=50.0,
        )
        alert_manager = PerformanceAlertManager(tracker, custom_thresholds)
        tracker.set_baseline("request_latency", 10000.0)

        # 15% increase triggers warning with custom thresholds
        alert = alert_manager.check_regression("request_latency", 11500.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_callback(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test alert callback is called."""
        tracker.set_baseline("request_latency", 10000.0)
        callback_alerts: list[PerformanceAlert] = []

        alert_manager.register_callback(lambda a: callback_alerts.append(a))
        alert_manager.check_regression("request_latency", 25000.0)

        assert len(callback_alerts) == 1

    def test_get_alerts_filtered(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test filtering alerts by severity."""
        tracker.set_baseline("metric1", 100.0)
        tracker.set_baseline("metric2", 100.0)

        alert_manager.check_regression("metric1", 130.0)  # Warning
        alert_manager.check_regression("metric2", 250.0)  # Critical

        warnings = alert_manager.get_alerts(severity=AlertSeverity.WARNING)
        critical = alert_manager.get_alerts(severity=AlertSeverity.CRITICAL)

        assert len(warnings) == 1
        assert len(critical) == 1

    def test_clear_alerts(
        self, tracker: PerformanceTracker, alert_manager: PerformanceAlertManager
    ) -> None:
        """Test clearing alerts."""
        tracker.set_baseline("request_latency", 10000.0)
        alert_manager.check_regression("request_latency", 25000.0)

        cleared = alert_manager.clear_alerts()

        assert cleared == 1
        assert len(alert_manager.get_alerts()) == 0


# ============================================================================
# Test Performance Report Generator
# ============================================================================


class TestPerformanceReportGenerator:
    """Tests for PerformanceReportGenerator."""

    @pytest.fixture
    def tracker(self) -> PerformanceTracker:
        """Create a fresh tracker with sample data."""
        tracker = PerformanceTracker()

        # Add sample data
        for i in range(20):
            trace_id = f"trace-{i}"
            tracker.record_request_latency(trace_id, float(5000 + i * 100))
            tracker.record_agent_latency(trace_id, "research", float(2000 + i * 50))
            tracker.record_agent_latency(trace_id, "analysis", float(2500 + i * 50))
            tracker.record_tool_latency(trace_id, "get_market_data", float(500 + i * 10))
            tracker.record_llm_latency(trace_id, float(1000 + i * 20))

        return tracker

    @pytest.fixture
    def detector(self, tracker: PerformanceTracker) -> BottleneckDetector:
        """Create a bottleneck detector."""
        return BottleneckDetector(tracker)

    @pytest.fixture
    def alert_manager(self, tracker: PerformanceTracker) -> PerformanceAlertManager:
        """Create an alert manager."""
        return PerformanceAlertManager(tracker)

    @pytest.fixture
    def generator(
        self,
        tracker: PerformanceTracker,
        detector: BottleneckDetector,
        alert_manager: PerformanceAlertManager,
    ) -> PerformanceReportGenerator:
        """Create a report generator."""
        return PerformanceReportGenerator(tracker, detector, alert_manager)

    def test_generate_report(self, generator: PerformanceReportGenerator) -> None:
        """Test generating a performance report."""
        report = generator.generate_report()

        assert report.report_id.startswith("perf-report-")
        assert report.total_requests == 20
        assert report.request_latency_stats.count == 20
        assert "research" in report.agent_latency_stats
        assert "analysis" in report.agent_latency_stats
        assert "get_market_data" in report.tool_latency_stats

    def test_report_has_recommendations(self, generator: PerformanceReportGenerator) -> None:
        """Test that report includes recommendations."""
        report = generator.generate_report()

        assert len(report.recommendations) > 0

    def test_report_to_dict(self, generator: PerformanceReportGenerator) -> None:
        """Test converting report to dictionary."""
        report = generator.generate_report()
        result = report.to_dict()

        assert "report_id" in result
        assert "request_latency_stats" in result
        assert "recommendations" in result

    def test_report_with_time_window(self, generator: PerformanceReportGenerator) -> None:
        """Test report with custom time window."""
        report = generator.generate_report(time_window=timedelta(hours=24))

        # 24 hours is formatted as 1 day
        assert report.time_window == "1d"


# ============================================================================
# Test Module-Level Functions
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level singleton functions."""

    def setup_method(self) -> None:
        """Reset singletons before each test."""
        reset_performance_tracking()

    def teardown_method(self) -> None:
        """Reset singletons after each test."""
        reset_performance_tracking()

    def test_get_performance_tracker(self) -> None:
        """Test getting global tracker."""
        tracker1 = get_performance_tracker()
        tracker2 = get_performance_tracker()

        assert tracker1 is tracker2
        assert isinstance(tracker1, PerformanceTracker)

    def test_get_bottleneck_detector(self) -> None:
        """Test getting global detector."""
        detector1 = get_bottleneck_detector()
        detector2 = get_bottleneck_detector()

        assert detector1 is detector2
        assert isinstance(detector1, BottleneckDetector)

    def test_get_alert_manager(self) -> None:
        """Test getting global alert manager."""
        manager1 = get_alert_manager()
        manager2 = get_alert_manager()

        assert manager1 is manager2
        assert isinstance(manager1, PerformanceAlertManager)

    def test_get_report_generator(self) -> None:
        """Test getting global report generator."""
        generator1 = get_report_generator()
        generator2 = get_report_generator()

        assert generator1 is generator2
        assert isinstance(generator1, PerformanceReportGenerator)

    def test_reset_performance_tracking(self) -> None:
        """Test resetting all singletons."""
        tracker1 = get_performance_tracker()
        reset_performance_tracking()
        tracker2 = get_performance_tracker()

        assert tracker1 is not tracker2

    def test_identify_bottlenecks_function(self) -> None:
        """Test module-level identify_bottlenecks function."""
        tracker = get_performance_tracker()
        tracker.record_request_latency("trace-123", TARGET_END_TO_END_LATENCY_MS * 2)

        bottlenecks = identify_bottlenecks("trace-123")

        assert len(bottlenecks) >= 1

    def test_generate_performance_report_function(self) -> None:
        """Test module-level generate_performance_report function."""
        tracker = get_performance_tracker()
        tracker.record_request_latency("trace-123", 5000.0)

        report = generate_performance_report()

        assert report.total_requests == 1

    def test_format_performance_report_function(self) -> None:
        """Test formatting a performance report."""
        tracker = get_performance_tracker()
        tracker.record_request_latency("trace-123", 5000.0)
        tracker.record_agent_latency("trace-123", "research", 2000.0)

        report = generate_performance_report()
        formatted = format_performance_report(report)

        assert "PERFORMANCE REPORT" in formatted
        assert "REQUEST LATENCY" in formatted
        assert "RECOMMENDATIONS" in formatted


# ============================================================================
# Test Integration
# ============================================================================


class TestPerformanceIntegration:
    """Integration tests for the performance monitoring system."""

    def setup_method(self) -> None:
        """Reset singletons before each test."""
        reset_performance_tracking()

    def teardown_method(self) -> None:
        """Reset singletons after each test."""
        reset_performance_tracking()

    def test_full_workflow(self) -> None:
        """Test complete monitoring workflow."""
        tracker = get_performance_tracker()

        # Simulate a request
        trace_id = "integration-trace-001"
        tracker.record_time_to_first_agent(trace_id, 100.0)
        tracker.record_agent_latency(trace_id, "research", 5000.0)
        tracker.record_tool_latency(trace_id, "get_market_data", 1000.0)
        tracker.record_llm_latency(trace_id, 2000.0, tokens_generated=500)
        tracker.record_agent_latency(trace_id, "analysis", 4000.0)
        tracker.record_request_latency(trace_id, 12000.0)

        # Get metrics
        metrics = tracker.get_metrics(trace_id)
        assert metrics is not None
        assert metrics.total_latency_ms == 12000.0
        assert metrics.slowest_agent == "research"

        # Check performance level
        assert metrics.get_performance_level() == PerformanceLevel.EXCELLENT

        # Identify bottlenecks
        bottlenecks = identify_bottlenecks(trace_id)
        # Should have no bottlenecks since we're under all targets
        assert len(bottlenecks) == 0

        # Generate report
        report = generate_performance_report()
        assert report.total_requests == 1
        assert len(report.recommendations) > 0

    def test_regression_detection(self) -> None:
        """Test detecting performance regressions."""
        tracker = get_performance_tracker()
        alert_manager = get_alert_manager()

        # Set baseline
        tracker.set_baseline("request_latency", 5000.0)

        # Check for regression with doubled latency
        alert = alert_manager.check_regression("request_latency", 10000.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.deviation_percent == 100.0

    def test_multiple_traces_aggregation(self) -> None:
        """Test aggregating multiple traces."""
        tracker = get_performance_tracker()

        # Add multiple traces
        for i in range(10):
            trace_id = f"multi-trace-{i}"
            tracker.record_request_latency(trace_id, float(5000 + i * 500))
            tracker.record_agent_latency(trace_id, "research", float(2000 + i * 100))

        # Get percentiles
        request_stats = tracker.get_percentiles("request")
        agent_stats = tracker.get_percentiles("agent", "research")

        assert request_stats.count == 10
        assert agent_stats.count == 10
        assert request_stats.p50 > 0
        assert agent_stats.p95 > 0
