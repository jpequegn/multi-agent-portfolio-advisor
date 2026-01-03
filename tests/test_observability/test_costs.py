"""Tests for cost attribution system."""

from datetime import UTC, datetime, timedelta

import pytest

from src.observability.costs import (
    AlertSeverity,
    CostAlert,
    CostAlertManager,
    CostBreakdown,
    CostReport,
    CostThresholds,
    CostTracker,
    ModelPricing,
    ModelProvider,
    TokenUsage,
    calculate_llm_cost,
    estimate_request_cost,
    format_cost,
    generate_cost_report,
    get_cost_tracker,
    get_model_pricing,
    reset_cost_tracker,
)


# ============================================================================
# Model Pricing Tests
# ============================================================================


class TestModelPricing:
    """Tests for ModelPricing class."""

    def test_pricing_properties(self) -> None:
        """Test pricing calculation properties."""
        pricing = ModelPricing(
            model_id="test-model",
            provider=ModelProvider.ANTHROPIC,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        )

        assert pricing.input_cost_per_token == 3.00 / 1_000_000
        assert pricing.output_cost_per_token == 15.00 / 1_000_000

    def test_calculate_cost(self) -> None:
        """Test cost calculation."""
        pricing = ModelPricing(
            model_id="test-model",
            provider=ModelProvider.ANTHROPIC,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        )

        # 1000 input tokens = $0.003, 500 output tokens = $0.0075
        cost = pricing.calculate_cost(1000, 500)
        assert abs(cost - 0.0105) < 0.0001

    def test_calculate_cost_zero_tokens(self) -> None:
        """Test cost calculation with zero tokens."""
        pricing = ModelPricing(
            model_id="test-model",
            provider=ModelProvider.ANTHROPIC,
            input_cost_per_million=3.00,
            output_cost_per_million=15.00,
        )

        cost = pricing.calculate_cost(0, 0)
        assert cost == 0.0


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_exact_match(self) -> None:
        """Test exact model ID match."""
        pricing = get_model_pricing("claude-sonnet-4-20250514")
        assert pricing.model_id == "claude-sonnet-4-20250514"
        assert pricing.input_cost_per_million == 3.00

    def test_partial_match(self) -> None:
        """Test partial model ID match."""
        pricing = get_model_pricing("claude-3-5-sonnet-latest")
        # Should match claude-3-5-sonnet-20241022
        assert pricing.provider == ModelProvider.ANTHROPIC

    def test_unknown_model_returns_default(self) -> None:
        """Test unknown model returns default pricing."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing.model_id == "unknown"
        assert pricing.input_cost_per_million == 3.00


# ============================================================================
# Token Usage Tests
# ============================================================================


class TestTokenUsage:
    """Tests for TokenUsage class."""

    def test_default_values(self) -> None:
        """Test TokenUsage with default values."""
        usage = TokenUsage()

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.model == ""
        assert usage.agent_name == ""
        assert isinstance(usage.timestamp, datetime)

    def test_total_tokens(self) -> None:
        """Test total_tokens property."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)

        assert usage.total_tokens == 1500

    def test_calculate_cost(self) -> None:
        """Test cost calculation."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
        )

        cost = usage.calculate_cost()
        # 1000 * $3/M + 500 * $15/M = $0.003 + $0.0075 = $0.0105
        assert abs(cost - 0.0105) < 0.0001

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            agent_name="research",
        )

        result = usage.to_dict()

        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["agent_name"] == "research"
        assert "cost" in result
        assert "timestamp" in result


# ============================================================================
# Cost Breakdown Tests
# ============================================================================


class TestCostBreakdown:
    """Tests for CostBreakdown class."""

    def test_default_values(self) -> None:
        """Test CostBreakdown with default values."""
        breakdown = CostBreakdown()

        assert breakdown.trace_id == ""
        assert breakdown.total_cost == 0.0
        assert breakdown.by_agent == {}
        assert breakdown.by_model == {}
        assert breakdown.input_tokens == 0
        assert breakdown.output_tokens == 0
        assert breakdown.usages == []

    def test_add_usage(self) -> None:
        """Test adding token usage."""
        breakdown = CostBreakdown(trace_id="test-123")

        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            agent_name="research",
        )
        breakdown.add_usage(usage)

        assert breakdown.input_tokens == 1000
        assert breakdown.output_tokens == 500
        assert breakdown.total_cost > 0
        assert "research" in breakdown.by_agent
        assert "claude-sonnet-4-20250514" in breakdown.by_model
        assert len(breakdown.usages) == 1

    def test_add_multiple_usages(self) -> None:
        """Test adding multiple usages."""
        breakdown = CostBreakdown()

        breakdown.add_usage(TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            agent_name="research",
        ))
        breakdown.add_usage(TokenUsage(
            input_tokens=800,
            output_tokens=400,
            model="claude-sonnet-4-20250514",
            agent_name="analysis",
        ))

        assert breakdown.input_tokens == 1800
        assert breakdown.output_tokens == 900
        assert len(breakdown.by_agent) == 2
        assert len(breakdown.usages) == 2

    def test_add_agent_cost(self) -> None:
        """Test adding cost for an agent."""
        breakdown = CostBreakdown()

        breakdown.add_agent_cost("research", 0.05)
        breakdown.add_agent_cost("research", 0.03)
        breakdown.add_agent_cost("analysis", 0.02)

        assert breakdown.by_agent["research"] == 0.08
        assert breakdown.by_agent["analysis"] == 0.02
        assert breakdown.total_cost == 0.10

    def test_finalize(self) -> None:
        """Test finalizing breakdown."""
        breakdown = CostBreakdown()

        assert breakdown.ended_at is None

        breakdown.finalize()

        assert breakdown.ended_at is not None

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        start = datetime.now(UTC)
        breakdown = CostBreakdown(started_at=start)
        breakdown.ended_at = start + timedelta(seconds=5)

        assert breakdown.duration_seconds == 5.0

    def test_duration_seconds_not_finalized(self) -> None:
        """Test duration when not finalized."""
        breakdown = CostBreakdown()

        assert breakdown.duration_seconds is None

    def test_cost_per_token(self) -> None:
        """Test cost per token calculation."""
        breakdown = CostBreakdown()
        breakdown.total_cost = 0.10
        breakdown.input_tokens = 5000
        breakdown.output_tokens = 5000

        assert breakdown.cost_per_token == 0.00001

    def test_cost_per_token_zero_tokens(self) -> None:
        """Test cost per token with zero tokens."""
        breakdown = CostBreakdown()

        assert breakdown.cost_per_token == 0.0

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        breakdown = CostBreakdown(
            trace_id="test-123",
            workflow_id="wf-456",
        )
        breakdown.add_usage(TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            agent_name="research",
        ))
        breakdown.finalize()

        result = breakdown.to_dict()

        assert result["trace_id"] == "test-123"
        assert result["workflow_id"] == "wf-456"
        assert "total_cost" in result
        assert "total_cost_formatted" in result
        assert "by_agent" in result
        assert "by_model" in result
        assert "duration_seconds" in result


# ============================================================================
# Cost Alert Tests
# ============================================================================


class TestCostAlert:
    """Tests for CostAlert class."""

    def test_create_alert(self) -> None:
        """Test creating a cost alert."""
        alert = CostAlert(
            severity=AlertSeverity.WARNING,
            message="Cost exceeded threshold",
            threshold=0.10,
            actual_value=0.15,
            context={"trace_id": "test-123"},
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Cost exceeded threshold"
        assert alert.threshold == 0.10
        assert alert.actual_value == 0.15
        assert alert.context == {"trace_id": "test-123"}

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        alert = CostAlert(
            severity=AlertSeverity.CRITICAL,
            message="Critical cost",
            threshold=0.50,
            actual_value=0.75,
        )

        result = alert.to_dict()

        assert result["severity"] == "critical"
        assert result["message"] == "Critical cost"
        assert result["threshold"] == 0.50
        assert result["actual_value"] == 0.75
        assert "timestamp" in result


class TestCostAlertManager:
    """Tests for CostAlertManager class."""

    def test_default_thresholds(self) -> None:
        """Test default thresholds are set."""
        manager = CostAlertManager()

        assert manager.thresholds.per_request_warning == 0.10
        assert manager.thresholds.per_request_critical == 0.50

    def test_check_request_cost_below_threshold(self) -> None:
        """Test no alerts when below threshold."""
        manager = CostAlertManager()
        breakdown = CostBreakdown()
        breakdown.total_cost = 0.05  # Below warning threshold

        alerts = manager.check_request_cost(breakdown)

        assert len(alerts) == 0

    def test_check_request_cost_warning(self) -> None:
        """Test warning alert when exceeds warning threshold."""
        manager = CostAlertManager()
        breakdown = CostBreakdown(trace_id="test-123")
        breakdown.total_cost = 0.15  # Between warning and critical

        alerts = manager.check_request_cost(breakdown)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_check_request_cost_critical(self) -> None:
        """Test critical alert when exceeds critical threshold."""
        manager = CostAlertManager()
        breakdown = CostBreakdown(trace_id="test-123")
        breakdown.total_cost = 0.60  # Above critical threshold

        alerts = manager.check_request_cost(breakdown)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_check_per_agent_cost(self) -> None:
        """Test per-agent cost alerts."""
        manager = CostAlertManager()
        breakdown = CostBreakdown()
        breakdown.by_agent = {"research": 0.30}  # Above agent critical

        alerts = manager.check_request_cost(breakdown)

        agent_alerts = [a for a in alerts if "Agent" in a.message]
        assert len(agent_alerts) == 1
        assert agent_alerts[0].severity == AlertSeverity.CRITICAL

    def test_check_token_usage(self) -> None:
        """Test token usage alerts."""
        manager = CostAlertManager()
        breakdown = CostBreakdown()
        breakdown.input_tokens = 40000
        breakdown.output_tokens = 20000  # Total 60k, above 50k threshold

        alerts = manager.check_request_cost(breakdown)

        token_alerts = [a for a in alerts if "Token" in a.message]
        assert len(token_alerts) == 1

    def test_on_alert_callback(self) -> None:
        """Test on_alert callback is called."""
        callback_alerts: list[CostAlert] = []

        def callback(alert: CostAlert) -> None:
            callback_alerts.append(alert)

        manager = CostAlertManager(on_alert=callback)
        breakdown = CostBreakdown()
        breakdown.total_cost = 0.60  # Above critical

        manager.check_request_cost(breakdown)

        assert len(callback_alerts) == 1

    def test_record_and_check_daily_cost(self) -> None:
        """Test daily cost tracking and alerts."""
        thresholds = CostThresholds(daily_warning=1.00, daily_critical=5.00)
        manager = CostAlertManager(thresholds=thresholds)

        manager.record_daily_cost(0.50)
        manager.record_daily_cost(0.60)  # Total 1.10, above warning

        alerts = manager.check_daily_cost()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_get_daily_cost(self) -> None:
        """Test getting daily cost."""
        manager = CostAlertManager()

        manager.record_daily_cost(0.50)
        manager.record_daily_cost(0.30)

        assert manager.get_daily_cost() == 0.80

    def test_clear_alerts(self) -> None:
        """Test clearing alerts."""
        manager = CostAlertManager()
        breakdown = CostBreakdown()
        breakdown.total_cost = 0.60

        manager.check_request_cost(breakdown)
        assert len(manager.alerts) > 0

        manager.clear_alerts()
        assert len(manager.alerts) == 0


# ============================================================================
# Cost Tracker Tests
# ============================================================================


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_start_request(self) -> None:
        """Test starting a request."""
        tracker = CostTracker()

        breakdown = tracker.start_request(
            trace_id="test-123",
            workflow_id="wf-456",
        )

        assert breakdown.trace_id == "test-123"
        assert breakdown.workflow_id == "wf-456"
        assert tracker.current_breakdown is breakdown

    def test_record_usage(self) -> None:
        """Test recording token usage."""
        tracker = CostTracker()
        tracker.start_request(trace_id="test-123")

        usage = tracker.record_usage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            agent_name="research",
        )

        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert tracker.current_breakdown is not None
        assert tracker.current_breakdown.input_tokens == 1000

    def test_end_request(self) -> None:
        """Test ending a request."""
        tracker = CostTracker()
        tracker.start_request(trace_id="test-123")
        tracker.record_usage(1000, 500, "claude-sonnet-4-20250514", "research")

        breakdown = tracker.end_request()

        assert breakdown is not None
        assert breakdown.ended_at is not None
        assert tracker.current_breakdown is None
        assert len(tracker.completed_breakdowns) == 1

    def test_end_request_no_active(self) -> None:
        """Test ending request when none active."""
        tracker = CostTracker()

        result = tracker.end_request()

        assert result is None

    def test_multiple_requests(self) -> None:
        """Test tracking multiple requests."""
        tracker = CostTracker()

        # First request
        tracker.start_request(trace_id="test-1")
        tracker.record_usage(1000, 500, "claude-sonnet-4-20250514", "research")
        tracker.end_request()

        # Second request
        tracker.start_request(trace_id="test-2")
        tracker.record_usage(800, 400, "claude-sonnet-4-20250514", "analysis")
        tracker.end_request()

        assert len(tracker.completed_breakdowns) == 2


# ============================================================================
# Cost Report Tests
# ============================================================================


class TestCostReport:
    """Tests for CostReport class."""

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        report = CostReport(
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC),
            total_cost=1.50,
            total_requests=10,
            total_tokens=100000,
            by_agent={"research": 0.80, "analysis": 0.70},
            average_cost_per_request=0.15,
        )

        result = report.to_dict()

        assert result["total_cost"] == 1.50
        assert result["total_cost_formatted"] == "$1.50"
        assert result["total_requests"] == 10
        assert result["total_tokens"] == 100000


class TestGenerateCostReport:
    """Tests for generate_cost_report function."""

    def test_empty_breakdowns(self) -> None:
        """Test report with no breakdowns."""
        report = generate_cost_report([])

        assert report.total_cost == 0.0
        assert report.total_requests == 0

    def test_single_breakdown(self) -> None:
        """Test report with single breakdown."""
        breakdown = CostBreakdown(trace_id="test-1")
        breakdown.add_usage(TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-20250514",
            agent_name="research",
        ))
        breakdown.finalize()

        report = generate_cost_report([breakdown])

        assert report.total_requests == 1
        assert report.total_cost > 0
        assert "research" in report.by_agent
        assert "claude-sonnet-4-20250514" in report.by_model

    def test_multiple_breakdowns(self) -> None:
        """Test report with multiple breakdowns."""
        breakdowns = []

        for i in range(3):
            breakdown = CostBreakdown(trace_id=f"test-{i}")
            breakdown.add_usage(TokenUsage(
                input_tokens=1000,
                output_tokens=500,
                model="claude-sonnet-4-20250514",
                agent_name="research" if i % 2 == 0 else "analysis",
            ))
            breakdown.finalize()
            breakdowns.append(breakdown)

        report = generate_cost_report(breakdowns)

        assert report.total_requests == 3
        assert len(report.by_agent) == 2
        assert report.average_cost_per_request > 0

    def test_with_alerts(self) -> None:
        """Test report includes alert count."""
        alerts = [
            CostAlert(
                severity=AlertSeverity.WARNING,
                message="Test",
                threshold=0.10,
                actual_value=0.15,
            ),
        ]

        report = generate_cost_report([], alerts=alerts)

        assert report.alerts_count == 1


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_llm_cost(self) -> None:
        """Test calculate_llm_cost function."""
        cost = calculate_llm_cost(
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )

        assert abs(cost - 0.0105) < 0.0001

    def test_format_cost_small(self) -> None:
        """Test format_cost for small amounts."""
        assert format_cost(0.001) == "$0.0010"
        assert format_cost(0.0001) == "$0.0001"

    def test_format_cost_medium(self) -> None:
        """Test format_cost for medium amounts."""
        assert format_cost(0.05) == "$0.050"
        assert format_cost(0.123) == "$0.123"

    def test_format_cost_large(self) -> None:
        """Test format_cost for large amounts."""
        assert format_cost(1.50) == "$1.50"
        assert format_cost(10.00) == "$10.00"

    def test_estimate_request_cost(self) -> None:
        """Test estimate_request_cost function."""
        cost = estimate_request_cost(
            symbols_count=5,
            model="claude-sonnet-4-20250514",
            agents_count=3,
        )

        # Should be a reasonable estimate
        assert cost > 0
        assert cost < 1.0  # Shouldn't be more than $1 for 5 symbols


class TestGlobalTracker:
    """Tests for global tracker functions."""

    def test_get_cost_tracker(self) -> None:
        """Test getting global tracker."""
        reset_cost_tracker()

        tracker1 = get_cost_tracker()
        tracker2 = get_cost_tracker()

        assert tracker1 is tracker2

    def test_reset_cost_tracker(self) -> None:
        """Test resetting global tracker."""
        tracker1 = get_cost_tracker()
        reset_cost_tracker()
        tracker2 = get_cost_tracker()

        assert tracker1 is not tracker2
