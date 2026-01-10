"""Tests for cache observability integration."""

from unittest.mock import MagicMock, patch

from src.caching.metrics import CacheMetrics
from src.caching.observability import (
    CacheObservabilityConfig,
    CacheObservabilityReporter,
    create_cache_reporter,
)


class TestCacheObservabilityConfig:
    """Tests for CacheObservabilityConfig."""

    def test_default_values(self):
        """Default config values should be sensible."""
        config = CacheObservabilityConfig()
        assert config.alert_on_low_hit_rate is True
        assert config.low_hit_rate_threshold == 0.8
        assert config.track_by_agent is True
        assert config.track_by_model is True

    def test_custom_values(self):
        """Custom config values should be applied."""
        config = CacheObservabilityConfig(
            alert_on_low_hit_rate=False,
            low_hit_rate_threshold=0.5,
            track_by_agent=False,
            track_by_model=False,
        )
        assert config.alert_on_low_hit_rate is False
        assert config.low_hit_rate_threshold == 0.5
        assert config.track_by_agent is False
        assert config.track_by_model is False


class TestCacheObservabilityReporter:
    """Tests for CacheObservabilityReporter."""

    def test_initialization(self):
        """Reporter should initialize with correct values."""
        reporter = CacheObservabilityReporter(
            trace_id="trace-123",
            session_id="sess-456",
        )
        assert reporter.trace_id == "trace-123"
        assert reporter.session_id == "sess-456"
        assert reporter.config is not None

    def test_initialization_with_config(self):
        """Reporter should accept custom config."""
        config = CacheObservabilityConfig(low_hit_rate_threshold=0.5)
        reporter = CacheObservabilityReporter(
            trace_id="trace-123",
            config=config,
        )
        assert reporter.config.low_hit_rate_threshold == 0.5

    @patch("src.caching.observability.get_client")
    def test_record_metrics(self, mock_get_client):
        """record should add metrics and report to Langfuse."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        reporter = CacheObservabilityReporter(trace_id="trace-123")
        metrics = CacheMetrics(
            input_tokens=100,
            cache_read_tokens=400,
            agent_name="research",
        )
        reporter.record(metrics)

        # Should create a span for the metrics
        mock_client.span.assert_called_once()
        span_kwargs = mock_client.span.call_args.kwargs
        assert span_kwargs["trace_id"] == "trace-123"
        assert "research" in span_kwargs["name"]

    @patch("src.caching.observability.get_client")
    def test_record_metrics_error_handling(self, mock_get_client):
        """record should handle Langfuse errors gracefully."""
        mock_get_client.side_effect = Exception("Connection failed")

        reporter = CacheObservabilityReporter(trace_id="trace-123")
        metrics = CacheMetrics(input_tokens=100)

        # Should not raise
        reporter.record(metrics)

    @patch("src.caching.observability.get_client")
    def test_report_summary(self, mock_get_client):
        """report_summary should report aggregated metrics."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        reporter = CacheObservabilityReporter(trace_id="trace-123")
        reporter.record(CacheMetrics(input_tokens=100, cache_read_tokens=400))

        summary = reporter.report_summary()

        assert "overall_cache_hit_rate" in summary
        assert "overall_savings_percent" in summary
        mock_client.score.assert_called()

    @patch("src.caching.observability.get_client")
    def test_report_summary_low_hit_rate_alert(self, mock_get_client):
        """report_summary should alert on low hit rate."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        config = CacheObservabilityConfig(
            alert_on_low_hit_rate=True,
            low_hit_rate_threshold=0.8,
        )
        reporter = CacheObservabilityReporter(
            trace_id="trace-123",
            config=config,
        )
        # Add metrics with low hit rate (50%)
        reporter.record(CacheMetrics(input_tokens=500, cache_read_tokens=500))

        reporter.report_summary()

        # Should update trace with alert
        mock_client.trace_update.assert_called()

    @patch("src.caching.observability.get_client")
    def test_report_summary_no_alert_above_threshold(self, mock_get_client):
        """report_summary should not alert when hit rate is above threshold."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        config = CacheObservabilityConfig(
            alert_on_low_hit_rate=True,
            low_hit_rate_threshold=0.8,
        )
        reporter = CacheObservabilityReporter(
            trace_id="trace-123",
            config=config,
        )
        # Add metrics with high hit rate (90%)
        reporter.record(CacheMetrics(input_tokens=100, cache_read_tokens=900))

        reporter.report_summary()

        # Should NOT update trace with alert
        mock_client.trace_update.assert_not_called()

    @patch("src.caching.observability.get_client")
    def test_report_summary_per_agent_breakdown(self, mock_get_client):
        """report_summary should include per-agent breakdown."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        reporter = CacheObservabilityReporter(trace_id="trace-123")
        reporter.record(
            CacheMetrics(input_tokens=100, cache_read_tokens=400, agent_name="agent1")
        )
        reporter.record(
            CacheMetrics(input_tokens=200, cache_read_tokens=300, agent_name="agent2")
        )

        summary = reporter.report_summary()

        assert "by_agent" in summary
        assert "agent1" in summary["by_agent"]
        assert "agent2" in summary["by_agent"]

        # Should have multiple score calls for per-agent metrics
        assert mock_client.score.call_count >= 2

    @patch("src.caching.observability.get_client")
    def test_report_summary_error_handling(self, mock_get_client):
        """report_summary should handle errors gracefully."""
        mock_get_client.side_effect = Exception("Connection failed")

        reporter = CacheObservabilityReporter(trace_id="trace-123")
        reporter.record(CacheMetrics(input_tokens=100))

        # Should not raise, should return summary
        summary = reporter.report_summary()
        assert summary is not None

    def test_get_metrics_for_state(self):
        """get_metrics_for_state should return state-friendly dict."""
        reporter = CacheObservabilityReporter(trace_id="trace-123")
        reporter._collector.record(
            CacheMetrics(input_tokens=100, cache_read_tokens=400)
        )

        state_metrics = reporter.get_metrics_for_state()

        assert "overall_cache_hit_rate" in state_metrics
        assert "overall_savings_percent" in state_metrics
        assert "total_cache_read_tokens" in state_metrics
        assert "cache_hits" in state_metrics
        assert "cache_misses" in state_metrics
        assert "by_agent" in state_metrics

    def test_get_metrics_for_state_empty(self):
        """get_metrics_for_state should handle empty collector."""
        reporter = CacheObservabilityReporter(trace_id="trace-123")
        state_metrics = reporter.get_metrics_for_state()

        assert state_metrics["overall_cache_hit_rate"] == 0.0
        assert state_metrics["cache_hits"] == 0


class TestCreateCacheReporter:
    """Tests for create_cache_reporter factory function."""

    def test_creates_reporter(self):
        """create_cache_reporter should return a reporter."""
        reporter = create_cache_reporter(trace_id="trace-123")
        assert isinstance(reporter, CacheObservabilityReporter)
        assert reporter.trace_id == "trace-123"

    def test_passes_session_id(self):
        """create_cache_reporter should pass session_id."""
        reporter = create_cache_reporter(
            trace_id="trace-123",
            session_id="sess-456",
        )
        assert reporter.session_id == "sess-456"

    def test_default_config(self):
        """create_cache_reporter should use default config."""
        reporter = create_cache_reporter(trace_id="trace-123")
        assert reporter.config.alert_on_low_hit_rate is True
        assert reporter.config.low_hit_rate_threshold == 0.8
