"""Tests for cache metrics tracking."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.caching.metrics import (
    CACHE_READ_COST_MULTIPLIER,
    CACHE_WRITE_COST_MULTIPLIER,
    CacheMetrics,
    CacheMetricsCollector,
)


class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_default_values(self):
        """Default values should be zero/empty."""
        metrics = CacheMetrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.cache_creation_tokens == 0
        assert metrics.cache_read_tokens == 0
        assert metrics.model == ""
        assert metrics.agent_name == ""
        assert isinstance(metrics.timestamp, datetime)

    def test_total_input_tokens(self):
        """total_input_tokens should sum regular and cached."""
        metrics = CacheMetrics(input_tokens=100, cache_read_tokens=500)
        assert metrics.total_input_tokens == 600

    def test_cache_hit_rate_with_hits(self):
        """cache_hit_rate should calculate correctly."""
        metrics = CacheMetrics(input_tokens=200, cache_read_tokens=800)
        # 800 / (200 + 800) = 0.8
        assert metrics.cache_hit_rate == 0.8

    def test_cache_hit_rate_no_cached(self):
        """cache_hit_rate should be 0 when no cached tokens."""
        metrics = CacheMetrics(input_tokens=100, cache_read_tokens=0)
        assert metrics.cache_hit_rate == 0.0

    def test_cache_hit_rate_zero_total(self):
        """cache_hit_rate should be 0 when no input tokens."""
        metrics = CacheMetrics(input_tokens=0, cache_read_tokens=0)
        assert metrics.cache_hit_rate == 0.0

    def test_estimated_savings_percent_with_cache(self):
        """estimated_savings_percent should calculate correctly."""
        # 100 regular + 900 cached = 1000 total
        # Uncached cost = 1000
        # Cached cost = 100 + (900 * 0.1) + (0 * 1.25) = 190
        # Savings = (1000 - 190) / 1000 * 100 = 81%
        metrics = CacheMetrics(input_tokens=100, cache_read_tokens=900)
        assert 80 <= metrics.estimated_savings_percent <= 82

    def test_estimated_savings_percent_no_cache(self):
        """estimated_savings_percent should be 0 without cache hits."""
        metrics = CacheMetrics(input_tokens=100, cache_read_tokens=0)
        assert metrics.estimated_savings_percent == 0.0

    def test_estimated_savings_with_creation(self):
        """estimated_savings_percent accounts for cache creation cost."""
        # First call: creates cache (more expensive)
        metrics = CacheMetrics(
            input_tokens=100,
            cache_read_tokens=0,
            cache_creation_tokens=500,
        )
        # No savings on cache miss
        assert metrics.estimated_savings_percent == 0.0

    def test_is_cache_hit_true(self):
        """is_cache_hit should be True when cache_read_tokens > 0."""
        metrics = CacheMetrics(cache_read_tokens=100)
        assert metrics.is_cache_hit is True

    def test_is_cache_hit_false(self):
        """is_cache_hit should be False when no cache reads."""
        metrics = CacheMetrics(cache_read_tokens=0)
        assert metrics.is_cache_hit is False

    def test_is_cache_creation_true(self):
        """is_cache_creation should be True when cache created."""
        metrics = CacheMetrics(cache_creation_tokens=100)
        assert metrics.is_cache_creation is True

    def test_is_cache_creation_false(self):
        """is_cache_creation should be False when no creation."""
        metrics = CacheMetrics(cache_creation_tokens=0)
        assert metrics.is_cache_creation is False

    def test_to_dict(self):
        """to_dict should return all relevant fields."""
        metrics = CacheMetrics(
            input_tokens=100,
            output_tokens=50,
            cache_creation_tokens=0,
            cache_read_tokens=400,
            model="claude-3-sonnet",
            agent_name="research",
        )
        result = metrics.to_dict()

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_creation_tokens"] == 0
        assert result["cache_read_tokens"] == 400
        assert result["total_input_tokens"] == 500
        assert result["model"] == "claude-3-sonnet"
        assert result["agent_name"] == "research"
        assert "cache_hit_rate" in result
        assert "estimated_savings_percent" in result
        assert "is_cache_hit" in result
        assert "is_cache_creation" in result
        assert "timestamp" in result

    def test_from_response_with_usage(self):
        """from_response should extract usage data."""
        mock_response = MagicMock()
        mock_response.usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=800,
        )

        metrics = CacheMetrics.from_response(
            mock_response,
            model="claude-3-opus",
            agent_name="analysis",
        )

        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.cache_creation_tokens == 200
        assert metrics.cache_read_tokens == 800
        assert metrics.model == "claude-3-opus"
        assert metrics.agent_name == "analysis"

    def test_from_response_no_usage(self):
        """from_response should handle missing usage."""
        mock_response = MagicMock(spec=[])

        metrics = CacheMetrics.from_response(
            mock_response,
            model="claude-3-sonnet",
            agent_name="test",
        )

        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.model == "claude-3-sonnet"
        assert metrics.agent_name == "test"

    def test_from_usage_dict(self):
        """from_usage_dict should extract from dictionary."""
        usage = {
            "input_tokens": 150,
            "output_tokens": 75,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 600,
        }

        metrics = CacheMetrics.from_usage_dict(
            usage,
            model="claude-3-haiku",
            agent_name="synthesis",
        )

        assert metrics.input_tokens == 150
        assert metrics.output_tokens == 75
        assert metrics.cache_creation_tokens == 300
        assert metrics.cache_read_tokens == 600
        assert metrics.model == "claude-3-haiku"
        assert metrics.agent_name == "synthesis"

    def test_from_usage_dict_missing_keys(self):
        """from_usage_dict should handle missing keys."""
        usage = {"input_tokens": 100}

        metrics = CacheMetrics.from_usage_dict(usage)

        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 0
        assert metrics.cache_creation_tokens == 0
        assert metrics.cache_read_tokens == 0


class TestCacheMetricsCollector:
    """Tests for CacheMetricsCollector class."""

    def test_initialization(self):
        """Collector should initialize properly."""
        collector = CacheMetricsCollector(
            workflow_id="wf-123",
            session_id="sess-456",
        )
        assert collector.workflow_id == "wf-123"
        assert collector.session_id == "sess-456"

    def test_record_metrics(self):
        """record should add metrics to the list."""
        collector = CacheMetricsCollector()
        metrics = CacheMetrics(input_tokens=100, cache_read_tokens=400)
        collector.record(metrics)

        summary = collector.get_summary()
        assert summary["call_count"] == 1

    def test_get_summary_empty(self):
        """get_summary should return zeros when empty."""
        collector = CacheMetricsCollector()
        summary = collector.get_summary()

        assert summary["call_count"] == 0
        assert summary["total_input_tokens"] == 0
        assert summary["overall_cache_hit_rate"] == 0.0
        assert summary["cache_hits"] == 0
        assert summary["cache_misses"] == 0

    def test_get_summary_aggregates(self):
        """get_summary should aggregate multiple calls."""
        collector = CacheMetricsCollector(workflow_id="test-wf")

        # First call: cache miss (creation)
        collector.record(
            CacheMetrics(
                input_tokens=500,
                output_tokens=100,
                cache_creation_tokens=500,
                cache_read_tokens=0,
                agent_name="research",
            )
        )

        # Second call: cache hit
        collector.record(
            CacheMetrics(
                input_tokens=100,
                output_tokens=150,
                cache_read_tokens=400,
                agent_name="research",
            )
        )

        summary = collector.get_summary()

        assert summary["call_count"] == 2
        assert summary["total_input_tokens"] == 600  # 500 + 100
        assert summary["total_output_tokens"] == 250  # 100 + 150
        assert summary["total_cache_creation_tokens"] == 500
        assert summary["total_cache_read_tokens"] == 400
        assert summary["cache_hits"] == 1
        assert summary["cache_misses"] == 1
        assert summary["workflow_id"] == "test-wf"

    def test_get_summary_overall_hit_rate(self):
        """overall_cache_hit_rate should be calculated correctly."""
        collector = CacheMetricsCollector()

        # Add metrics with known hit rate
        collector.record(CacheMetrics(input_tokens=200, cache_read_tokens=800))

        summary = collector.get_summary()
        # 800 / (200 + 800) = 0.8
        assert summary["overall_cache_hit_rate"] == 0.8

    def test_by_agent_summary(self):
        """Summary should include per-agent breakdown."""
        collector = CacheMetricsCollector()

        collector.record(
            CacheMetrics(input_tokens=100, cache_read_tokens=400, agent_name="agent1")
        )
        collector.record(
            CacheMetrics(input_tokens=200, cache_read_tokens=300, agent_name="agent2")
        )

        summary = collector.get_summary()

        assert "by_agent" in summary
        assert "agent1" in summary["by_agent"]
        assert "agent2" in summary["by_agent"]
        assert summary["by_agent"]["agent1"]["call_count"] == 1
        assert summary["by_agent"]["agent2"]["call_count"] == 1

    def test_by_agent_unknown(self):
        """Metrics without agent_name should be grouped as unknown."""
        collector = CacheMetricsCollector()
        collector.record(CacheMetrics(input_tokens=100))

        summary = collector.get_summary()

        assert "unknown" in summary["by_agent"]

    def test_clear(self):
        """clear should reset all metrics."""
        collector = CacheMetricsCollector()
        collector.record(CacheMetrics(input_tokens=100))
        collector.clear()

        summary = collector.get_summary()
        assert summary["call_count"] == 0

    @patch("src.caching.metrics.get_client")
    def test_report_to_langfuse(self, mock_get_client):
        """report_to_langfuse should call Langfuse client."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        collector = CacheMetricsCollector(
            workflow_id="wf-123",
            session_id="sess-456",
        )
        collector.record(CacheMetrics(input_tokens=100, cache_read_tokens=400))
        collector.report_to_langfuse()

        # Should call score at least once
        mock_client.score.assert_called()

    @patch("src.caching.metrics.get_client")
    def test_report_to_langfuse_no_session(self, mock_get_client):
        """report_to_langfuse should handle missing session_id."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        collector = CacheMetricsCollector(workflow_id="wf-123")
        collector.record(CacheMetrics(input_tokens=100))
        collector.report_to_langfuse()

        # Should not fail
        mock_client.score.assert_not_called()

    @patch("src.caching.metrics.get_client")
    def test_report_to_langfuse_error_handling(self, mock_get_client):
        """report_to_langfuse should handle errors gracefully."""
        mock_get_client.side_effect = Exception("Connection failed")

        collector = CacheMetricsCollector(session_id="sess-123")
        collector.record(CacheMetrics(input_tokens=100))

        # Should not raise
        collector.report_to_langfuse()


class TestCostConstants:
    """Tests for cost multiplier constants."""

    def test_cache_read_cost_multiplier(self):
        """Cache read should be 10% of base cost."""
        assert CACHE_READ_COST_MULTIPLIER == 0.10

    def test_cache_write_cost_multiplier(self):
        """Cache write should be 125% of base cost."""
        assert CACHE_WRITE_COST_MULTIPLIER == 1.25
