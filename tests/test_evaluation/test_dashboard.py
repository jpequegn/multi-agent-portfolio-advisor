"""Tests for evaluation dashboard."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evaluation.dashboard import EvaluationDashboard
from src.evaluation.models import (
    ABTestConfig,
    ABTestResult,
    DimensionScore,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
)


@pytest.fixture
def mock_storage():
    """Create a mock storage instance."""
    storage = MagicMock()
    storage.list_summaries = AsyncMock(return_value=[])
    storage.get_summary = AsyncMock(return_value=None)
    storage.get_ab_test = AsyncMock(return_value=None)
    storage.list_ab_tests = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def dashboard(mock_storage):
    """Create dashboard with mock storage."""
    return EvaluationDashboard(storage=mock_storage)


@pytest.fixture
def sample_summary() -> EvaluationRunSummary:
    """Create a sample evaluation summary."""
    config = EvaluationRunConfig(
        run_id="test-run-001",
        name="Test Run",
        metadata={"agent_name": "portfolio_agent"},
    )
    return EvaluationRunSummary(
        run_id="test-run-001",
        config=config,
        total_cases=10,
        passed_cases=8,
        failed_cases=2,
        pass_rate=0.8,
        mean_scores={
            "accuracy": 0.85,
            "relevance": 0.90,
            "aggregate": 0.87,
        },
        std_scores={
            "accuracy": 0.1,
            "relevance": 0.05,
            "aggregate": 0.08,
        },
        total_latency_ms=5000,
        mean_latency_ms=500.0,
        total_cost_usd=0.05,
        started_at=datetime.utcnow() - timedelta(hours=1),
        results=[
            EvaluationResult(
                case_id="case-1",
                run_id="test-run-001",
                model="claude-sonnet-4-20250514",
                scores=[
                    DimensionScore(
                        dimension=EvaluationDimension.ACCURACY,
                        score=0.9,
                    ),
                ],
                aggregate_score=0.9,
                passed=True,
            ),
        ],
    )


class TestEvaluationDashboard:
    """Tests for EvaluationDashboard class."""

    @pytest.mark.asyncio
    async def test_get_score_trends_empty(self, dashboard, mock_storage):
        """Test trends with no data."""
        mock_storage.list_summaries.return_value = []

        result = await dashboard.get_score_trends(days=30)

        assert result["labels"] == []
        assert result["datasets"] == []
        assert result["metadata"]["count"] == 0

    @pytest.mark.asyncio
    async def test_get_score_trends_with_data(self, dashboard, mock_storage, sample_summary):
        """Test trends with data."""
        mock_storage.list_summaries.return_value = [sample_summary]

        result = await dashboard.get_score_trends(days=30)

        assert len(result["labels"]) > 0
        assert len(result["datasets"]) > 0
        assert result["metadata"]["count"] == 1

    @pytest.mark.asyncio
    async def test_get_score_trends_with_filter(self, dashboard, mock_storage, sample_summary):
        """Test trends with agent filter."""
        mock_storage.list_summaries.return_value = [sample_summary]

        await dashboard.get_score_trends(
            days=30,
            agent_filter="portfolio",
        )

        mock_storage.list_summaries.assert_called_once()
        call_kwargs = mock_storage.list_summaries.call_args[1]
        assert call_kwargs["agent_filter"] == "portfolio"

    @pytest.mark.asyncio
    async def test_get_score_distribution_empty(self, dashboard, mock_storage):
        """Test distribution with no data."""
        mock_storage.list_summaries.return_value = []

        result = await dashboard.get_score_distribution(
            dimension=EvaluationDimension.ACCURACY,
            days=30,
        )

        assert result["data"] == []
        assert result["metadata"]["count"] == 0

    @pytest.mark.asyncio
    async def test_get_score_distribution_with_data(self, dashboard, mock_storage, sample_summary):
        """Test distribution with data."""
        mock_storage.list_summaries.return_value = [sample_summary]

        result = await dashboard.get_score_distribution(
            dimension=EvaluationDimension.ACCURACY,
            days=30,
            bins=10,
        )

        assert len(result["labels"]) == 10
        assert len(result["data"]) == 10
        assert result["metadata"]["dimension"] == "accuracy"

    @pytest.mark.asyncio
    async def test_get_agent_comparison_empty(self, dashboard, mock_storage):
        """Test agent comparison with no data."""
        mock_storage.list_summaries.return_value = []

        result = await dashboard.get_agent_comparison(days=30)

        assert result["labels"] == []
        assert result["datasets"] == []

    @pytest.mark.asyncio
    async def test_get_agent_comparison_with_data(self, dashboard, mock_storage, sample_summary):
        """Test agent comparison with data."""
        mock_storage.list_summaries.return_value = [sample_summary]

        result = await dashboard.get_agent_comparison(days=30)

        assert "portfolio_agent" in result["labels"]
        assert len(result["datasets"]) > 0

    @pytest.mark.asyncio
    async def test_get_cost_quality_tradeoff(self, dashboard, mock_storage, sample_summary):
        """Test cost vs quality data."""
        mock_storage.list_summaries.return_value = [sample_summary]

        result = await dashboard.get_cost_quality_tradeoff(days=30)

        assert len(result["datasets"]) > 0
        assert result["metadata"]["total_runs"] == 1

    @pytest.mark.asyncio
    async def test_get_regression_alerts_no_data(self, dashboard, mock_storage):
        """Test regression alerts with insufficient data."""
        mock_storage.list_summaries.return_value = []

        alerts = await dashboard.get_regression_alerts(days=7)

        assert alerts == []

    @pytest.mark.asyncio
    async def test_get_regression_alerts_no_regression(self, dashboard, mock_storage, sample_summary):
        """Test regression alerts when no regression exists."""
        # Two summaries with same scores
        summary2 = sample_summary.model_copy()
        summary2.run_id = "test-run-002"
        summary2.started_at = datetime.utcnow()

        mock_storage.list_summaries.return_value = [sample_summary, summary2]

        alerts = await dashboard.get_regression_alerts(days=7, threshold=0.05)

        assert alerts == []

    @pytest.mark.asyncio
    async def test_get_regression_alerts_with_regression(self, dashboard, mock_storage, sample_summary):
        """Test regression alerts when regression exists."""
        # Create second summary with lower scores
        config2 = EvaluationRunConfig(run_id="test-run-002")
        summary2 = EvaluationRunSummary(
            run_id="test-run-002",
            config=config2,
            total_cases=10,
            passed_cases=6,
            failed_cases=4,
            pass_rate=0.6,
            mean_scores={
                "accuracy": 0.70,  # Dropped from 0.85
                "relevance": 0.90,
                "aggregate": 0.80,
            },
            std_scores={},
            started_at=datetime.utcnow(),
            results=[],
        )

        mock_storage.list_summaries.return_value = [sample_summary, summary2]

        alerts = await dashboard.get_regression_alerts(days=7, threshold=0.05)

        # Should detect accuracy regression
        accuracy_alerts = [a for a in alerts if a["dimension"] == "accuracy"]
        assert len(accuracy_alerts) == 1
        assert accuracy_alerts[0]["drop"] == pytest.approx(0.15, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_dashboard_overview_empty(self, dashboard, mock_storage):
        """Test overview with no data."""
        mock_storage.list_summaries.return_value = []

        overview = await dashboard.get_dashboard_overview(days=7)

        assert overview["total_runs"] == 0
        assert overview["total_cases"] == 0
        assert overview["overall_pass_rate"] == 0

    @pytest.mark.asyncio
    async def test_get_dashboard_overview_with_data(self, dashboard, mock_storage, sample_summary):
        """Test overview with data."""
        mock_storage.list_summaries.return_value = [sample_summary]

        overview = await dashboard.get_dashboard_overview(days=7)

        assert overview["total_runs"] == 1
        assert overview["total_cases"] == 10
        assert overview["overall_pass_rate"] == 0.8
        assert overview["total_cost_usd"] == 0.05


class TestABTestFormatting:
    """Tests for A/B test result formatting."""

    @pytest.fixture
    def sample_ab_result(self, sample_summary) -> ABTestResult:
        """Create sample A/B test result."""
        control_config = EvaluationRunConfig(run_id="control-001")
        treatment_config = EvaluationRunConfig(run_id="treatment-001")

        config = ABTestConfig(
            name="Test A/B",
            description="Testing prompt changes",
            control=control_config,
            treatment=treatment_config,
        )

        control_summary = sample_summary.model_copy()
        control_summary.run_id = "control-001"

        treatment_summary = sample_summary.model_copy()
        treatment_summary.run_id = "treatment-001"
        treatment_summary.mean_scores = {
            "accuracy": 0.90,
            "relevance": 0.92,
            "aggregate": 0.91,
        }

        return ABTestResult(
            config=config,
            control_summary=control_summary,
            treatment_summary=treatment_summary,
            dimension_comparisons={
                "accuracy": {
                    "control_mean": 0.85,
                    "control_std": 0.1,
                    "treatment_mean": 0.90,
                    "treatment_std": 0.08,
                    "p_value": 0.03,
                    "effect_size": 0.45,
                    "is_significant": True,
                    "winner": "treatment",
                },
            },
            winner="treatment",
            is_significant=True,
            p_value=0.03,
            effect_size=0.45,
            recommendation="Adopt treatment variant",
        )

    @pytest.mark.asyncio
    async def test_get_ab_test_summary(self, dashboard, mock_storage, sample_ab_result):
        """Test A/B test summary retrieval."""
        mock_storage.get_ab_test.return_value = sample_ab_result

        result = await dashboard.get_ab_test_summary("test-001")

        assert result is not None
        assert result["test_name"] == "Test A/B"
        assert result["winner"] == "treatment"
        assert result["is_significant"] is True
        assert len(result["dimensions"]) == 1

    @pytest.mark.asyncio
    async def test_get_ab_test_not_found(self, dashboard, mock_storage):
        """Test A/B test not found."""
        mock_storage.get_ab_test.return_value = None

        result = await dashboard.get_ab_test_summary("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_ab_tests(self, dashboard, mock_storage, sample_ab_result):
        """Test listing recent A/B tests."""
        mock_storage.list_ab_tests.return_value = [sample_ab_result]

        results = await dashboard.get_recent_ab_tests(limit=10)

        assert len(results) == 1
        assert results[0]["test_name"] == "Test A/B"

    def test_format_ab_test_result_confidence_intervals(self, dashboard, sample_ab_result):
        """Test that confidence intervals are calculated correctly."""
        formatted = dashboard._format_ab_test_result(sample_ab_result)

        accuracy_dim = next(
            d for d in formatted["dimensions"] if d["dimension"] == "accuracy"
        )

        # Check control CI
        assert accuracy_dim["control"]["mean"] == 0.85
        assert accuracy_dim["control"]["ci_lower"] < 0.85
        assert accuracy_dim["control"]["ci_upper"] > 0.85

        # Check treatment CI
        assert accuracy_dim["treatment"]["mean"] == 0.90
        assert accuracy_dim["treatment"]["ci_lower"] < 0.90
        assert accuracy_dim["treatment"]["ci_upper"] > 0.90
