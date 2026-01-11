"""Tests for evaluation storage."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.evaluation.models import (
    ABTestConfig,
    ABTestResult,
    DimensionScore,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
)
from src.evaluation.storage import EvaluationStorage


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def storage(temp_db):
    """Create storage with temp database."""
    return EvaluationStorage(db_path=temp_db)


@pytest.fixture
def sample_config() -> EvaluationRunConfig:
    """Create a sample config."""
    return EvaluationRunConfig(
        run_id="test-run-001",
        name="Test Run",
        model="claude-sonnet-4-20250514",
        metadata={"agent_name": "test_agent"},
    )


@pytest.fixture
def sample_result() -> EvaluationResult:
    """Create a sample result."""
    return EvaluationResult(
        case_id="case-001",
        run_id="test-run-001",
        model="claude-sonnet-4-20250514",
        scores=[
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=0.9,
                reasoning="Good accuracy",
            ),
            DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=0.85,
                reasoning="Relevant response",
            ),
        ],
        aggregate_score=0.875,
        passed=True,
        latency_ms=150,
        cost_usd=0.01,
    )


@pytest.fixture
def sample_summary(sample_config, sample_result) -> EvaluationRunSummary:
    """Create a sample summary."""
    return EvaluationRunSummary(
        run_id="test-run-001",
        config=sample_config,
        total_cases=5,
        passed_cases=4,
        failed_cases=1,
        pass_rate=0.8,
        mean_scores={"accuracy": 0.85, "relevance": 0.9, "aggregate": 0.875},
        std_scores={"accuracy": 0.1, "relevance": 0.05, "aggregate": 0.08},
        total_latency_ms=1000,
        mean_latency_ms=200.0,
        total_cost_usd=0.05,
        total_input_tokens=5000,
        total_output_tokens=2000,
        started_at=datetime.utcnow() - timedelta(hours=1),
        completed_at=datetime.utcnow(),
        duration_seconds=60.0,
        results=[sample_result],
    )


class TestEvaluationStorage:
    """Tests for EvaluationStorage class."""

    @pytest.mark.asyncio
    async def test_initialization(self, storage, temp_db):
        """Test storage initializes correctly."""
        await storage._ensure_initialized()
        assert storage._initialized
        assert temp_db.exists()

    @pytest.mark.asyncio
    async def test_save_and_get_summary(self, storage, sample_summary):
        """Test saving and retrieving a summary."""
        await storage.save_summary(sample_summary)

        retrieved = await storage.get_summary("test-run-001")

        assert retrieved is not None
        assert retrieved.run_id == sample_summary.run_id
        assert retrieved.total_cases == sample_summary.total_cases
        assert retrieved.pass_rate == sample_summary.pass_rate
        assert len(retrieved.results) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_summary(self, storage):
        """Test getting a non-existent summary."""
        result = await storage.get_summary("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_summaries_empty(self, storage):
        """Test listing with no data."""
        summaries = await storage.list_summaries()
        assert summaries == []

    @pytest.mark.asyncio
    async def test_list_summaries_with_data(self, storage, sample_summary):
        """Test listing summaries."""
        await storage.save_summary(sample_summary)

        summaries = await storage.list_summaries()

        assert len(summaries) == 1
        assert summaries[0].run_id == "test-run-001"

    @pytest.mark.asyncio
    async def test_list_summaries_with_date_filter(self, storage, sample_summary):
        """Test listing with date filter."""
        await storage.save_summary(sample_summary)

        # Filter for recent only
        since = datetime.utcnow() - timedelta(days=1)
        summaries = await storage.list_summaries(since=since)

        assert len(summaries) == 1

        # Filter for future date
        since_future = datetime.utcnow() + timedelta(days=1)
        summaries = await storage.list_summaries(since=since_future)

        assert len(summaries) == 0

    @pytest.mark.asyncio
    async def test_list_summaries_with_agent_filter(self, storage, sample_summary):
        """Test listing with agent filter."""
        await storage.save_summary(sample_summary)

        # Matching filter
        summaries = await storage.list_summaries(agent_filter="test")
        assert len(summaries) == 1

        # Non-matching filter
        summaries = await storage.list_summaries(agent_filter="other_agent")
        assert len(summaries) == 0

    @pytest.mark.asyncio
    async def test_list_summaries_with_limit(self, storage):
        """Test listing with limit."""
        # Create multiple summaries
        for i in range(5):
            config = EvaluationRunConfig(
                run_id=f"test-run-{i:03d}",
                name=f"Test Run {i}",
            )
            summary = EvaluationRunSummary(
                run_id=f"test-run-{i:03d}",
                config=config,
                total_cases=5,
                passed_cases=4,
                failed_cases=1,
                pass_rate=0.8,
                mean_scores={},
                std_scores={},
                started_at=datetime.utcnow() - timedelta(hours=i),
                results=[],
            )
            await storage.save_summary(summary)

        summaries = await storage.list_summaries(limit=3)
        assert len(summaries) == 3

    @pytest.mark.asyncio
    async def test_update_existing_summary(self, storage, sample_summary):
        """Test updating an existing summary."""
        await storage.save_summary(sample_summary)

        # Update the summary
        sample_summary.passed_cases = 5
        sample_summary.pass_rate = 1.0
        await storage.save_summary(sample_summary)

        retrieved = await storage.get_summary("test-run-001")
        assert retrieved.passed_cases == 5
        assert retrieved.pass_rate == 1.0


class TestABTestStorage:
    """Tests for A/B test storage."""

    @pytest.fixture
    def sample_ab_result(self, sample_summary) -> ABTestResult:
        """Create sample A/B test result."""
        control_config = EvaluationRunConfig(run_id="control-001")
        treatment_config = EvaluationRunConfig(run_id="treatment-001")

        config = ABTestConfig(
            name="Test A/B",
            control=control_config,
            treatment=treatment_config,
        )

        control_summary = sample_summary.model_copy()
        control_summary.run_id = "control-001"
        control_summary.config = control_config

        treatment_summary = sample_summary.model_copy()
        treatment_summary.run_id = "treatment-001"
        treatment_summary.config = treatment_config

        return ABTestResult(
            config=config,
            control_summary=control_summary,
            treatment_summary=treatment_summary,
            dimension_comparisons={"accuracy": {"p_value": 0.05}},
            winner="treatment",
            is_significant=True,
            p_value=0.03,
            effect_size=0.5,
            recommendation="Adopt treatment",
        )

    @pytest.mark.asyncio
    async def test_save_and_get_ab_test(self, storage, sample_ab_result):
        """Test saving and retrieving A/B test."""
        await storage.save_ab_test(sample_ab_result, "ab-test-001")

        retrieved = await storage.get_ab_test("ab-test-001")

        assert retrieved is not None
        assert retrieved.config.name == "Test A/B"
        assert retrieved.winner == "treatment"
        assert retrieved.is_significant is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_ab_test(self, storage):
        """Test getting non-existent A/B test."""
        result = await storage.get_ab_test("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_ab_tests(self, storage, sample_ab_result):
        """Test listing A/B tests."""
        await storage.save_ab_test(sample_ab_result, "ab-test-001")

        tests = await storage.list_ab_tests(limit=10)

        assert len(tests) == 1
        assert tests[0].config.name == "Test A/B"


class TestBaselineRetrieval:
    """Tests for baseline summary retrieval."""

    @pytest.mark.asyncio
    async def test_get_baseline_no_data(self, storage):
        """Test baseline with no data."""
        baseline = await storage.get_baseline_summary()
        assert baseline is None

    @pytest.mark.asyncio
    async def test_get_baseline_with_passing_run(self, storage, sample_summary):
        """Test getting baseline when passing runs exist."""
        await storage.save_summary(sample_summary)

        baseline = await storage.get_baseline_summary()

        assert baseline is not None
        assert baseline.run_id == "test-run-001"

    @pytest.mark.asyncio
    async def test_get_baseline_with_agent_filter(self, storage, sample_summary):
        """Test baseline with agent filter."""
        await storage.save_summary(sample_summary)

        # Matching filter
        baseline = await storage.get_baseline_summary(agent_name="test")
        assert baseline is not None

        # Non-matching filter
        baseline = await storage.get_baseline_summary(agent_name="other")
        assert baseline is None


class TestDataCleanup:
    """Tests for old data cleanup."""

    @pytest.mark.asyncio
    async def test_delete_old_data(self, storage, sample_config):
        """Test deleting old data."""
        # Create an old summary
        old_summary = EvaluationRunSummary(
            run_id="old-run",
            config=sample_config,
            total_cases=5,
            passed_cases=4,
            failed_cases=1,
            pass_rate=0.8,
            mean_scores={},
            std_scores={},
            started_at=datetime.utcnow() - timedelta(days=100),
            results=[],
        )
        await storage.save_summary(old_summary)

        # Create a recent summary
        recent_config = EvaluationRunConfig(run_id="recent-run")
        recent_summary = EvaluationRunSummary(
            run_id="recent-run",
            config=recent_config,
            total_cases=5,
            passed_cases=4,
            failed_cases=1,
            pass_rate=0.8,
            mean_scores={},
            std_scores={},
            started_at=datetime.utcnow() - timedelta(days=10),
            results=[],
        )
        await storage.save_summary(recent_summary)

        # Delete data older than 90 days
        deleted = await storage.delete_old_data(days=90)

        assert deleted >= 1

        # Verify old is gone, recent remains
        old = await storage.get_summary("old-run")
        recent = await storage.get_summary("recent-run")

        assert old is None
        assert recent is not None
