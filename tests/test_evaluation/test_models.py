"""Tests for evaluation data models."""

from datetime import datetime

import pytest

from src.evaluation.models import (
    ABTestConfig,
    ABTestResult,
    DimensionScore,
    EvaluationCase,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
    EvaluationType,
)


class TestEvaluationDimension:
    """Tests for EvaluationDimension enum."""

    def test_all_dimensions_exist(self) -> None:
        """Test that all expected dimensions are defined."""
        expected = {"accuracy", "relevance", "completeness", "consistency", "safety", "actionability"}
        actual = {d.value for d in EvaluationDimension}
        assert actual == expected

    def test_dimension_string_values(self) -> None:
        """Test that dimensions have correct string values."""
        assert EvaluationDimension.ACCURACY.value == "accuracy"
        assert EvaluationDimension.RELEVANCE.value == "relevance"
        assert EvaluationDimension.COMPLETENESS.value == "completeness"


class TestEvaluationType:
    """Tests for EvaluationType enum."""

    def test_all_types_exist(self) -> None:
        """Test that all expected evaluation types are defined."""
        expected = {"golden_set", "ab_test", "regression", "production_sample"}
        actual = {t.value for t in EvaluationType}
        assert actual == expected


class TestEvaluationCase:
    """Tests for EvaluationCase model."""

    def test_minimal_case(self) -> None:
        """Test creating case with minimal required fields."""
        case = EvaluationCase(
            id="test-001",
            name="Test Case",
            input={"query": "test query"},
        )
        assert case.id == "test-001"
        assert case.name == "Test Case"
        assert case.input == {"query": "test query"}
        assert case.expected_output is None
        assert len(case.evaluation_criteria) == 4  # Default criteria

    def test_full_case(self) -> None:
        """Test creating case with all fields."""
        case = EvaluationCase(
            id="test-002",
            name="Full Test Case",
            description="A complete test case",
            input={"query": "analyze portfolio", "portfolio": {"AAPL": 100}},
            expected_output={"recommendation": "hold"},
            evaluation_criteria=[EvaluationDimension.ACCURACY, EvaluationDimension.SAFETY],
            tags=["portfolio", "integration"],
            evaluation_type=EvaluationType.GOLDEN_SET,
            metadata={"priority": "high"},
        )
        assert case.description == "A complete test case"
        assert case.expected_output == {"recommendation": "hold"}
        assert len(case.evaluation_criteria) == 2
        assert "portfolio" in case.tags
        assert case.evaluation_type == EvaluationType.GOLDEN_SET

    def test_default_criteria(self) -> None:
        """Test that default criteria are applied."""
        case = EvaluationCase(id="test", name="Test", input={})
        default_dims = set(case.evaluation_criteria)
        assert EvaluationDimension.ACCURACY in default_dims
        assert EvaluationDimension.RELEVANCE in default_dims
        assert EvaluationDimension.COMPLETENESS in default_dims
        assert EvaluationDimension.ACTIONABILITY in default_dims


class TestDimensionScore:
    """Tests for DimensionScore model."""

    def test_valid_score(self) -> None:
        """Test creating a valid dimension score."""
        score = DimensionScore(
            dimension=EvaluationDimension.ACCURACY,
            score=0.85,
            reasoning="Good accuracy",
            confidence=0.9,
        )
        assert score.dimension == EvaluationDimension.ACCURACY
        assert score.score == 0.85
        assert score.reasoning == "Good accuracy"
        assert score.confidence == 0.9

    def test_score_bounds(self) -> None:
        """Test that scores are bounded 0-1."""
        with pytest.raises(ValueError):
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=1.5,  # Invalid
            )

        with pytest.raises(ValueError):
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=-0.1,  # Invalid
            )

    def test_confidence_bounds(self) -> None:
        """Test that confidence is bounded 0-1."""
        with pytest.raises(ValueError):
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=0.5,
                confidence=1.5,  # Invalid
            )


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_minimal_result(self) -> None:
        """Test creating result with minimal fields."""
        result = EvaluationResult(
            case_id="test-001",
            run_id="run-001",
            model="claude-sonnet-4-20250514",
        )
        assert result.case_id == "test-001"
        assert result.run_id == "run-001"
        assert result.passed is True
        assert result.aggregate_score == 0.0

    def test_get_score(self) -> None:
        """Test getting score for a specific dimension."""
        result = EvaluationResult(
            case_id="test",
            run_id="run",
            model="claude-sonnet-4-20250514",
            scores=[
                DimensionScore(
                    dimension=EvaluationDimension.ACCURACY,
                    score=0.9,
                ),
                DimensionScore(
                    dimension=EvaluationDimension.RELEVANCE,
                    score=0.8,
                ),
            ],
        )
        assert result.get_score(EvaluationDimension.ACCURACY) == 0.9
        assert result.get_score(EvaluationDimension.RELEVANCE) == 0.8
        assert result.get_score(EvaluationDimension.SAFETY) is None

    def test_timestamp_auto_set(self) -> None:
        """Test that timestamp is automatically set."""
        before = datetime.utcnow()
        result = EvaluationResult(
            case_id="test",
            run_id="run",
            model="claude-sonnet-4-20250514",
        )
        after = datetime.utcnow()
        assert before <= result.timestamp <= after


class TestEvaluationRunConfig:
    """Tests for EvaluationRunConfig model."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = EvaluationRunConfig(run_id="run-001")
        assert config.model == "claude-sonnet-4-20250514"
        assert config.judge_model == "claude-sonnet-4-20250514"
        assert config.parallel_workers == 5
        assert config.timeout_seconds == 120
        assert "accuracy" in config.score_thresholds
        assert config.tags_filter is None

    def test_custom_thresholds(self) -> None:
        """Test custom score thresholds."""
        config = EvaluationRunConfig(
            run_id="run-002",
            score_thresholds={
                "accuracy": 0.9,
                "aggregate": 0.85,
            },
        )
        assert config.score_thresholds["accuracy"] == 0.9
        assert config.score_thresholds["aggregate"] == 0.85

    def test_worker_bounds(self) -> None:
        """Test parallel worker bounds."""
        with pytest.raises(ValueError):
            EvaluationRunConfig(run_id="test", parallel_workers=0)

        with pytest.raises(ValueError):
            EvaluationRunConfig(run_id="test", parallel_workers=25)


class TestEvaluationRunSummary:
    """Tests for EvaluationRunSummary model."""

    def test_summary_creation(self) -> None:
        """Test creating a run summary."""
        config = EvaluationRunConfig(run_id="run-001")
        summary = EvaluationRunSummary(
            run_id="run-001",
            config=config,
            total_cases=10,
            passed_cases=8,
            failed_cases=2,
            pass_rate=0.8,
            mean_scores={"accuracy": 0.85, "relevance": 0.9},
            total_latency_ms=5000,
            mean_latency_ms=500.0,
        )
        assert summary.total_cases == 10
        assert summary.pass_rate == 0.8
        assert summary.mean_scores["accuracy"] == 0.85


class TestABTestConfig:
    """Tests for ABTestConfig model."""

    def test_ab_config_creation(self) -> None:
        """Test creating A/B test configuration."""
        control = EvaluationRunConfig(run_id="control-001")
        treatment = EvaluationRunConfig(
            run_id="treatment-001",
            prompt_version="v2",
        )

        config = ABTestConfig(
            name="Prompt V2 Test",
            description="Testing new prompt version",
            control=control,
            treatment=treatment,
            min_samples=50,
            confidence_level=0.95,
        )
        assert config.name == "Prompt V2 Test"
        assert config.control.run_id == "control-001"
        assert config.treatment.run_id == "treatment-001"
        assert config.min_samples == 50
        assert config.confidence_level == 0.95


class TestABTestResult:
    """Tests for ABTestResult model."""

    def test_ab_result_creation(self) -> None:
        """Test creating A/B test result."""
        control_config = EvaluationRunConfig(run_id="control-001")
        treatment_config = EvaluationRunConfig(run_id="treatment-001")

        config = ABTestConfig(
            name="Test",
            control=control_config,
            treatment=treatment_config,
        )

        control_summary = EvaluationRunSummary(
            run_id="control-001",
            config=control_config,
            total_cases=30,
            passed_cases=25,
            failed_cases=5,
            pass_rate=0.833,
        )

        treatment_summary = EvaluationRunSummary(
            run_id="treatment-001",
            config=treatment_config,
            total_cases=30,
            passed_cases=28,
            failed_cases=2,
            pass_rate=0.933,
        )

        result = ABTestResult(
            config=config,
            control_summary=control_summary,
            treatment_summary=treatment_summary,
            winner="treatment",
            is_significant=True,
            p_value=0.03,
            effect_size=0.45,
            recommendation="Adopt treatment variant",
        )
        assert result.winner == "treatment"
        assert result.is_significant is True
        assert result.p_value == 0.03
