"""Tests for statistical analysis functions."""


from src.evaluation.models import (
    ABTestConfig,
    DimensionScore,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
)
from src.evaluation.statistics import (
    StatisticalAnalyzer,
    calculate_cohens_d,
    calculate_mean,
    calculate_p_value,
    calculate_standard_error,
    calculate_std,
    interpret_effect_size,
    welch_t_test,
)


class TestBasicStatistics:
    """Tests for basic statistical functions."""

    def test_calculate_mean_empty(self) -> None:
        """Test mean of empty list."""
        assert calculate_mean([]) == 0.0

    def test_calculate_mean_single(self) -> None:
        """Test mean of single value."""
        assert calculate_mean([5.0]) == 5.0

    def test_calculate_mean_multiple(self) -> None:
        """Test mean of multiple values."""
        assert calculate_mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_calculate_std_empty(self) -> None:
        """Test std of empty list."""
        assert calculate_std([]) == 0.0

    def test_calculate_std_single(self) -> None:
        """Test std of single value."""
        assert calculate_std([5.0]) == 0.0

    def test_calculate_std_multiple(self) -> None:
        """Test std of multiple values."""
        # Known values: [2, 4, 4, 4, 5, 5, 7, 9]
        # Mean = 5, Variance = 4, Std = 2
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        std = calculate_std(values)
        assert abs(std - 2.138) < 0.01  # Sample std with n-1

    def test_calculate_std_with_mean(self) -> None:
        """Test std with pre-calculated mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = 3.0
        std = calculate_std(values, mean)
        assert std > 0

    def test_calculate_standard_error(self) -> None:
        """Test standard error calculation."""
        std = 10.0
        n = 100
        se = calculate_standard_error(std, n)
        assert se == 1.0  # 10 / sqrt(100)

    def test_calculate_standard_error_zero_n(self) -> None:
        """Test standard error with n=0."""
        assert calculate_standard_error(10.0, 0) == 0.0


class TestWelchTTest:
    """Tests for Welch's t-test."""

    def test_equal_means(self) -> None:
        """Test t-test with equal means."""
        t_stat, df = welch_t_test(
            mean1=5.0, std1=1.0, n1=30,
            mean2=5.0, std2=1.0, n2=30,
        )
        assert abs(t_stat) < 0.01
        assert df > 0

    def test_different_means(self) -> None:
        """Test t-test with different means."""
        t_stat, df = welch_t_test(
            mean1=5.0, std1=1.0, n1=30,
            mean2=6.0, std2=1.0, n2=30,
        )
        assert t_stat < 0  # mean1 < mean2
        assert df > 0

    def test_insufficient_samples(self) -> None:
        """Test t-test with insufficient samples."""
        t_stat, df = welch_t_test(
            mean1=5.0, std1=1.0, n1=1,
            mean2=6.0, std2=1.0, n2=1,
        )
        assert t_stat == 0.0
        assert df == 0.0


class TestPValue:
    """Tests for p-value calculation."""

    def test_p_value_equal_means(self) -> None:
        """Test p-value with equal means is high."""
        p = calculate_p_value(
            mean1=5.0, std1=1.0, n1=30,
            mean2=5.0, std2=1.0, n2=30,
        )
        # Should be close to 1 (no significant difference)
        assert p > 0.5

    def test_p_value_very_different_means(self) -> None:
        """Test p-value with very different means is low."""
        p = calculate_p_value(
            mean1=1.0, std1=0.5, n1=100,
            mean2=5.0, std2=0.5, n2=100,
        )
        # Should be very small (highly significant)
        assert p < 0.05


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_cohens_d_equal_means(self) -> None:
        """Test Cohen's d with equal means."""
        d = calculate_cohens_d(
            mean1=5.0, std1=1.0, n1=30,
            mean2=5.0, std2=1.0, n2=30,
        )
        assert abs(d) < 0.01

    def test_cohens_d_different_means(self) -> None:
        """Test Cohen's d with different means."""
        d = calculate_cohens_d(
            mean1=6.0, std1=1.0, n1=30,
            mean2=5.0, std2=1.0, n2=30,
        )
        assert d > 0  # mean1 > mean2

    def test_cohens_d_large_effect(self) -> None:
        """Test Cohen's d with large effect."""
        d = calculate_cohens_d(
            mean1=10.0, std1=1.0, n1=30,
            mean2=5.0, std2=1.0, n2=30,
        )
        assert abs(d) > 0.8  # Large effect


class TestEffectSizeInterpretation:
    """Tests for effect size interpretation."""

    def test_negligible(self) -> None:
        """Test negligible effect size."""
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(-0.1) == "negligible"

    def test_small(self) -> None:
        """Test small effect size."""
        assert interpret_effect_size(0.3) == "small"
        assert interpret_effect_size(-0.3) == "small"

    def test_medium(self) -> None:
        """Test medium effect size."""
        assert interpret_effect_size(0.6) == "medium"
        assert interpret_effect_size(-0.6) == "medium"

    def test_large(self) -> None:
        """Test large effect size."""
        assert interpret_effect_size(1.0) == "large"
        assert interpret_effect_size(-1.0) == "large"


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""

    def test_calculate_run_statistics_empty(self) -> None:
        """Test statistics with no results."""
        analyzer = StatisticalAnalyzer()
        stats = analyzer.calculate_run_statistics([])
        assert stats == {}

    def test_calculate_run_statistics(self) -> None:
        """Test statistics calculation."""
        analyzer = StatisticalAnalyzer()

        results = [
            EvaluationResult(
                case_id="case-1",
                run_id="run-1",
                model="test",
                scores=[
                    DimensionScore(dimension=EvaluationDimension.ACCURACY, score=0.8),
                    DimensionScore(dimension=EvaluationDimension.RELEVANCE, score=0.9),
                ],
                aggregate_score=0.85,
            ),
            EvaluationResult(
                case_id="case-2",
                run_id="run-1",
                model="test",
                scores=[
                    DimensionScore(dimension=EvaluationDimension.ACCURACY, score=0.7),
                    DimensionScore(dimension=EvaluationDimension.RELEVANCE, score=0.85),
                ],
                aggregate_score=0.775,
            ),
        ]

        stats = analyzer.calculate_run_statistics(results)

        assert "accuracy" in stats
        assert "relevance" in stats
        assert "aggregate" in stats

        assert stats["accuracy"]["mean"] == 0.75
        assert stats["accuracy"]["count"] == 2
        assert stats["relevance"]["mean"] == 0.875

    def test_compare_runs(self) -> None:
        """Test run comparison."""
        analyzer = StatisticalAnalyzer()

        control_config = EvaluationRunConfig(run_id="control")
        treatment_config = EvaluationRunConfig(run_id="treatment")

        control = EvaluationRunSummary(
            run_id="control",
            config=control_config,
            total_cases=50,
            passed_cases=40,
            failed_cases=10,
            pass_rate=0.8,
            mean_scores={"accuracy": 0.75, "relevance": 0.80},
            std_scores={"accuracy": 0.1, "relevance": 0.1},
        )

        treatment = EvaluationRunSummary(
            run_id="treatment",
            config=treatment_config,
            total_cases=50,
            passed_cases=45,
            failed_cases=5,
            pass_rate=0.9,
            mean_scores={"accuracy": 0.85, "relevance": 0.85},
            std_scores={"accuracy": 0.1, "relevance": 0.1},
        )

        comparisons = analyzer.compare_runs(control, treatment, confidence_level=0.95)

        assert "accuracy" in comparisons
        assert "relevance" in comparisons

        # Treatment should be better
        assert comparisons["accuracy"]["difference"] > 0
        assert comparisons["accuracy"]["treatment_mean"] > comparisons["accuracy"]["control_mean"]

    def test_analyze_ab_test(self) -> None:
        """Test full A/B test analysis."""
        analyzer = StatisticalAnalyzer()

        control_config = EvaluationRunConfig(run_id="control")
        treatment_config = EvaluationRunConfig(run_id="treatment")

        config = ABTestConfig(
            name="Test Experiment",
            control=control_config,
            treatment=treatment_config,
            min_samples=30,
            confidence_level=0.95,
        )

        control_summary = EvaluationRunSummary(
            run_id="control",
            config=control_config,
            total_cases=50,
            passed_cases=40,
            failed_cases=10,
            pass_rate=0.8,
            mean_scores={"accuracy": 0.75, "aggregate": 0.75},
            std_scores={"accuracy": 0.15, "aggregate": 0.15},
        )

        treatment_summary = EvaluationRunSummary(
            run_id="treatment",
            config=treatment_config,
            total_cases=50,
            passed_cases=45,
            failed_cases=5,
            pass_rate=0.9,
            mean_scores={"accuracy": 0.90, "aggregate": 0.90},
            std_scores={"accuracy": 0.1, "aggregate": 0.1},
        )

        result = analyzer.analyze_ab_test(config, control_summary, treatment_summary)

        assert result.config == config
        assert result.control_summary == control_summary
        assert result.treatment_summary == treatment_summary
        assert "accuracy" in result.dimension_comparisons
        assert result.recommendation  # Should have a recommendation

    def test_detect_regression(self) -> None:
        """Test regression detection."""
        analyzer = StatisticalAnalyzer()

        baseline_config = EvaluationRunConfig(run_id="baseline")
        current_config = EvaluationRunConfig(run_id="current")

        baseline = EvaluationRunSummary(
            run_id="baseline",
            config=baseline_config,
            total_cases=100,
            passed_cases=90,
            failed_cases=10,
            pass_rate=0.9,
            mean_scores={"accuracy": 0.85, "relevance": 0.90},
            std_scores={"accuracy": 0.1, "relevance": 0.08},
        )

        # Current has regression in accuracy
        current = EvaluationRunSummary(
            run_id="current",
            config=current_config,
            total_cases=100,
            passed_cases=75,
            failed_cases=25,
            pass_rate=0.75,
            mean_scores={"accuracy": 0.70, "relevance": 0.92},  # Accuracy dropped
            std_scores={"accuracy": 0.1, "relevance": 0.08},
        )

        result = analyzer.detect_regression(
            baseline=baseline,
            current=current,
            threshold=0.05,
            confidence_level=0.95,
        )

        assert "has_regression" in result
        assert "regressions" in result
        assert "improvements" in result
        assert "comparisons" in result
