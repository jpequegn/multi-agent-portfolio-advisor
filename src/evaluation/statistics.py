"""Statistical analysis for evaluation results.

This module provides statistical significance testing for A/B tests
and regression detection for quality monitoring.
"""

import math
from typing import Any

import structlog

from src.evaluation.models import (
    ABTestConfig,
    ABTestResult,
    EvaluationResult,
    EvaluationRunSummary,
)

logger = structlog.get_logger(__name__)


def calculate_mean(values: list[float]) -> float:
    """Calculate the arithmetic mean of a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Mean value, or 0.0 if empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_std(values: list[float], mean: float | None = None) -> float:
    """Calculate the standard deviation of a list of values.

    Args:
        values: List of numeric values.
        mean: Pre-calculated mean (computed if not provided).

    Returns:
        Standard deviation, or 0.0 if insufficient data.
    """
    if len(values) < 2:
        return 0.0

    if mean is None:
        mean = calculate_mean(values)

    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_standard_error(std: float, n: int) -> float:
    """Calculate the standard error of the mean.

    Args:
        std: Standard deviation.
        n: Sample size.

    Returns:
        Standard error, or 0.0 if n is 0.
    """
    if n == 0:
        return 0.0
    return std / math.sqrt(n)


def welch_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> tuple[float, float]:
    """Perform Welch's t-test for unequal variances.

    Args:
        mean1: Mean of first sample.
        std1: Standard deviation of first sample.
        n1: Size of first sample.
        mean2: Mean of second sample.
        std2: Standard deviation of second sample.
        n2: Size of second sample.

    Returns:
        Tuple of (t-statistic, degrees of freedom).
    """
    if n1 < 2 or n2 < 2:
        return 0.0, 0.0

    # Standard errors
    se1 = (std1 ** 2) / n1
    se2 = (std2 ** 2) / n2

    # Pooled standard error
    se_diff = math.sqrt(se1 + se2)

    if se_diff == 0:
        return 0.0, 0.0

    # t-statistic
    t_stat = (mean1 - mean2) / se_diff

    # Welch-Satterthwaite degrees of freedom
    numerator = (se1 + se2) ** 2
    denominator = (se1 ** 2) / (n1 - 1) + (se2 ** 2) / (n2 - 1)

    df = min(n1, n2) - 1 if denominator == 0 else numerator / denominator

    return t_stat, df


def t_distribution_cdf(t: float, df: float) -> float:
    """Approximate the CDF of the t-distribution.

    Uses a simple approximation suitable for most practical purposes.

    Args:
        t: t-statistic.
        df: Degrees of freedom.

    Returns:
        Approximate p-value (two-tailed).
    """
    if df <= 0:
        return 1.0

    # Use normal approximation for large df
    if df > 100:
        # Standard normal approximation
        return 2 * (1 - _normal_cdf(abs(t)))

    # Simple approximation using the relationship with the beta function
    # This is a rough approximation but works reasonably well
    x = df / (df + t ** 2)
    p = _incomplete_beta(df / 2, 0.5, x)
    return p


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function approximation.

    Args:
        x: Value to evaluate.

    Returns:
        Approximate CDF value.
    """
    # Approximation using tanh
    return 0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Rough approximation of incomplete beta function.

    Args:
        a: First shape parameter.
        b: Second shape parameter.
        x: Value to evaluate (0 <= x <= 1).

    Returns:
        Approximate incomplete beta value.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Simple numerical approximation using trapezoidal rule
    n_steps = 100
    h = x / n_steps
    result = 0.0

    for i in range(n_steps):
        t = i * h + h / 2
        if 0 < t < 1:
            integrand = (t ** (a - 1)) * ((1 - t) ** (b - 1))
            result += integrand * h

    # Normalize (approximate)
    beta_ab = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return result / beta_ab


def calculate_p_value(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> float:
    """Calculate p-value for difference between two samples.

    Args:
        mean1: Mean of first sample.
        std1: Standard deviation of first sample.
        n1: Size of first sample.
        mean2: Mean of second sample.
        std2: Standard deviation of second sample.
        n2: Size of second sample.

    Returns:
        Two-tailed p-value.
    """
    t_stat, df = welch_t_test(mean1, std1, n1, mean2, std2, n2)

    if df == 0:
        return 1.0

    return t_distribution_cdf(t_stat, df)


def calculate_cohens_d(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> float:
    """Calculate Cohen's d effect size.

    Args:
        mean1: Mean of first sample.
        std1: Standard deviation of first sample.
        n1: Size of first sample.
        mean2: Mean of second sample.
        std2: Standard deviation of second sample.
        n2: Size of second sample.

    Returns:
        Cohen's d effect size.
    """
    if n1 < 2 or n2 < 2:
        return 0.0

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Human-readable interpretation.
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


class StatisticalAnalyzer:
    """Analyzer for evaluation results with statistical testing."""

    def __init__(self) -> None:
        """Initialize the statistical analyzer."""
        self._logger = logger.bind(component="statistical_analyzer")

    def calculate_run_statistics(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, dict[str, float]]:
        """Calculate statistics for an evaluation run.

        Args:
            results: List of evaluation results.

        Returns:
            Dictionary with mean, std, min, max for each dimension.
        """
        if not results:
            return {}

        # Collect scores by dimension
        scores_by_dimension: dict[str, list[float]] = {}

        for result in results:
            for dim_score in result.scores:
                dim_name = dim_score.dimension.value
                if dim_name not in scores_by_dimension:
                    scores_by_dimension[dim_name] = []
                scores_by_dimension[dim_name].append(dim_score.score)

        # Calculate statistics
        stats: dict[str, dict[str, float]] = {}

        for dim_name, scores in scores_by_dimension.items():
            mean = calculate_mean(scores)
            stats[dim_name] = {
                "mean": mean,
                "std": calculate_std(scores, mean),
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "count": len(scores),
            }

        # Add aggregate statistics
        aggregate_scores = [r.aggregate_score for r in results]
        if aggregate_scores:
            mean = calculate_mean(aggregate_scores)
            stats["aggregate"] = {
                "mean": mean,
                "std": calculate_std(aggregate_scores, mean),
                "min": min(aggregate_scores),
                "max": max(aggregate_scores),
                "count": len(aggregate_scores),
            }

        return stats

    def compare_runs(
        self,
        control: EvaluationRunSummary,
        treatment: EvaluationRunSummary,
        confidence_level: float = 0.95,
    ) -> dict[str, dict[str, Any]]:
        """Compare two evaluation runs statistically.

        Args:
            control: Control run summary.
            treatment: Treatment run summary.
            confidence_level: Required confidence level (default 0.95).

        Returns:
            Comparison results for each dimension.
        """
        alpha = 1 - confidence_level
        comparisons: dict[str, dict[str, Any]] = {}

        # Get all dimensions from both runs
        all_dimensions = set(control.mean_scores.keys()) | set(treatment.mean_scores.keys())

        for dim in all_dimensions:
            control_mean = control.mean_scores.get(dim, 0.0)
            control_std = control.std_scores.get(dim, 0.0)
            treatment_mean = treatment.mean_scores.get(dim, 0.0)
            treatment_std = treatment.std_scores.get(dim, 0.0)

            # Calculate p-value
            p_value = calculate_p_value(
                control_mean,
                control_std,
                control.total_cases,
                treatment_mean,
                treatment_std,
                treatment.total_cases,
            )

            # Calculate effect size
            effect_size = calculate_cohens_d(
                treatment_mean,  # Treatment first so positive = treatment better
                treatment_std,
                treatment.total_cases,
                control_mean,
                control_std,
                control.total_cases,
            )

            # Determine significance and winner
            is_significant = p_value < alpha
            diff = treatment_mean - control_mean

            winner = ("treatment" if diff > 0 else "control") if is_significant else None

            comparisons[dim] = {
                "control_mean": control_mean,
                "control_std": control_std,
                "treatment_mean": treatment_mean,
                "treatment_std": treatment_std,
                "difference": diff,
                "p_value": p_value,
                "effect_size": effect_size,
                "effect_interpretation": interpret_effect_size(effect_size),
                "is_significant": is_significant,
                "winner": winner,
                "confidence_level": confidence_level,
            }

        self._logger.info(
            "run_comparison_complete",
            control_run=control.run_id,
            treatment_run=treatment.run_id,
            dimensions_compared=len(comparisons),
        )

        return comparisons

    def analyze_ab_test(
        self,
        config: ABTestConfig,
        control_summary: EvaluationRunSummary,
        treatment_summary: EvaluationRunSummary,
    ) -> ABTestResult:
        """Perform full A/B test analysis.

        Args:
            config: A/B test configuration.
            control_summary: Control variant results.
            treatment_summary: Treatment variant results.

        Returns:
            Complete A/B test result with statistical analysis.
        """
        # Check minimum sample size
        min_samples = min(control_summary.total_cases, treatment_summary.total_cases)
        if min_samples < config.min_samples:
            self._logger.warning(
                "insufficient_samples_for_ab_test",
                required=config.min_samples,
                actual=min_samples,
            )

        # Compare runs
        comparisons = self.compare_runs(
            control_summary,
            treatment_summary,
            config.confidence_level,
        )

        # Determine overall winner based on aggregate score
        aggregate_comparison = comparisons.get("aggregate", {})
        overall_winner = aggregate_comparison.get("winner")
        overall_significant = aggregate_comparison.get("is_significant", False)
        overall_p_value = aggregate_comparison.get("p_value")
        overall_effect = aggregate_comparison.get("effect_size")

        # Build recommendation
        if not overall_significant:
            recommendation = (
                f"No statistically significant difference detected between variants "
                f"(p={overall_p_value:.4f}). Consider running more samples or "
                f"keeping the current implementation."
            )
        elif overall_winner == "treatment":
            effect_interp = interpret_effect_size(overall_effect or 0)
            recommendation = (
                f"Treatment variant shows {effect_interp} improvement over control "
                f"(p={overall_p_value:.4f}, d={overall_effect:.3f}). "
                f"Recommend adopting treatment variant."
            )
        else:
            effect_interp = interpret_effect_size(overall_effect or 0)
            recommendation = (
                f"Control variant outperforms treatment "
                f"(p={overall_p_value:.4f}, d={overall_effect:.3f}). "
                f"Recommend keeping current implementation."
            )

        result = ABTestResult(
            config=config,
            control_summary=control_summary,
            treatment_summary=treatment_summary,
            dimension_comparisons=comparisons,
            winner=overall_winner,
            is_significant=overall_significant,
            p_value=overall_p_value,
            effect_size=overall_effect,
            recommendation=recommendation,
        )

        self._logger.info(
            "ab_test_analysis_complete",
            test_name=config.name,
            winner=overall_winner,
            is_significant=overall_significant,
            p_value=overall_p_value,
        )

        return result

    def detect_regression(
        self,
        baseline: EvaluationRunSummary,
        current: EvaluationRunSummary,
        threshold: float = 0.05,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Detect quality regression between runs.

        Args:
            baseline: Previous/baseline run summary.
            current: Current run summary to compare.
            threshold: Minimum score drop to consider a regression.
            confidence_level: Required confidence level.

        Returns:
            Regression analysis results.
        """
        comparisons = self.compare_runs(baseline, current, confidence_level)

        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []

        for dim, comp in comparisons.items():
            diff = comp["difference"]
            is_sig = comp["is_significant"]

            if is_sig and diff < -threshold:
                regressions.append({
                    "dimension": dim,
                    "baseline_score": comp["control_mean"],
                    "current_score": comp["treatment_mean"],
                    "drop": abs(diff),
                    "p_value": comp["p_value"],
                    "effect_size": comp["effect_size"],
                })
            elif is_sig and diff > threshold:
                improvements.append({
                    "dimension": dim,
                    "baseline_score": comp["control_mean"],
                    "current_score": comp["treatment_mean"],
                    "gain": diff,
                    "p_value": comp["p_value"],
                    "effect_size": comp["effect_size"],
                })

        has_regression = len(regressions) > 0

        result = {
            "has_regression": has_regression,
            "regressions": regressions,
            "improvements": improvements,
            "comparisons": comparisons,
            "threshold": threshold,
            "confidence_level": confidence_level,
        }

        if has_regression:
            self._logger.warning(
                "regression_detected",
                baseline_run=baseline.run_id,
                current_run=current.run_id,
                regression_count=len(regressions),
                dimensions=[r["dimension"] for r in regressions],
            )

        return result
