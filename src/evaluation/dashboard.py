"""Evaluation dashboard for visualizing model quality metrics.

This module provides dashboard configurations for evaluation metrics,
integrating with the existing observability dashboard infrastructure.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from src.evaluation.models import (
    ABTestResult,
    EvaluationDimension,
)
from src.evaluation.storage import EvaluationStorage

logger = structlog.get_logger(__name__)


class EvaluationDashboard:
    """Dashboard for evaluation metrics visualization.

    Provides data aggregation and formatting for evaluation metrics,
    designed to integrate with the observability dashboard infrastructure.
    """

    def __init__(self, storage: EvaluationStorage | None = None) -> None:
        """Initialize the evaluation dashboard.

        Args:
            storage: Storage backend for evaluation data. Creates default if not provided.
        """
        self.storage = storage or EvaluationStorage()
        self._logger = logger.bind(component="evaluation_dashboard")

    async def get_score_trends(
        self,
        days: int = 30,
        agent_filter: str | None = None,
        dimension: EvaluationDimension | None = None,
    ) -> dict[str, Any]:
        """Get evaluation score trends over time.

        Args:
            days: Number of days to look back.
            agent_filter: Optional agent name filter.
            dimension: Optional dimension to filter by.

        Returns:
            Trend data for charting.
        """
        since = datetime.utcnow() - timedelta(days=days)
        summaries = await self.storage.list_summaries(
            since=since,
            agent_filter=agent_filter,
        )

        if not summaries:
            return {
                "labels": [],
                "datasets": [],
                "metadata": {"days": days, "count": 0},
            }

        # Group by date
        scores_by_date: dict[str, dict[str, list[float]]] = {}

        for summary in summaries:
            date_key = summary.started_at.strftime("%Y-%m-%d")
            if date_key not in scores_by_date:
                scores_by_date[date_key] = {}

            for dim_name, score in summary.mean_scores.items():
                if dimension and dim_name != dimension.value:
                    continue
                if dim_name not in scores_by_date[date_key]:
                    scores_by_date[date_key][dim_name] = []
                scores_by_date[date_key][dim_name].append(score)

        # Calculate daily averages
        labels = sorted(scores_by_date.keys())
        dimensions_seen: set[str] = set()
        for date_scores in scores_by_date.values():
            dimensions_seen.update(date_scores.keys())

        datasets = []
        for dim_name in sorted(dimensions_seen):
            values = []
            for date in labels:
                day_scores = scores_by_date.get(date, {}).get(dim_name, [])
                avg = sum(day_scores) / len(day_scores) if day_scores else None
                values.append(avg)

            datasets.append({
                "label": dim_name.replace("_", " ").title(),
                "data": values,
            })

        return {
            "labels": labels,
            "datasets": datasets,
            "metadata": {
                "days": days,
                "count": len(summaries),
                "agent_filter": agent_filter,
            },
        }

    async def get_score_distribution(
        self,
        dimension: EvaluationDimension,
        days: int = 30,
        bins: int = 10,
    ) -> dict[str, Any]:
        """Get score distribution for a dimension.

        Args:
            dimension: Dimension to analyze.
            days: Number of days to look back.
            bins: Number of histogram bins.

        Returns:
            Distribution data for charting.
        """
        since = datetime.utcnow() - timedelta(days=days)
        summaries = await self.storage.list_summaries(since=since)

        scores = []
        for summary in summaries:
            for result in summary.results:
                score = result.get_score(dimension)
                if score is not None:
                    scores.append(score)

        if not scores:
            return {
                "labels": [],
                "data": [],
                "metadata": {"dimension": dimension.value, "count": 0},
            }

        # Create histogram bins
        bin_width = 1.0 / bins
        bin_counts = [0] * bins

        for score in scores:
            bin_idx = min(int(score / bin_width), bins - 1)
            bin_counts[bin_idx] += 1

        labels = [f"{i * bin_width:.1f}-{(i + 1) * bin_width:.1f}" for i in range(bins)]

        return {
            "labels": labels,
            "data": bin_counts,
            "metadata": {
                "dimension": dimension.value,
                "count": len(scores),
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            },
        }

    async def get_agent_comparison(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Compare scores across different agents.

        Args:
            days: Number of days to look back.

        Returns:
            Comparison data for charting.
        """
        since = datetime.utcnow() - timedelta(days=days)
        summaries = await self.storage.list_summaries(since=since)

        # Group by agent (from config metadata)
        agent_scores: dict[str, dict[str, list[float]]] = {}

        for summary in summaries:
            agent_name = summary.config.metadata.get("agent_name", "unknown")
            if agent_name not in agent_scores:
                agent_scores[agent_name] = {}

            for dim_name, score in summary.mean_scores.items():
                if dim_name not in agent_scores[agent_name]:
                    agent_scores[agent_name][dim_name] = []
                agent_scores[agent_name][dim_name].append(score)

        # Calculate averages per agent
        agents = sorted(agent_scores.keys())
        dimensions_set: set[str] = set()
        for scores in agent_scores.values():
            dimensions_set.update(scores.keys())
        dimensions = sorted(dimensions_set)

        datasets = []
        for dim in dimensions:
            values = []
            for agent in agents:
                agent_dim_scores = agent_scores.get(agent, {}).get(dim, [])
                avg = sum(agent_dim_scores) / len(agent_dim_scores) if agent_dim_scores else 0
                values.append(avg)

            datasets.append({
                "label": dim.replace("_", " ").title(),
                "data": values,
            })

        return {
            "labels": agents,
            "datasets": datasets,
            "metadata": {"days": days, "agent_count": len(agents)},
        }

    async def get_cost_quality_tradeoff(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get cost vs quality tradeoff data.

        Args:
            days: Number of days to look back.

        Returns:
            Scatter plot data for cost vs quality.
        """
        since = datetime.utcnow() - timedelta(days=days)
        summaries = await self.storage.list_summaries(since=since)

        points = []
        for summary in summaries:
            aggregate_score = summary.mean_scores.get("aggregate", 0)
            cost = summary.total_cost_usd
            model = summary.config.model

            points.append({
                "x": cost,
                "y": aggregate_score,
                "label": summary.run_id[:8],
                "model": model,
            })

        # Group by model for different series
        models = list({p["model"] for p in points})
        datasets = []

        for model_name in models:
            model_points = [p for p in points if p["model"] == model_name]
            datasets.append({
                "label": model_name,
                "data": [{"x": p["x"], "y": p["y"]} for p in model_points],
            })

        return {
            "datasets": datasets,
            "metadata": {
                "days": days,
                "total_runs": len(points),
                "models": models,
            },
        }

    async def get_regression_alerts(
        self,
        threshold: float = 0.05,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get alerts for score regressions.

        Args:
            threshold: Minimum score drop to alert on.
            days: Days to look back.

        Returns:
            List of regression alerts.
        """
        since = datetime.utcnow() - timedelta(days=days)
        summaries = await self.storage.list_summaries(since=since)

        if len(summaries) < 2:
            return []

        # Sort by date
        sorted_summaries = sorted(summaries, key=lambda s: s.started_at)

        alerts = []
        for i in range(1, len(sorted_summaries)):
            prev = sorted_summaries[i - 1]
            curr = sorted_summaries[i]

            for dim in curr.mean_scores:
                prev_score = prev.mean_scores.get(dim, 0)
                curr_score = curr.mean_scores.get(dim, 0)
                drop = prev_score - curr_score

                if drop > threshold:
                    alerts.append({
                        "dimension": dim,
                        "previous_score": prev_score,
                        "current_score": curr_score,
                        "drop": drop,
                        "previous_run_id": prev.run_id,
                        "current_run_id": curr.run_id,
                        "timestamp": curr.started_at.isoformat(),
                        "severity": "high" if drop > 0.1 else "medium",
                    })

        # Sort by severity and drop amount
        def alert_sort_key(a: dict[str, Any]) -> tuple[int, float]:
            severity_order = -1 if a["severity"] == "high" else 0
            drop_value = float(a["drop"])
            return (severity_order, -drop_value)

        alerts.sort(key=alert_sort_key)

        return alerts

    async def get_ab_test_summary(
        self,
        test_id: str,
    ) -> dict[str, Any] | None:
        """Get summary data for an A/B test.

        Args:
            test_id: ID of the A/B test.

        Returns:
            A/B test summary data or None if not found.
        """
        result = await self.storage.get_ab_test(test_id)
        if not result:
            return None

        return self._format_ab_test_result(result)

    async def get_recent_ab_tests(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent A/B test results.

        Args:
            limit: Maximum number of tests to return.

        Returns:
            List of A/B test summaries.
        """
        tests = await self.storage.list_ab_tests(limit=limit)
        return [self._format_ab_test_result(t) for t in tests]

    def _format_ab_test_result(self, result: ABTestResult) -> dict[str, Any]:
        """Format A/B test result for dashboard display.

        Args:
            result: A/B test result.

        Returns:
            Formatted data for display.
        """
        # Calculate confidence intervals for each dimension
        dimension_data = []
        for dim, comparison in result.dimension_comparisons.items():
            control_mean = comparison.get("control_mean", 0)
            treatment_mean = comparison.get("treatment_mean", 0)
            control_std = comparison.get("control_std", 0)
            treatment_std = comparison.get("treatment_std", 0)

            dimension_data.append({
                "dimension": dim,
                "control": {
                    "mean": control_mean,
                    "ci_lower": max(0, control_mean - 1.96 * control_std),
                    "ci_upper": min(1, control_mean + 1.96 * control_std),
                },
                "treatment": {
                    "mean": treatment_mean,
                    "ci_lower": max(0, treatment_mean - 1.96 * treatment_std),
                    "ci_upper": min(1, treatment_mean + 1.96 * treatment_std),
                },
                "p_value": comparison.get("p_value"),
                "effect_size": comparison.get("effect_size"),
                "is_significant": comparison.get("is_significant", False),
                "winner": comparison.get("winner"),
            })

        return {
            "test_name": result.config.name,
            "description": result.config.description,
            "control_run_id": result.control_summary.run_id,
            "treatment_run_id": result.treatment_summary.run_id,
            "winner": result.winner,
            "is_significant": result.is_significant,
            "p_value": result.p_value,
            "effect_size": result.effect_size,
            "recommendation": result.recommendation,
            "dimensions": dimension_data,
            "control_summary": {
                "total_cases": result.control_summary.total_cases,
                "pass_rate": result.control_summary.pass_rate,
                "mean_latency_ms": result.control_summary.mean_latency_ms,
                "total_cost_usd": result.control_summary.total_cost_usd,
            },
            "treatment_summary": {
                "total_cases": result.treatment_summary.total_cases,
                "pass_rate": result.treatment_summary.pass_rate,
                "mean_latency_ms": result.treatment_summary.mean_latency_ms,
                "total_cost_usd": result.treatment_summary.total_cost_usd,
            },
        }

    async def get_dashboard_overview(
        self,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get overview metrics for dashboard header.

        Args:
            days: Number of days to look back.

        Returns:
            Overview metrics.
        """
        since = datetime.utcnow() - timedelta(days=days)
        summaries = await self.storage.list_summaries(since=since)

        if not summaries:
            return {
                "total_runs": 0,
                "total_cases": 0,
                "overall_pass_rate": 0,
                "average_score": 0,
                "total_cost_usd": 0,
                "alerts_count": 0,
            }

        total_cases = sum(s.total_cases for s in summaries)
        total_passed = sum(s.passed_cases for s in summaries)
        total_cost = sum(s.total_cost_usd for s in summaries)

        aggregate_scores = [
            s.mean_scores.get("aggregate", 0)
            for s in summaries
            if "aggregate" in s.mean_scores
        ]
        avg_score = sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0

        alerts = await self.get_regression_alerts(days=days)

        return {
            "total_runs": len(summaries),
            "total_cases": total_cases,
            "overall_pass_rate": total_passed / total_cases if total_cases > 0 else 0,
            "average_score": avg_score,
            "total_cost_usd": total_cost,
            "alerts_count": len(alerts),
            "period_days": days,
        }
