"""API endpoints for evaluation dashboard.

This module provides REST endpoints for evaluation metrics,
dashboard data, and A/B test results.
"""

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query

from src.evaluation.dashboard import EvaluationDashboard
from src.evaluation.models import EvaluationDimension
from src.evaluation.storage import EvaluationStorage

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Shared instances
_storage: EvaluationStorage | None = None
_dashboard: EvaluationDashboard | None = None


def get_storage() -> EvaluationStorage:
    """Get or create storage instance."""
    global _storage
    if _storage is None:
        _storage = EvaluationStorage()
    return _storage


def get_dashboard() -> EvaluationDashboard:
    """Get or create dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = EvaluationDashboard(get_storage())
    return _dashboard


@router.get("/overview")
async def get_overview(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
) -> dict[str, Any]:
    """Get dashboard overview metrics.

    Returns summary statistics including total runs, pass rate,
    average scores, and cost for the specified period.
    """
    dashboard = get_dashboard()
    return await dashboard.get_dashboard_overview(days=days)


@router.get("/trends")
async def get_score_trends(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    agent: str | None = Query(default=None, description="Filter by agent name"),
    dimension: str | None = Query(default=None, description="Filter by dimension"),
) -> dict[str, Any]:
    """Get evaluation score trends over time.

    Returns time series data for charting score trends by dimension.
    """
    dashboard = get_dashboard()

    dim = None
    if dimension:
        try:
            dim = EvaluationDimension(dimension)
        except ValueError as e:
            valid = [d.value for d in EvaluationDimension]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimension '{dimension}'. Valid: {valid}",
            ) from e

    return await dashboard.get_score_trends(
        days=days,
        agent_filter=agent,
        dimension=dim,
    )


@router.get("/distribution/{dimension}")
async def get_score_distribution(
    dimension: str,
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    bins: int = Query(default=10, ge=5, le=20, description="Number of histogram bins"),
) -> dict[str, Any]:
    """Get score distribution for a dimension.

    Returns histogram data showing score frequency distribution.
    """
    try:
        dim = EvaluationDimension(dimension)
    except ValueError as e:
        valid = [d.value for d in EvaluationDimension]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dimension '{dimension}'. Valid: {valid}",
        ) from e

    dashboard = get_dashboard()
    return await dashboard.get_score_distribution(
        dimension=dim,
        days=days,
        bins=bins,
    )


@router.get("/agents")
async def get_agent_comparison(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
) -> dict[str, Any]:
    """Compare scores across different agents.

    Returns comparative data for all agents evaluated in the period.
    """
    dashboard = get_dashboard()
    return await dashboard.get_agent_comparison(days=days)


@router.get("/cost-quality")
async def get_cost_quality_tradeoff(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
) -> dict[str, Any]:
    """Get cost vs quality tradeoff data.

    Returns scatter plot data comparing cost and aggregate scores.
    """
    dashboard = get_dashboard()
    return await dashboard.get_cost_quality_tradeoff(days=days)


@router.get("/alerts")
async def get_regression_alerts(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to look back"),
    threshold: float = Query(
        default=0.05, ge=0.01, le=0.5, description="Minimum score drop to alert on"
    ),
) -> list[dict[str, Any]]:
    """Get regression alerts for quality drops.

    Returns a list of detected score regressions sorted by severity.
    """
    dashboard = get_dashboard()
    return await dashboard.get_regression_alerts(
        threshold=threshold,
        days=days,
    )


@router.get("/runs")
async def list_evaluation_runs(
    days: int = Query(default=30, ge=1, le=365, description="Number of days to look back"),
    agent: str | None = Query(default=None, description="Filter by agent name"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum results"),
) -> list[dict[str, Any]]:
    """List recent evaluation runs.

    Returns a summary of each evaluation run in the period.
    """
    from datetime import timedelta

    storage = get_storage()
    since = datetime.utcnow() - timedelta(days=days)

    summaries = await storage.list_summaries(
        since=since,
        agent_filter=agent,
        limit=limit,
    )

    return [
        {
            "run_id": s.run_id,
            "name": s.config.name,
            "total_cases": s.total_cases,
            "passed_cases": s.passed_cases,
            "failed_cases": s.failed_cases,
            "pass_rate": s.pass_rate,
            "mean_aggregate_score": s.mean_scores.get("aggregate", 0),
            "total_cost_usd": s.total_cost_usd,
            "mean_latency_ms": s.mean_latency_ms,
            "duration_seconds": s.duration_seconds,
            "started_at": s.started_at.isoformat(),
            "model": s.config.model,
        }
        for s in summaries
    ]


@router.get("/runs/{run_id}")
async def get_evaluation_run(run_id: str) -> dict[str, Any]:
    """Get details for a specific evaluation run.

    Returns full details including per-case results.
    """
    storage = get_storage()
    summary = await storage.get_summary(run_id)

    if not summary:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return {
        "run_id": summary.run_id,
        "config": summary.config.model_dump(),
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "failed_cases": summary.failed_cases,
        "error_cases": summary.error_cases,
        "pass_rate": summary.pass_rate,
        "mean_scores": summary.mean_scores,
        "std_scores": summary.std_scores,
        "min_scores": summary.min_scores,
        "max_scores": summary.max_scores,
        "total_latency_ms": summary.total_latency_ms,
        "mean_latency_ms": summary.mean_latency_ms,
        "total_cost_usd": summary.total_cost_usd,
        "total_input_tokens": summary.total_input_tokens,
        "total_output_tokens": summary.total_output_tokens,
        "started_at": summary.started_at.isoformat(),
        "completed_at": summary.completed_at.isoformat() if summary.completed_at else None,
        "duration_seconds": summary.duration_seconds,
        "results": [r.model_dump() for r in summary.results],
    }


@router.get("/ab-tests")
async def list_ab_tests(
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
) -> list[dict[str, Any]]:
    """List recent A/B test results.

    Returns summary of each A/B test.
    """
    dashboard = get_dashboard()
    return await dashboard.get_recent_ab_tests(limit=limit)


@router.get("/ab-tests/{test_id}")
async def get_ab_test(test_id: str) -> dict[str, Any]:
    """Get details for a specific A/B test.

    Returns full results including dimension comparisons and confidence intervals.
    """
    dashboard = get_dashboard()
    result = await dashboard.get_ab_test_summary(test_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"A/B test '{test_id}' not found")

    return result


@router.delete("/data")
async def cleanup_old_data(
    days: int = Query(default=90, ge=30, le=365, description="Delete data older than this"),
) -> dict[str, Any]:
    """Delete evaluation data older than specified days.

    Used for storage management and GDPR compliance.
    """
    storage = get_storage()
    deleted = await storage.delete_old_data(days=days)

    logger.info("evaluation_data_cleaned", days=days, deleted_count=deleted)

    return {
        "deleted_records": deleted,
        "cutoff_days": days,
    }
