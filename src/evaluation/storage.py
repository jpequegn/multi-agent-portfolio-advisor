"""Storage backend for evaluation data.

This module provides persistence for evaluation results,
supporting historical analysis and regression detection.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

from src.evaluation.models import (
    ABTestConfig,
    ABTestResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
)

logger = structlog.get_logger(__name__)

DEFAULT_DB_PATH = Path("data/evaluations.db")


class EvaluationStorage:
    """SQLite-based storage for evaluation results.

    Provides persistence for evaluation run summaries, A/B test results,
    and historical data for trend analysis.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the evaluation storage.

        Args:
            db_path: Path to SQLite database. Uses default if not provided.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._logger = logger.bind(component="evaluation_storage")
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure database tables exist."""
        if self._initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            # Evaluation run summaries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_summaries (
                    run_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    total_cases INTEGER NOT NULL,
                    passed_cases INTEGER NOT NULL,
                    failed_cases INTEGER NOT NULL,
                    error_cases INTEGER NOT NULL,
                    pass_rate REAL NOT NULL,
                    mean_scores_json TEXT NOT NULL,
                    std_scores_json TEXT NOT NULL,
                    total_latency_ms INTEGER NOT NULL,
                    mean_latency_ms REAL NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    total_input_tokens INTEGER NOT NULL,
                    total_output_tokens INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_seconds REAL NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # A/B test results table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    control_summary_run_id TEXT NOT NULL,
                    treatment_summary_run_id TEXT NOT NULL,
                    dimension_comparisons_json TEXT NOT NULL,
                    winner TEXT,
                    is_significant INTEGER NOT NULL,
                    p_value REAL,
                    effect_size REAL,
                    recommendation TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (control_summary_run_id) REFERENCES evaluation_summaries(run_id),
                    FOREIGN KEY (treatment_summary_run_id) REFERENCES evaluation_summaries(run_id)
                )
            """)

            # Indexes for common queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_started_at
                ON evaluation_summaries(started_at)
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_ab_tests_created_at
                ON ab_test_results(created_at)
            """)

            await db.commit()

        self._initialized = True
        self._logger.info("storage_initialized", db_path=str(self.db_path))

    async def save_summary(self, summary: EvaluationRunSummary) -> None:
        """Save an evaluation run summary.

        Args:
            summary: Summary to save.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO evaluation_summaries (
                    run_id, config_json, total_cases, passed_cases, failed_cases,
                    error_cases, pass_rate, mean_scores_json, std_scores_json,
                    total_latency_ms, mean_latency_ms, total_cost_usd,
                    total_input_tokens, total_output_tokens, started_at,
                    completed_at, duration_seconds, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.run_id,
                    summary.config.model_dump_json(),
                    summary.total_cases,
                    summary.passed_cases,
                    summary.failed_cases,
                    summary.error_cases,
                    summary.pass_rate,
                    json.dumps(summary.mean_scores),
                    json.dumps(summary.std_scores),
                    summary.total_latency_ms,
                    summary.mean_latency_ms,
                    summary.total_cost_usd,
                    summary.total_input_tokens,
                    summary.total_output_tokens,
                    summary.started_at.isoformat(),
                    summary.completed_at.isoformat() if summary.completed_at else None,
                    summary.duration_seconds,
                    json.dumps([r.model_dump() for r in summary.results], default=str),
                ),
            )
            await db.commit()

        self._logger.debug("summary_saved", run_id=summary.run_id)

    async def get_summary(self, run_id: str) -> EvaluationRunSummary | None:
        """Get an evaluation summary by ID.

        Args:
            run_id: Run ID to look up.

        Returns:
            Summary or None if not found.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM evaluation_summaries WHERE run_id = ?",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_summary(dict(row))

    async def list_summaries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        agent_filter: str | None = None,
        limit: int = 100,
    ) -> list[EvaluationRunSummary]:
        """List evaluation summaries with optional filters.

        Args:
            since: Only include runs after this time.
            until: Only include runs before this time.
            agent_filter: Filter by agent name in config metadata.
            limit: Maximum results to return.

        Returns:
            List of matching summaries.
        """
        await self._ensure_initialized()

        conditions = []
        params: list[Any] = []

        if since:
            conditions.append("started_at >= ?")
            params.append(since.isoformat())

        if until:
            conditions.append("started_at <= ?")
            params.append(until.isoformat())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT * FROM evaluation_summaries
            {where_clause}
            ORDER BY started_at DESC
            LIMIT ?
        """
        params.append(limit)

        summaries = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    summary = self._row_to_summary(dict(row))

                    # Apply agent filter if specified
                    if agent_filter:
                        agent_name = summary.config.metadata.get("agent_name", "")
                        if agent_filter.lower() not in agent_name.lower():
                            continue

                    summaries.append(summary)

        return summaries

    async def save_ab_test(self, result: ABTestResult, test_id: str) -> None:
        """Save an A/B test result.

        Args:
            result: A/B test result to save.
            test_id: Unique ID for this test.
        """
        await self._ensure_initialized()

        # Save summaries first
        await self.save_summary(result.control_summary)
        await self.save_summary(result.treatment_summary)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO ab_test_results (
                    test_id, config_json, control_summary_run_id,
                    treatment_summary_run_id, dimension_comparisons_json,
                    winner, is_significant, p_value, effect_size, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    test_id,
                    result.config.model_dump_json(),
                    result.control_summary.run_id,
                    result.treatment_summary.run_id,
                    json.dumps(result.dimension_comparisons),
                    result.winner,
                    1 if result.is_significant else 0,
                    result.p_value,
                    result.effect_size,
                    result.recommendation,
                ),
            )
            await db.commit()

        self._logger.debug("ab_test_saved", test_id=test_id)

    async def get_ab_test(self, test_id: str) -> ABTestResult | None:
        """Get an A/B test result by ID.

        Args:
            test_id: Test ID to look up.

        Returns:
            A/B test result or None if not found.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ab_test_results WHERE test_id = ?",
                (test_id,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            return None

        row_dict = dict(row)

        # Get the related summaries
        control_summary = await self.get_summary(row_dict["control_summary_run_id"])
        treatment_summary = await self.get_summary(row_dict["treatment_summary_run_id"])

        if not control_summary or not treatment_summary:
            self._logger.error(
                "ab_test_missing_summaries",
                test_id=test_id,
                control_found=control_summary is not None,
                treatment_found=treatment_summary is not None,
            )
            return None

        return ABTestResult(
            config=ABTestConfig.model_validate_json(row_dict["config_json"]),
            control_summary=control_summary,
            treatment_summary=treatment_summary,
            dimension_comparisons=json.loads(row_dict["dimension_comparisons_json"]),
            winner=row_dict["winner"],
            is_significant=bool(row_dict["is_significant"]),
            p_value=row_dict["p_value"],
            effect_size=row_dict["effect_size"],
            recommendation=row_dict["recommendation"],
        )

    async def list_ab_tests(
        self,
        limit: int = 10,
    ) -> list[ABTestResult]:
        """List recent A/B test results.

        Args:
            limit: Maximum results to return.

        Returns:
            List of A/B test results.
        """
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT test_id FROM ab_test_results
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

        results = []
        for row in rows:
            result = await self.get_ab_test(row["test_id"])
            if result:
                results.append(result)

        return results

    async def get_baseline_summary(
        self,
        agent_name: str | None = None,
    ) -> EvaluationRunSummary | None:
        """Get the most recent passing summary as a baseline.

        Args:
            agent_name: Optional agent name filter.

        Returns:
            Most recent passing summary or None.
        """
        summaries = await self.list_summaries(
            agent_filter=agent_name,
            limit=10,
        )

        for summary in summaries:
            if summary.pass_rate >= 0.7:  # Consider passing if 70%+ passed
                return summary

        return summaries[0] if summaries else None

    async def delete_old_data(self, days: int = 90) -> int:
        """Delete evaluation data older than specified days.

        Args:
            days: Delete data older than this many days.

        Returns:
            Number of records deleted.
        """
        await self._ensure_initialized()

        cutoff = datetime.utcnow()
        cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta

        cutoff = cutoff - timedelta(days=days)

        async with aiosqlite.connect(self.db_path) as db:
            # Delete old summaries (cascade will handle related data)
            cursor = await db.execute(
                "DELETE FROM evaluation_summaries WHERE started_at < ?",
                (cutoff.isoformat(),),
            )
            deleted_summaries = cursor.rowcount

            # Delete old A/B tests
            cursor = await db.execute(
                "DELETE FROM ab_test_results WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            deleted_tests = cursor.rowcount

            await db.commit()

        total_deleted = deleted_summaries + deleted_tests
        self._logger.info(
            "old_data_deleted",
            days=days,
            summaries_deleted=deleted_summaries,
            tests_deleted=deleted_tests,
        )

        return total_deleted

    def _row_to_summary(self, row: dict[str, Any]) -> EvaluationRunSummary:
        """Convert a database row to EvaluationRunSummary.

        Args:
            row: Database row as dict.

        Returns:
            EvaluationRunSummary instance.
        """
        from src.evaluation.models import EvaluationResult

        config = EvaluationRunConfig.model_validate_json(row["config_json"])

        # Parse results JSON
        results_data = json.loads(row["results_json"])
        results = [EvaluationResult.model_validate(r) for r in results_data]

        completed_at = None
        if row["completed_at"]:
            completed_at = datetime.fromisoformat(row["completed_at"])

        return EvaluationRunSummary(
            run_id=row["run_id"],
            config=config,
            total_cases=row["total_cases"],
            passed_cases=row["passed_cases"],
            failed_cases=row["failed_cases"],
            error_cases=row["error_cases"],
            pass_rate=row["pass_rate"],
            mean_scores=json.loads(row["mean_scores_json"]),
            std_scores=json.loads(row["std_scores_json"]),
            total_latency_ms=row["total_latency_ms"],
            mean_latency_ms=row["mean_latency_ms"],
            total_cost_usd=row["total_cost_usd"],
            total_input_tokens=row["total_input_tokens"],
            total_output_tokens=row["total_output_tokens"],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=completed_at,
            duration_seconds=row["duration_seconds"],
            results=results,
        )
