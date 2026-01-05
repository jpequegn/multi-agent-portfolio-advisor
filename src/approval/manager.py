"""Approval Manager for orchestrating human approval workflows.

This module provides the core logic for evaluating recommendations,
creating approval requests, and managing the approval lifecycle.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from pydantic import BaseModel

from src.approval.models import (
    ApprovalConfig,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalTrigger,
    RiskLevel,
)

logger = structlog.get_logger(__name__)


# Global singleton for the approval manager
_approval_manager: "ApprovalManager | None" = None


def get_approval_manager() -> "ApprovalManager":
    """Get the global approval manager instance.

    Returns:
        The singleton ApprovalManager instance.
    """
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalManager()
    return _approval_manager


def set_approval_manager(manager: "ApprovalManager") -> None:
    """Set the global approval manager instance.

    Args:
        manager: The ApprovalManager instance to use.
    """
    global _approval_manager
    _approval_manager = manager


class ApprovalEvaluation(BaseModel):
    """Result of evaluating whether approval is required."""

    requires_approval: bool
    risk_level: RiskLevel
    triggers: list[ApprovalTrigger]
    details: dict[str, Any]


class ApprovalManager:
    """Manages human approval workflows for high-risk recommendations.

    The ApprovalManager is responsible for:
    - Evaluating whether recommendations require approval
    - Creating and storing approval requests
    - Processing approval decisions
    - Handling timeouts for stale requests
    """

    def __init__(
        self,
        config: ApprovalConfig | None = None,
        persistence: "ApprovalPersistence | None" = None,
    ) -> None:
        """Initialize the ApprovalManager.

        Args:
            config: Approval configuration (uses defaults if not provided).
            persistence: Persistence handler for storing approvals.
        """
        self.config = config or ApprovalConfig()
        self.persistence = persistence or InMemoryApprovalPersistence()
        logger.info(
            "approval_manager_initialized",
            config={
                "high_value_threshold": self.config.high_value_threshold,
                "concentration_threshold": self.config.concentration_threshold,
                "auto_approve_low_risk": self.config.auto_approve_low_risk,
            },
        )

    def evaluate_recommendation(
        self,
        trades: list[dict[str, Any]],
        portfolio_value: float,
        compliance: dict[str, Any],
        analysis_context: dict[str, Any] | None = None,
        known_symbols: set[str] | None = None,
        confidence_score: float | None = None,
    ) -> ApprovalEvaluation:
        """Evaluate whether a recommendation requires human approval.

        Args:
            trades: List of proposed trades.
            portfolio_value: Total portfolio value.
            compliance: Compliance check results.
            analysis_context: Additional context from analysis.
            known_symbols: Set of previously traded symbols.
            confidence_score: Model confidence score (0-1).

        Returns:
            ApprovalEvaluation with the determination.
        """
        triggers: list[ApprovalTrigger] = []
        details: dict[str, Any] = {}
        analysis_context = analysis_context or {}
        known_symbols = known_symbols or set()

        # Calculate total trade value
        total_value = sum(
            abs(t.get("estimated_value", 0)) for t in trades
        )
        details["total_trade_value"] = total_value

        # Check high value threshold
        if total_value > self.config.high_value_threshold:
            triggers.append(ApprovalTrigger.HIGH_VALUE_TRADE)
            details["value_threshold_exceeded"] = {
                "threshold": self.config.high_value_threshold,
                "actual": total_value,
            }

        # Check portfolio concentration
        max_concentration = self._calculate_max_concentration(trades, portfolio_value)
        details["max_concentration"] = max_concentration
        if max_concentration > self.config.concentration_threshold:
            triggers.append(ApprovalTrigger.HIGH_CONCENTRATION)
            details["concentration_threshold_exceeded"] = {
                "threshold": self.config.concentration_threshold,
                "actual": max_concentration,
            }

        # Check for high-risk asset types
        high_risk_symbols = self._find_high_risk_assets(trades)
        if high_risk_symbols:
            triggers.append(ApprovalTrigger.HIGH_RISK_ASSET)
            details["high_risk_assets"] = high_risk_symbols

        # Check for first-time symbols
        new_symbols = self._find_new_symbols(trades, known_symbols)
        if new_symbols:
            triggers.append(ApprovalTrigger.FIRST_TIME_SYMBOL)
            details["new_symbols"] = new_symbols

        # Check confidence score
        if confidence_score is not None and confidence_score < self.config.low_confidence_threshold:
            triggers.append(ApprovalTrigger.LOW_CONFIDENCE)
            details["confidence"] = {
                "threshold": self.config.low_confidence_threshold,
                "actual": confidence_score,
            }

        # Check compliance warnings
        compliance_warnings = compliance.get("warnings", [])
        compliance_violations = compliance.get("violations", [])
        if compliance_warnings or compliance_violations:
            triggers.append(ApprovalTrigger.COMPLIANCE_WARNING)
            details["compliance_issues"] = {
                "warnings": compliance_warnings,
                "violations": compliance_violations,
            }

        # Check if compliance explicitly requires approval
        if (
            compliance.get("requires_approval", False)
            and ApprovalTrigger.COMPLIANCE_WARNING not in triggers
        ):
            triggers.append(ApprovalTrigger.COMPLIANCE_WARNING)

        # Determine risk level based on triggers and values
        risk_level = self._determine_risk_level(triggers, total_value, max_concentration)

        # Determine if approval is required
        requires_approval = len(triggers) > 0
        if self.config.auto_approve_low_risk and risk_level == RiskLevel.LOW:
            requires_approval = False

        logger.debug(
            "recommendation_evaluated",
            requires_approval=requires_approval,
            risk_level=risk_level.value,
            trigger_count=len(triggers),
            triggers=[t.value for t in triggers],
        )

        return ApprovalEvaluation(
            requires_approval=requires_approval,
            risk_level=risk_level,
            triggers=triggers,
            details=details,
        )

    def _calculate_max_concentration(
        self, trades: list[dict[str, Any]], portfolio_value: float
    ) -> float:
        """Calculate maximum position concentration after proposed trades."""
        if portfolio_value <= 0:
            return 0.0

        # Group trades by symbol
        position_values: dict[str, float] = {}
        for trade in trades:
            symbol = trade.get("symbol", "")
            value = abs(trade.get("estimated_value", 0))
            current_weight = trade.get("current_weight", 0) or 0
            target_weight = trade.get("target_weight", 0) or 0

            # Use target weight if available, otherwise estimate from value
            if target_weight > 0:
                position_values[symbol] = target_weight
            else:
                position_values[symbol] = max(
                    current_weight,
                    value / portfolio_value if portfolio_value > 0 else 0,
                )

        return max(position_values.values()) if position_values else 0.0

    def _find_high_risk_assets(self, trades: list[dict[str, Any]]) -> list[str]:
        """Find trades involving high-risk asset types."""
        high_risk_symbols: list[str] = []

        for trade in trades:
            symbol = trade.get("symbol", "").upper()
            # Check for common high-risk indicators in symbol names
            # (In production, this would use asset classification data)
            if any(
                indicator in symbol
                for indicator in ["X3", "X2", "3X", "2X", "TQQQ", "SQQQ", "SPXS", "SPXL"]
            ):
                high_risk_symbols.append(symbol)

        return high_risk_symbols

    def _find_new_symbols(
        self, trades: list[dict[str, Any]], known_symbols: set[str]
    ) -> list[str]:
        """Find symbols that haven't been traded before."""
        new_symbols: list[str] = []

        for trade in trades:
            symbol = trade.get("symbol", "").upper()
            action = trade.get("action", "").lower()
            # Only flag buy actions for new symbols
            if action == "buy" and symbol not in known_symbols:
                new_symbols.append(symbol)

        return new_symbols

    def _determine_risk_level(
        self,
        triggers: list[ApprovalTrigger],
        total_value: float,
        max_concentration: float,
    ) -> RiskLevel:
        """Determine overall risk level based on triggers and values."""
        if not triggers:
            return RiskLevel.LOW

        # Critical if exceeds critical thresholds
        if total_value > self.config.critical_value_threshold:
            return RiskLevel.CRITICAL
        if max_concentration > self.config.critical_concentration_threshold:
            return RiskLevel.CRITICAL

        # High if multiple triggers or high-risk assets
        if len(triggers) >= 3:
            return RiskLevel.HIGH
        if ApprovalTrigger.HIGH_RISK_ASSET in triggers:
            return RiskLevel.HIGH

        # Medium for standard approval cases
        if triggers:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    async def create_approval_request(
        self,
        workflow_id: str,
        trace_id: str,
        trades: list[dict[str, Any]],
        portfolio_value: float,
        compliance: dict[str, Any],
        summary: str = "",
        analysis_context: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> ApprovalRequest:
        """Create an approval request for a recommendation.

        Args:
            workflow_id: Associated workflow ID.
            trace_id: Trace ID for observability.
            trades: Proposed trades.
            portfolio_value: Total portfolio value.
            compliance: Compliance check results.
            summary: Summary of the recommendation.
            analysis_context: Additional context from analysis.
            user_id: User who initiated the analysis.

        Returns:
            The created ApprovalRequest.
        """
        analysis_context = analysis_context or {}

        # Evaluate the recommendation
        evaluation = self.evaluate_recommendation(
            trades=trades,
            portfolio_value=portfolio_value,
            compliance=compliance,
            analysis_context=analysis_context,
        )

        # Calculate total trade value
        total_value = sum(abs(t.get("estimated_value", 0)) for t in trades)

        # Calculate expiration based on risk level
        timeout_hours = (
            self.config.critical_timeout_hours
            if evaluation.risk_level == RiskLevel.CRITICAL
            else self.config.default_timeout_hours
        )
        expires_at = datetime.now(UTC) + timedelta(hours=timeout_hours)

        # Create the approval request
        approval_request = ApprovalRequest(
            workflow_id=workflow_id,
            trace_id=trace_id,
            user_id=user_id,
            status=ApprovalStatus.PENDING,
            risk_level=evaluation.risk_level,
            triggers=evaluation.triggers,
            trades=trades,
            total_value=total_value,
            portfolio_value=portfolio_value,
            compliance=compliance,
            summary=summary,
            analysis_context={
                **analysis_context,
                "evaluation_details": evaluation.details,
            },
            expires_at=expires_at,
        )

        # Store the approval request
        await self.persistence.save(approval_request)

        logger.info(
            "approval_request_created",
            approval_id=approval_request.approval_id,
            workflow_id=workflow_id,
            risk_level=evaluation.risk_level.value,
            trigger_count=len(evaluation.triggers),
            expires_at=expires_at.isoformat(),
        )

        return approval_request

    async def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID.

        Args:
            approval_id: The approval request ID.

        Returns:
            The ApprovalRequest if found, None otherwise.
        """
        return await self.persistence.get(approval_id)

    async def get_approval_by_workflow(
        self, workflow_id: str
    ) -> ApprovalRequest | None:
        """Get an approval request by workflow ID.

        Args:
            workflow_id: The workflow ID.

        Returns:
            The ApprovalRequest if found, None otherwise.
        """
        return await self.persistence.get_by_workflow(workflow_id)

    async def list_pending(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalRequest]:
        """List pending approval requests.

        Args:
            user_id: Filter by user ID.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of pending ApprovalRequests.
        """
        # First, expire any stale requests
        await self._expire_stale_requests()

        return await self.persistence.list_pending(
            user_id=user_id, limit=limit, offset=offset
        )

    async def approve(
        self,
        approval_id: str,
        reviewer_id: str,
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        """Approve a pending request.

        Args:
            approval_id: The approval request ID.
            reviewer_id: ID of the reviewer.
            reason: Optional reason for approval.

        Returns:
            The updated ApprovalRequest, or None if not found.
        """
        approval = await self.persistence.get(approval_id)
        if approval is None:
            logger.warning("approval_not_found", approval_id=approval_id)
            return None

        if not approval.is_pending():
            logger.warning(
                "approval_not_pending",
                approval_id=approval_id,
                status=approval.status.value,
            )
            return approval

        if approval.is_expired():
            approval.mark_expired()
            await self.persistence.save(approval)
            logger.warning("approval_expired", approval_id=approval_id)
            return approval

        approval.approve(reviewer_id=reviewer_id, reason=reason)
        await self.persistence.save(approval)

        logger.info(
            "approval_approved",
            approval_id=approval_id,
            reviewer_id=reviewer_id,
            workflow_id=approval.workflow_id,
        )

        return approval

    async def reject(
        self,
        approval_id: str,
        reviewer_id: str,
        reason: str,
    ) -> ApprovalRequest | None:
        """Reject a pending request.

        Args:
            approval_id: The approval request ID.
            reviewer_id: ID of the reviewer.
            reason: Reason for rejection.

        Returns:
            The updated ApprovalRequest, or None if not found.
        """
        approval = await self.persistence.get(approval_id)
        if approval is None:
            logger.warning("approval_not_found", approval_id=approval_id)
            return None

        if not approval.is_pending():
            logger.warning(
                "approval_not_pending",
                approval_id=approval_id,
                status=approval.status.value,
            )
            return approval

        if approval.is_expired():
            approval.mark_expired()
            await self.persistence.save(approval)
            logger.warning("approval_expired", approval_id=approval_id)
            return approval

        approval.reject(reviewer_id=reviewer_id, reason=reason)
        await self.persistence.save(approval)

        logger.info(
            "approval_rejected",
            approval_id=approval_id,
            reviewer_id=reviewer_id,
            workflow_id=approval.workflow_id,
            reason=reason,
        )

        return approval

    async def modify(
        self,
        approval_id: str,
        reviewer_id: str,
        modified_trades: list[dict[str, Any]],
        reason: str | None = None,
    ) -> ApprovalRequest | None:
        """Approve with modifications.

        Args:
            approval_id: The approval request ID.
            reviewer_id: ID of the reviewer.
            modified_trades: The modified trade list.
            reason: Optional reason for modification.

        Returns:
            The updated ApprovalRequest, or None if not found.
        """
        approval = await self.persistence.get(approval_id)
        if approval is None:
            logger.warning("approval_not_found", approval_id=approval_id)
            return None

        if not approval.is_pending():
            logger.warning(
                "approval_not_pending",
                approval_id=approval_id,
                status=approval.status.value,
            )
            return approval

        if approval.is_expired():
            approval.mark_expired()
            await self.persistence.save(approval)
            logger.warning("approval_expired", approval_id=approval_id)
            return approval

        approval.modify(
            reviewer_id=reviewer_id,
            modified_trades=modified_trades,
            reason=reason,
        )
        await self.persistence.save(approval)

        logger.info(
            "approval_modified",
            approval_id=approval_id,
            reviewer_id=reviewer_id,
            workflow_id=approval.workflow_id,
            modified_trade_count=len(modified_trades),
        )

        return approval

    async def _expire_stale_requests(self) -> int:
        """Expire stale pending requests.

        Returns:
            Number of requests expired.
        """
        expired_count = await self.persistence.expire_stale()
        if expired_count > 0:
            logger.info("stale_approvals_expired", count=expired_count)
        return expired_count


class ApprovalPersistence:
    """Abstract base class for approval persistence."""

    async def save(self, approval: ApprovalRequest) -> None:
        """Save an approval request."""
        raise NotImplementedError

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID."""
        raise NotImplementedError

    async def get_by_workflow(self, workflow_id: str) -> ApprovalRequest | None:
        """Get an approval request by workflow ID."""
        raise NotImplementedError

    async def list_pending(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalRequest]:
        """List pending approval requests."""
        raise NotImplementedError

    async def expire_stale(self) -> int:
        """Expire stale pending requests."""
        raise NotImplementedError


class InMemoryApprovalPersistence(ApprovalPersistence):
    """In-memory persistence for development and testing."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._approvals: dict[str, ApprovalRequest] = {}
        self._by_workflow: dict[str, str] = {}

    async def save(self, approval: ApprovalRequest) -> None:
        """Save an approval request."""
        self._approvals[approval.approval_id] = approval
        self._by_workflow[approval.workflow_id] = approval.approval_id

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID."""
        return self._approvals.get(approval_id)

    async def get_by_workflow(self, workflow_id: str) -> ApprovalRequest | None:
        """Get an approval request by workflow ID."""
        approval_id = self._by_workflow.get(workflow_id)
        if approval_id:
            return self._approvals.get(approval_id)
        return None

    async def list_pending(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalRequest]:
        """List pending approval requests."""
        pending = [
            a
            for a in self._approvals.values()
            if a.status == ApprovalStatus.PENDING
            and (user_id is None or a.user_id == user_id)
        ]
        # Sort by created_at descending
        pending.sort(key=lambda a: a.created_at, reverse=True)
        return pending[offset : offset + limit]

    async def expire_stale(self) -> int:
        """Expire stale pending requests."""
        expired_count = 0
        now = datetime.now(UTC)

        for approval in self._approvals.values():
            if approval.is_pending() and approval.expires_at and now > approval.expires_at:
                approval.mark_expired()
                expired_count += 1

        return expired_count


class PostgresApprovalPersistence(ApprovalPersistence):
    """PostgreSQL persistence for production use."""

    def __init__(self, connection_string: str | None = None) -> None:
        """Initialize PostgreSQL persistence.

        Args:
            connection_string: PostgreSQL connection string.
                If not provided, uses DATABASE_URL environment variable.
        """
        import os

        self.connection_string = connection_string or os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/portfolio_advisor"
        )
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=2,
                    max_size=10,
                )
                logger.info("approval_database_pool_created")
            except ImportError:
                logger.warning(
                    "asyncpg_not_installed",
                    message="PostgreSQL approval persistence disabled",
                )
                raise
            except Exception as e:
                logger.error("approval_database_pool_failed", error=str(e))
                raise

        return self._pool

    async def initialize_schema(self) -> None:
        """Create the approval_requests table if it doesn't exist."""
        pool = await self._get_pool()

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS approval_requests (
            approval_id UUID PRIMARY KEY,
            workflow_id UUID NOT NULL,
            trace_id UUID NOT NULL,
            user_id TEXT,
            status TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            triggers TEXT[] NOT NULL DEFAULT '{}',
            trades JSONB NOT NULL DEFAULT '[]'::jsonb,
            total_value NUMERIC NOT NULL DEFAULT 0,
            portfolio_value NUMERIC NOT NULL DEFAULT 0,
            compliance JSONB NOT NULL DEFAULT '{}'::jsonb,
            summary TEXT,
            analysis_context JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ,
            decision JSONB,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_approval_requests_workflow_id
            ON approval_requests(workflow_id);
        CREATE INDEX IF NOT EXISTS idx_approval_requests_status
            ON approval_requests(status);
        CREATE INDEX IF NOT EXISTS idx_approval_requests_user_id
            ON approval_requests(user_id);
        CREATE INDEX IF NOT EXISTS idx_approval_requests_created_at
            ON approval_requests(created_at);
        CREATE INDEX IF NOT EXISTS idx_approval_requests_expires_at
            ON approval_requests(expires_at);
        """

        async with pool.acquire() as conn:
            await conn.execute(create_table_sql)
            logger.info("approval_schema_initialized")

    async def save(self, approval: ApprovalRequest) -> None:
        """Save an approval request."""
        pool = await self._get_pool()

        upsert_sql = """
        INSERT INTO approval_requests (
            approval_id, workflow_id, trace_id, user_id, status, risk_level,
            triggers, trades, total_value, portfolio_value, compliance,
            summary, analysis_context, created_at, expires_at, decision, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, NOW()
        )
        ON CONFLICT (approval_id) DO UPDATE SET
            status = EXCLUDED.status,
            decision = EXCLUDED.decision,
            updated_at = NOW()
        """

        import uuid as uuid_lib

        async with pool.acquire() as conn:
            await conn.execute(
                upsert_sql,
                uuid_lib.UUID(approval.approval_id),
                uuid_lib.UUID(approval.workflow_id),
                uuid_lib.UUID(approval.trace_id),
                approval.user_id,
                approval.status.value,
                approval.risk_level.value,
                [t.value for t in approval.triggers],
                json.dumps([dict(t) for t in approval.trades]),
                approval.total_value,
                approval.portfolio_value,
                json.dumps(approval.compliance),
                approval.summary,
                json.dumps(approval.analysis_context),
                approval.created_at,
                approval.expires_at,
                json.dumps(approval.decision.model_dump(mode="json"))
                if approval.decision
                else None,
            )

        logger.debug("approval_saved", approval_id=approval.approval_id)

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        """Get an approval request by ID."""
        pool = await self._get_pool()

        import uuid as uuid_lib

        select_sql = """
        SELECT approval_id, workflow_id, trace_id, user_id, status, risk_level,
               triggers, trades, total_value, portfolio_value, compliance,
               summary, analysis_context, created_at, expires_at, decision
        FROM approval_requests
        WHERE approval_id = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(select_sql, uuid_lib.UUID(approval_id))

        if not row:
            return None

        return self._row_to_approval(row)

    async def get_by_workflow(self, workflow_id: str) -> ApprovalRequest | None:
        """Get an approval request by workflow ID."""
        pool = await self._get_pool()

        import uuid as uuid_lib

        select_sql = """
        SELECT approval_id, workflow_id, trace_id, user_id, status, risk_level,
               triggers, trades, total_value, portfolio_value, compliance,
               summary, analysis_context, created_at, expires_at, decision
        FROM approval_requests
        WHERE workflow_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(select_sql, uuid_lib.UUID(workflow_id))

        if not row:
            return None

        return self._row_to_approval(row)

    async def list_pending(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ApprovalRequest]:
        """List pending approval requests."""
        pool = await self._get_pool()

        conditions = ["status = 'pending'"]
        params: list[Any] = []
        param_idx = 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        select_sql = f"""
        SELECT approval_id, workflow_id, trace_id, user_id, status, risk_level,
               triggers, trades, total_value, portfolio_value, compliance,
               summary, analysis_context, created_at, expires_at, decision
        FROM approval_requests
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with pool.acquire() as conn:
            rows = await conn.fetch(select_sql, *params)

        return [self._row_to_approval(row) for row in rows]

    async def expire_stale(self) -> int:
        """Expire stale pending requests."""
        pool = await self._get_pool()

        update_sql = """
        UPDATE approval_requests
        SET status = 'expired', updated_at = NOW()
        WHERE status = 'pending'
          AND expires_at IS NOT NULL
          AND expires_at < NOW()
        """

        async with pool.acquire() as conn:
            result = await conn.execute(update_sql)

        # Parse the result to get affected rows
        if result.startswith("UPDATE "):
            return int(result.split()[1])
        return 0

    def _row_to_approval(self, row: Any) -> ApprovalRequest:
        """Convert a database row to an ApprovalRequest."""
        from src.approval.models import ApprovalDecision

        decision = None
        if row["decision"]:
            decision_data = json.loads(row["decision"])
            decision = ApprovalDecision(**decision_data)

        return ApprovalRequest(
            approval_id=str(row["approval_id"]),
            workflow_id=str(row["workflow_id"]),
            trace_id=str(row["trace_id"]),
            user_id=row["user_id"],
            status=ApprovalStatus(row["status"]),
            risk_level=RiskLevel(row["risk_level"]),
            triggers=[ApprovalTrigger(t) for t in row["triggers"]],
            trades=json.loads(row["trades"]),
            total_value=float(row["total_value"]),
            portfolio_value=float(row["portfolio_value"]),
            compliance=json.loads(row["compliance"]),
            summary=row["summary"] or "",
            analysis_context=json.loads(row["analysis_context"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            decision=decision,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("approval_database_pool_closed")
