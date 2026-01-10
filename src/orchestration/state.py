"""State schema and management for multi-agent workflow.

This module defines the state that flows through the portfolio advisor workflow,
including input/output contracts for each agent and state persistence.
"""

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypedDict

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


# ============================================================================
# Enums
# ============================================================================


class WorkflowStatus(str, Enum):
    """Status of the workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentName(str, Enum):
    """Names of agents in the workflow."""

    RESEARCH = "research"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"


# ============================================================================
# Portfolio Input Schema
# ============================================================================


class Position(BaseModel):
    """A single portfolio position."""

    symbol: str = Field(..., description="Stock ticker symbol")
    quantity: float = Field(..., description="Number of shares held")
    cost_basis: float | None = Field(default=None, description="Average cost per share")
    current_price: float | None = Field(default=None, description="Current market price")
    market_value: float | None = Field(default=None, description="Current market value")
    weight: float | None = Field(default=None, description="Portfolio weight (0-1)")
    sector: str | None = Field(default=None, description="Sector classification")


class Portfolio(BaseModel):
    """Portfolio representation for workflow input."""

    positions: list[Position] = Field(default_factory=list)
    total_value: float = Field(default=0.0, description="Total portfolio value")
    cash: float = Field(default=0.0, description="Cash balance")
    account_type: str = Field(default="taxable", description="Account type")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("positions", mode="before")
    @classmethod
    def parse_positions(cls, v: Any) -> list[Position]:
        """Parse positions from various formats."""
        if not v:
            return []
        if isinstance(v, list):
            result: list[Position] = [
                Position(**p) if isinstance(p, dict) else p for p in v
            ]
            return result
        # If not a list, return empty (Pydantic will validate further)
        return []

    @property
    def symbols(self) -> list[str]:
        """Get list of symbols in portfolio."""
        return [p.symbol for p in self.positions]

    def get_weights(self) -> dict[str, float]:
        """Calculate portfolio weights."""
        if self.total_value <= 0:
            return {}
        return {
            p.symbol: (p.market_value or 0) / self.total_value
            for p in self.positions
            if p.market_value
        }


# ============================================================================
# Agent Output TypedDicts (for type hints in state)
# ============================================================================


class ResearchOutputDict(TypedDict, total=False):
    """TypedDict representation of ResearchAgent output."""

    market_data: dict[str, Any]
    news: list[dict[str, Any]]
    summary: str
    symbols_researched: list[str]
    data_freshness: str
    errors: list[str]


class AnalysisOutputDict(TypedDict, total=False):
    """TypedDict representation of AnalysisAgent output."""

    risk_metrics: dict[str, Any]
    correlations: dict[str, Any]
    benchmark_comparison: dict[str, Any]
    attribution: dict[str, Any]
    recommendations: list[str]
    summary: str
    errors: list[str]


class RecommendationOutputDict(TypedDict, total=False):
    """TypedDict representation of RecommendationAgent output."""

    trades: list[dict[str, Any]]
    tax_impact: dict[str, Any]
    execution_costs: dict[str, Any]
    compliance: dict[str, Any]
    summary: str
    total_trades: int
    buy_count: int
    sell_count: int
    hold_count: int
    errors: list[str]


# ============================================================================
# Workflow State Schema
# ============================================================================


class PortfolioState(TypedDict, total=False):
    """State that flows through the multi-agent workflow.

    This state is passed between agents and accumulates results from each step.
    Uses TypedDict for LangGraph compatibility.

    Flow:
        Request → State.portfolio, State.user_request
                ↓
        Research Agent → State.research (populated)
                ↓
        Analysis Agent ← reads State.research
                      → State.analysis (populated)
                ↓
        Recommendation Agent ← reads State.research, State.analysis
                            → State.recommendation (populated)
                ↓
        Response ← State.recommendation
    """

    # Identifiers
    workflow_id: str
    trace_id: str
    user_id: str | None

    # Input
    portfolio: dict[str, Any]  # Serialized Portfolio
    user_request: str
    symbols: list[str]

    # Agent outputs (populated as workflow progresses)
    research: ResearchOutputDict | None
    analysis: AnalysisOutputDict | None
    recommendation: RecommendationOutputDict | None

    # Workflow metadata
    status: str  # WorkflowStatus value
    current_agent: str | None  # AgentName value
    started_at: str  # ISO format datetime
    completed_at: str | None
    error: str | None

    # Accumulated messages and errors
    messages: list[dict[str, Any]]
    errors: list[str]

    # Context for passing additional data
    context: dict[str, Any]

    # Cache metrics (if prompt caching is enabled)
    cache_metrics: dict[str, Any] | None


# ============================================================================
# State Factory and Helpers
# ============================================================================


def create_initial_state(
    portfolio: Portfolio | dict[str, Any],
    user_request: str,
    user_id: str | None = None,
    trace_id: str | None = None,
) -> PortfolioState:
    """Create an initial workflow state.

    Args:
        portfolio: Portfolio data (model or dict).
        user_request: User's request/query.
        user_id: Optional user identifier.
        trace_id: Optional trace ID for observability.

    Returns:
        Initial PortfolioState ready for workflow execution.
    """
    workflow_id = str(uuid.uuid4())
    trace_id = trace_id or str(uuid.uuid4())

    # Convert portfolio to dict if needed
    if isinstance(portfolio, Portfolio):
        portfolio_dict = portfolio.model_dump(mode="json")
        symbols = portfolio.symbols
    else:
        portfolio_dict = portfolio
        symbols = [p.get("symbol", "") for p in portfolio.get("positions", [])]

    state: PortfolioState = {
        "workflow_id": workflow_id,
        "trace_id": trace_id,
        "user_id": user_id,
        "portfolio": portfolio_dict,
        "user_request": user_request,
        "symbols": symbols,
        "research": None,
        "analysis": None,
        "recommendation": None,
        "status": WorkflowStatus.PENDING.value,
        "current_agent": None,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "error": None,
        "messages": [],
        "errors": [],
        "context": {},
        "cache_metrics": None,
    }

    logger.info(
        "initial_state_created",
        workflow_id=workflow_id,
        trace_id=trace_id,
        symbol_count=len(symbols),
    )

    return state


def update_state_for_agent(
    state: PortfolioState,
    agent: AgentName,
) -> PortfolioState:
    """Update state when entering an agent.

    Args:
        state: Current workflow state.
        agent: Agent being entered.

    Returns:
        Updated state.
    """
    state["current_agent"] = agent.value
    state["status"] = WorkflowStatus.RUNNING.value
    return state


def update_state_with_result(
    state: PortfolioState,
    agent: AgentName,
    result: dict[str, Any],
) -> PortfolioState:
    """Update state with agent result.

    Args:
        state: Current workflow state.
        agent: Agent that produced the result.
        result: Agent output.

    Returns:
        Updated state.
    """
    if agent == AgentName.RESEARCH:
        state["research"] = result  # type: ignore[typeddict-item]
    elif agent == AgentName.ANALYSIS:
        state["analysis"] = result  # type: ignore[typeddict-item]
    elif agent == AgentName.RECOMMENDATION:
        state["recommendation"] = result  # type: ignore[typeddict-item]

    # Check for errors in result
    if result.get("errors"):
        state["errors"].extend(result["errors"])

    return state


def mark_state_completed(state: PortfolioState) -> PortfolioState:
    """Mark state as completed.

    Args:
        state: Current workflow state.

    Returns:
        Updated state.
    """
    state["status"] = WorkflowStatus.COMPLETED.value
    state["completed_at"] = datetime.now(UTC).isoformat()
    state["current_agent"] = None
    return state


def mark_state_failed(state: PortfolioState, error: str) -> PortfolioState:
    """Mark state as failed.

    Args:
        state: Current workflow state.
        error: Error message.

    Returns:
        Updated state.
    """
    state["status"] = WorkflowStatus.FAILED.value
    state["completed_at"] = datetime.now(UTC).isoformat()
    state["error"] = error
    state["errors"].append(error)
    return state


# ============================================================================
# State Validation
# ============================================================================


class StateValidationError(Exception):
    """Raised when state validation fails."""

    pass


def validate_state(state: PortfolioState) -> list[str]:
    """Validate workflow state.

    Args:
        state: State to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    # Required fields
    if not state.get("workflow_id"):
        errors.append("Missing workflow_id")
    if not state.get("trace_id"):
        errors.append("Missing trace_id")
    if not state.get("portfolio"):
        errors.append("Missing portfolio")
    if not state.get("user_request"):
        errors.append("Missing user_request")

    # Status must be valid
    status = state.get("status")
    if status and status not in [s.value for s in WorkflowStatus]:
        errors.append(f"Invalid status: {status}")

    # Current agent must be valid if set
    current_agent = state.get("current_agent")
    if current_agent and current_agent not in [a.value for a in AgentName]:
        errors.append(f"Invalid current_agent: {current_agent}")

    # Timestamps must be parseable
    started_at = state.get("started_at")
    if started_at:
        try:
            datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            errors.append(f"Invalid started_at format: {started_at}")

    completed_at = state.get("completed_at")
    if completed_at:
        try:
            datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            errors.append(f"Invalid completed_at format: {completed_at}")

    return errors


def validate_state_or_raise(state: PortfolioState) -> None:
    """Validate state and raise if invalid.

    Args:
        state: State to validate.

    Raises:
        StateValidationError: If validation fails.
    """
    errors = validate_state(state)
    if errors:
        raise StateValidationError(f"State validation failed: {'; '.join(errors)}")


# ============================================================================
# State Persistence (PostgreSQL)
# ============================================================================


class StatePersistence:
    """Handles state persistence to PostgreSQL.

    Uses asyncpg for async database operations.
    """

    def __init__(self, connection_string: str | None = None) -> None:
        """Initialize persistence handler.

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
                import asyncpg  # type: ignore[import-untyped]

                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=2,
                    max_size=10,
                )
                logger.info("database_pool_created")
            except ImportError:
                logger.warning("asyncpg_not_installed", message="State persistence disabled")
                raise
            except Exception as e:
                logger.error("database_pool_failed", error=str(e))
                raise

        return self._pool

    async def initialize_schema(self) -> None:
        """Create the workflow_states table if it doesn't exist."""
        pool = await self._get_pool()

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS workflow_states (
            workflow_id UUID PRIMARY KEY,
            trace_id UUID NOT NULL,
            user_id TEXT,
            portfolio JSONB NOT NULL,
            user_request TEXT NOT NULL,
            symbols TEXT[] NOT NULL,
            research JSONB,
            analysis JSONB,
            recommendation JSONB,
            status TEXT NOT NULL,
            current_agent TEXT,
            started_at TIMESTAMPTZ NOT NULL,
            completed_at TIMESTAMPTZ,
            error TEXT,
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            errors TEXT[] NOT NULL DEFAULT '{}',
            context JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_workflow_states_trace_id ON workflow_states(trace_id);
        CREATE INDEX IF NOT EXISTS idx_workflow_states_user_id ON workflow_states(user_id);
        CREATE INDEX IF NOT EXISTS idx_workflow_states_status ON workflow_states(status);
        CREATE INDEX IF NOT EXISTS idx_workflow_states_started_at ON workflow_states(started_at);
        """

        async with pool.acquire() as conn:
            await conn.execute(create_table_sql)
            logger.info("schema_initialized")

    async def save_state(self, state: PortfolioState) -> None:
        """Save or update workflow state.

        Args:
            state: State to save.
        """
        validate_state_or_raise(state)

        pool = await self._get_pool()

        upsert_sql = """
        INSERT INTO workflow_states (
            workflow_id, trace_id, user_id, portfolio, user_request, symbols,
            research, analysis, recommendation, status, current_agent,
            started_at, completed_at, error, messages, errors, context, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, NOW()
        )
        ON CONFLICT (workflow_id) DO UPDATE SET
            research = EXCLUDED.research,
            analysis = EXCLUDED.analysis,
            recommendation = EXCLUDED.recommendation,
            status = EXCLUDED.status,
            current_agent = EXCLUDED.current_agent,
            completed_at = EXCLUDED.completed_at,
            error = EXCLUDED.error,
            messages = EXCLUDED.messages,
            errors = EXCLUDED.errors,
            context = EXCLUDED.context,
            updated_at = NOW()
        """

        async with pool.acquire() as conn:
            await conn.execute(
                upsert_sql,
                uuid.UUID(state["workflow_id"]),
                uuid.UUID(state["trace_id"]),
                state.get("user_id"),
                json.dumps(state.get("portfolio", {})),
                state.get("user_request", ""),
                state.get("symbols", []),
                json.dumps(state.get("research")) if state.get("research") else None,
                json.dumps(state.get("analysis")) if state.get("analysis") else None,
                json.dumps(state.get("recommendation")) if state.get("recommendation") else None,
                state.get("status"),
                state.get("current_agent"),
                datetime.fromisoformat(state["started_at"].replace("Z", "+00:00")),
                (
                    datetime.fromisoformat(completed.replace("Z", "+00:00"))
                    if (completed := state.get("completed_at"))
                    else None
                ),
                state.get("error"),
                json.dumps(state.get("messages", [])),
                state.get("errors", []),
                json.dumps(state.get("context", {})),
            )

        logger.debug("state_saved", workflow_id=state["workflow_id"])

    async def load_state(self, workflow_id: str) -> PortfolioState | None:
        """Load workflow state by ID.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            State if found, None otherwise.
        """
        pool = await self._get_pool()

        select_sql = """
        SELECT workflow_id, trace_id, user_id, portfolio, user_request, symbols,
               research, analysis, recommendation, status, current_agent,
               started_at, completed_at, error, messages, errors, context
        FROM workflow_states
        WHERE workflow_id = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(select_sql, uuid.UUID(workflow_id))

        if not row:
            return None

        state: PortfolioState = {
            "workflow_id": str(row["workflow_id"]),
            "trace_id": str(row["trace_id"]),
            "user_id": row["user_id"],
            "portfolio": json.loads(row["portfolio"]),
            "user_request": row["user_request"],
            "symbols": list(row["symbols"]),
            "research": json.loads(row["research"]) if row["research"] else None,
            "analysis": json.loads(row["analysis"]) if row["analysis"] else None,
            "recommendation": json.loads(row["recommendation"]) if row["recommendation"] else None,
            "status": row["status"],
            "current_agent": row["current_agent"],
            "started_at": row["started_at"].isoformat(),
            "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
            "error": row["error"],
            "messages": json.loads(row["messages"]),
            "errors": list(row["errors"]),
            "context": json.loads(row["context"]),
        }

        logger.debug("state_loaded", workflow_id=workflow_id)
        return state

    async def list_states(
        self,
        user_id: str | None = None,
        status: WorkflowStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PortfolioState]:
        """List workflow states with optional filters.

        Args:
            user_id: Filter by user ID.
            status: Filter by status.
            limit: Maximum results to return.
            offset: Pagination offset.

        Returns:
            List of matching states.
        """
        pool = await self._get_pool()

        # Build query with filters
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status.value)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        select_sql = f"""
        SELECT workflow_id, trace_id, user_id, portfolio, user_request, symbols,
               research, analysis, recommendation, status, current_agent,
               started_at, completed_at, error, messages, errors, context
        FROM workflow_states
        {where_clause}
        ORDER BY started_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with pool.acquire() as conn:
            rows = await conn.fetch(select_sql, *params)

        states = []
        for row in rows:
            state: PortfolioState = {
                "workflow_id": str(row["workflow_id"]),
                "trace_id": str(row["trace_id"]),
                "user_id": row["user_id"],
                "portfolio": json.loads(row["portfolio"]),
                "user_request": row["user_request"],
                "symbols": list(row["symbols"]),
                "research": json.loads(row["research"]) if row["research"] else None,
                "analysis": json.loads(row["analysis"]) if row["analysis"] else None,
                "recommendation": json.loads(row["recommendation"])
                if row["recommendation"]
                else None,
                "status": row["status"],
                "current_agent": row["current_agent"],
                "started_at": row["started_at"].isoformat(),
                "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
                "error": row["error"],
                "messages": json.loads(row["messages"]),
                "errors": list(row["errors"]),
                "context": json.loads(row["context"]),
            }
            states.append(state)

        return states

    async def delete_state(self, workflow_id: str) -> bool:
        """Delete workflow state.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            True if deleted, False if not found.
        """
        pool = await self._get_pool()

        delete_sql = "DELETE FROM workflow_states WHERE workflow_id = $1"

        async with pool.acquire() as conn:
            result = await conn.execute(delete_sql, uuid.UUID(workflow_id))

        deleted: bool = result == "DELETE 1"
        if deleted:
            logger.debug("state_deleted", workflow_id=workflow_id)

        return deleted

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("database_pool_closed")


# ============================================================================
# Convenience Functions for State Serialization
# ============================================================================


def state_to_json(state: PortfolioState) -> str:
    """Serialize state to JSON string.

    Args:
        state: State to serialize.

    Returns:
        JSON string representation.
    """
    return json.dumps(state, default=str)


def state_from_json(json_str: str) -> PortfolioState:
    """Deserialize state from JSON string.

    Args:
        json_str: JSON string.

    Returns:
        Deserialized state.
    """
    data: PortfolioState = json.loads(json_str)
    return data


def get_state_summary(state: PortfolioState) -> dict[str, Any]:
    """Get a summary of the state for logging/display.

    Args:
        state: State to summarize.

    Returns:
        Summary dict.
    """
    return {
        "workflow_id": state.get("workflow_id"),
        "status": state.get("status"),
        "current_agent": state.get("current_agent"),
        "symbol_count": len(state.get("symbols", [])),
        "has_research": state.get("research") is not None,
        "has_analysis": state.get("analysis") is not None,
        "has_recommendation": state.get("recommendation") is not None,
        "error_count": len(state.get("errors", [])),
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
    }
