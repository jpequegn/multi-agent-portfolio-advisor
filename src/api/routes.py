"""FastAPI routes for the Portfolio Advisor API.

This module provides:
- /analyze endpoint for running portfolio analysis
- /analyze/{trace_id} for retrieving analysis results
- Health check endpoints integration
- Request/response models with validation
- CORS configuration
- Error handling
"""

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langfuse import get_client as get_langfuse_client
from langfuse import observe
from pydantic import BaseModel, Field

from src.api.health import (
    HealthService,
    ServiceStatus,
    get_health_service,
    set_health_service,
)
from src.observability.tracing import TraceContext
from src.orchestration.state import (
    Portfolio,
    PortfolioState,
    Position,
    WorkflowStatus,
    create_initial_state,
)
from src.orchestration.workflow import create_workflow

logger = structlog.get_logger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class PositionRequest(BaseModel):
    """Request model for a portfolio position."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "cost_basis": 150.00,
                    "sector": "Technology",
                },
                {
                    "symbol": "VTI",
                    "quantity": 50,
                },
            ]
        }
    }

    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL, GOOGL)", min_length=1, max_length=10)
    quantity: float = Field(..., description="Number of shares held", gt=0)
    cost_basis: float | None = Field(default=None, description="Average cost per share in USD", ge=0)
    sector: str | None = Field(default=None, description="Sector classification (e.g., Technology, Healthcare)")


class PortfolioRequest(BaseModel):
    """Request model for portfolio analysis."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "positions": [
                        {"symbol": "AAPL", "quantity": 100, "cost_basis": 150.00, "sector": "Technology"},
                        {"symbol": "GOOGL", "quantity": 50, "cost_basis": 2800.00, "sector": "Technology"},
                        {"symbol": "JNJ", "quantity": 75, "cost_basis": 160.00, "sector": "Healthcare"},
                    ],
                    "user_request": "Analyze portfolio risk and suggest rebalancing to reduce tech concentration",
                    "total_value": 250000.00,
                    "cash": 10000.00,
                    "account_type": "taxable",
                    "user_id": "user-123",
                }
            ]
        }
    }

    positions: list[PositionRequest] = Field(
        ..., description="List of portfolio positions (at least one required)", min_length=1
    )
    user_request: str = Field(
        default="Analyze risk and suggest rebalancing",
        description="Natural language request describing what analysis you want",
        min_length=1,
        max_length=1000,
    )
    total_value: float | None = Field(
        default=None, description="Total portfolio value in USD (calculated from positions if not provided)", ge=0
    )
    cash: float = Field(default=0.0, description="Available cash balance in USD", ge=0)
    account_type: str = Field(
        default="taxable", description="Account type: taxable, ira, roth_ira, 401k"
    )
    user_id: str | None = Field(default=None, description="Optional user identifier for tracking")


# ============================================================================
# Response Models
# ============================================================================


class ResearchOutput(BaseModel):
    """Research agent output containing market data and news."""

    market_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Market data for each symbol including price, volume, and metrics",
    )
    news: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent news articles relevant to portfolio holdings",
    )
    summary: str = Field(default="", description="AI-generated summary of research findings")
    symbols_researched: list[str] = Field(
        default_factory=list, description="List of symbols that were researched"
    )


class AnalysisOutput(BaseModel):
    """Analysis agent output containing risk metrics and portfolio analysis."""

    risk_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Risk metrics including volatility, beta, Sharpe ratio, max drawdown",
    )
    correlations: dict[str, Any] = Field(
        default_factory=dict,
        description="Correlation matrix between portfolio holdings",
    )
    benchmark_comparison: dict[str, Any] = Field(
        default_factory=dict,
        description="Comparison of portfolio performance vs benchmarks (S&P 500, etc.)",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="List of AI-generated recommendations based on analysis",
    )
    summary: str = Field(default="", description="AI-generated summary of portfolio analysis")


class RecommendationOutput(BaseModel):
    """Recommendation agent output containing trade suggestions."""

    trades: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of recommended trades with symbol, action (BUY/SELL/HOLD), quantity, and rationale",
    )
    summary: str = Field(default="", description="AI-generated summary of recommendations")
    total_trades: int = Field(default=0, description="Total number of trade recommendations")
    buy_count: int = Field(default=0, description="Number of BUY recommendations")
    sell_count: int = Field(default=0, description="Number of SELL recommendations")
    hold_count: int = Field(default=0, description="Number of HOLD recommendations")


class ApprovalInfo(BaseModel):
    """Information about approval status for high-risk recommendations."""

    required: bool = Field(
        default=False, description="Whether approval is required"
    )
    approval_id: str | None = Field(
        default=None, description="Approval request ID if pending"
    )
    status: str | None = Field(
        default=None, description="Approval status: pending, approved, rejected, expired"
    )
    risk_level: str | None = Field(
        default=None, description="Risk level: low, medium, high, critical"
    )
    triggers: list[str] = Field(
        default_factory=list, description="Reasons that triggered approval requirement"
    )


class AnalysisResponse(BaseModel):
    """Response model for portfolio analysis."""

    workflow_id: str = Field(..., description="Unique workflow identifier")
    trace_id: str = Field(..., description="Trace ID for observability")
    status: str = Field(..., description="Workflow status")
    research: ResearchOutput | None = Field(default=None, description="Research results")
    analysis: AnalysisOutput | None = Field(default=None, description="Analysis results")
    recommendations: RecommendationOutput | None = Field(
        default=None, description="Recommendations"
    )
    approval: ApprovalInfo | None = Field(
        default=None, description="Approval information for high-risk recommendations"
    )
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    latency_ms: float = Field(..., description="Total processing time in milliseconds")
    started_at: str = Field(..., description="When processing started")
    completed_at: str | None = Field(default=None, description="When processing completed")


class AnalysisSummary(BaseModel):
    """Summary of an analysis for listing."""

    workflow_id: str
    trace_id: str
    status: str
    symbol_count: int
    started_at: str
    completed_at: str | None
    has_errors: bool


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    trace_id: str | None = Field(default=None, description="Trace ID if available")


# ============================================================================
# Application Setup
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Application lifespan handler."""
    # Startup
    logger.info("application_starting")

    # Initialize health service if not already set
    if get_health_service() is None:
        health_service = HealthService(version=app.version)
        set_health_service(health_service)

    yield

    # Shutdown
    logger.info("application_shutting_down")


OPENAPI_TAGS = [
    {
        "name": "Analysis",
        "description": "Portfolio analysis endpoints. Submit portfolios for AI-powered analysis "
        "including risk assessment, market research, and trade recommendations.",
    },
    {
        "name": "Health",
        "description": "Health check endpoints for monitoring service status and readiness. "
        "Compatible with Kubernetes liveness and readiness probes.",
    },
]

API_DESCRIPTION = """
## Overview

The Portfolio Advisor API provides AI-powered portfolio analysis using a multi-agent system.
Submit your portfolio holdings and receive comprehensive analysis including:

- **Market Research**: Current market data, news, and sentiment analysis
- **Risk Analysis**: Volatility, correlations, benchmark comparisons
- **Recommendations**: AI-generated trade suggestions with rationale

## Quick Start

```bash
curl -X POST http://localhost:8000/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "positions": [
      {"symbol": "AAPL", "quantity": 100},
      {"symbol": "GOOGL", "quantity": 50}
    ]
  }'
```

## Observability

Each analysis request returns a `trace_id` that can be used to:
- Track the request through the system
- View detailed traces in Langfuse
- Debug issues and analyze performance

## Rate Limits

- 100 requests per minute per IP
- 10 concurrent analysis requests per user
"""


def create_app(
    title: str = "Portfolio Advisor API",
    version: str = "1.0.0",
    description: str | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title.
        version: API version.
        description: API description (uses default if not provided).
        cors_origins: Allowed CORS origins.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        version=version,
        description=description or API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
        contact={
            "name": "Portfolio Advisor Team",
            "url": "https://github.com/jpequegn/multi-agent-portfolio-advisor",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Configure CORS
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException  # noqa: ARG001
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                detail=None,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception  # noqa: ARG001
    ) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc) if app.debug else None,
            ).model_dump(),
        )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register all routes on the application.

    Args:
        app: FastAPI application.
    """
    # Import and include approval routes
    from src.api.approvals import router as approvals_router

    app.include_router(approvals_router)

    # Import and include streaming routes
    from src.api.streaming import router as streaming_router

    app.include_router(streaming_router)

    # Health endpoints
    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, Any]:
        """Basic liveness check.

        Returns simple status to confirm service is running.
        """
        health_service = get_health_service()
        if health_service:
            return await health_service.liveness()
        return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}

    @app.get("/health/live", tags=["Health"])
    async def liveness() -> dict[str, Any]:
        """Kubernetes-style liveness probe.

        Alias for /health endpoint.
        """
        return await health()

    @app.get("/health/ready", tags=["Health"])
    async def readiness() -> dict[str, Any]:
        """Full readiness check.

        Checks all dependencies and returns detailed status.
        """
        health_service = get_health_service()
        if health_service:
            result = await health_service.readiness()
            status_code = 200 if result.status != ServiceStatus.NOT_READY else 503
            return JSONResponse(content=result.to_dict(), status_code=status_code)
        return {"status": "ready", "checks": {}, "timestamp": datetime.now(UTC).isoformat()}

    # Analysis endpoints
    @app.post(
        "/analyze",
        response_model=AnalysisResponse,
        tags=["Analysis"],
        responses={
            200: {"description": "Analysis completed successfully"},
            400: {"description": "Invalid request", "model": ErrorResponse},
            500: {"description": "Internal error", "model": ErrorResponse},
        },
    )
    async def analyze_portfolio(request: PortfolioRequest) -> AnalysisResponse:
        """Run full portfolio analysis.

        Executes the multi-agent workflow to analyze the portfolio
        and generate recommendations.
        """
        start_time = time.monotonic()
        symbols = [p.symbol.upper() for p in request.positions]

        # Create trace context with metadata
        async with TraceContext(
            session_id=request.user_id,
            user_id=request.user_id,
            metadata={
                "portfolio_size": len(request.positions),
                "symbols": symbols,
                "account_type": request.account_type,
                "has_cash": request.cash > 0,
            },
            tags=["portfolio-analysis", request.account_type],
        ):
            try:
                # Convert request to Portfolio model
                positions = [
                    Position(
                        symbol=p.symbol.upper(),
                        quantity=p.quantity,
                        cost_basis=p.cost_basis,
                        sector=p.sector,
                    )
                    for p in request.positions
                ]

                portfolio = Portfolio(
                    positions=positions,
                    total_value=request.total_value or sum(p.quantity for p in positions),
                    cash=request.cash,
                    account_type=request.account_type,
                )

                # Create initial state
                state = create_initial_state(
                    portfolio=portfolio,
                    user_request=request.user_request,
                    user_id=request.user_id,
                )

                logger.info(
                    "analysis_started",
                    workflow_id=state["workflow_id"],
                    trace_id=state["trace_id"],
                    symbol_count=len(state["symbols"]),
                )

                # Create and run workflow with tracing
                workflow = create_workflow()
                result_state: PortfolioState = await _run_traced_workflow(
                    workflow, state, symbols
                )

                latency_ms = (time.monotonic() - start_time) * 1000

                # Score the trace based on success
                _score_trace(result_state, latency_ms)

                # Check if recommendations require approval
                approval_info = await _check_approval_required(
                    result_state, portfolio.total_value
                )

                logger.info(
                    "analysis_completed",
                    workflow_id=result_state["workflow_id"],
                    status=result_state.get("status"),
                    latency_ms=round(latency_ms, 2),
                    requires_approval=approval_info.required if approval_info else False,
                )

                return _state_to_response(result_state, latency_ms, approval_info)

            except Exception as e:
                latency_ms = (time.monotonic() - start_time) * 1000
                logger.error("analysis_failed", error=str(e), latency_ms=round(latency_ms, 2))

                # Record error in trace
                _record_trace_error(str(e))

                raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}") from e

    @app.get(
        "/analyze/{trace_id}",
        response_model=AnalysisResponse,
        tags=["Analysis"],
        responses={
            200: {"description": "Analysis found"},
            404: {"description": "Analysis not found", "model": ErrorResponse},
        },
    )
    async def get_analysis(trace_id: str) -> AnalysisResponse:
        """Retrieve a previous analysis by trace ID.

        Note: Requires state persistence to be configured.
        """
        # TODO: Implement state persistence lookup
        # For now, return 404 as we don't have persistence enabled by default
        raise HTTPException(
            status_code=404,
            detail=f"Analysis with trace_id '{trace_id}' not found. "
            "State persistence may not be configured.",
        )

    @app.get(
        "/analyze",
        response_model=list[AnalysisSummary],
        tags=["Analysis"],
        responses={
            200: {"description": "List of analyses"},
        },
    )
    async def list_analyses(
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AnalysisSummary]:
        """List previous analyses.

        Note: Requires state persistence to be configured.
        """
        # TODO: Implement state persistence lookup
        # For now, return empty list
        _ = (user_id, status, limit, offset)  # Mark as used
        return []


# ============================================================================
# Tracing Helper Functions
# ============================================================================


@observe(name="portfolio_workflow")
async def _run_traced_workflow(
    workflow: Any, state: PortfolioState, symbols: list[str]
) -> PortfolioState:
    """Run workflow with Langfuse tracing.

    Creates a traced span for the entire workflow execution.

    Args:
        workflow: LangGraph workflow instance.
        state: Initial workflow state.
        symbols: List of symbols being analyzed.

    Returns:
        Final workflow state.
    """
    # Add workflow metadata to current span
    try:
        langfuse = get_langfuse_client()
        langfuse.trace(
            name="portfolio_analysis",
            metadata={
                "workflow_id": state.get("workflow_id"),
                "trace_id": state.get("trace_id"),
                "symbols": symbols,
                "symbol_count": len(symbols),
            },
        )
    except Exception:
        # Tracing is optional, don't fail if unavailable
        pass

    result: PortfolioState = await workflow.ainvoke(state)
    return result


def _score_trace(state: PortfolioState, latency_ms: float) -> None:
    """Score the current trace based on workflow results.

    Args:
        state: Final workflow state.
        latency_ms: Total latency in milliseconds.
    """
    try:
        langfuse = get_langfuse_client()

        # Score based on completion success
        has_errors = bool(state.get("errors"))
        status = state.get("status", "unknown")
        success = status == WorkflowStatus.COMPLETED.value and not has_errors

        langfuse.score(
            name="completion_success",
            value=1.0 if success else 0.0,
            comment=f"Status: {status}, Errors: {len(state.get('errors', []))}",
        )

        # Score latency (1.0 for <5s, 0.5 for <15s, 0.0 for >15s)
        if latency_ms < 5000:
            latency_score = 1.0
        elif latency_ms < 15000:
            latency_score = 0.5
        else:
            latency_score = 0.0

        langfuse.score(
            name="latency_quality",
            value=latency_score,
            comment=f"Latency: {latency_ms:.0f}ms",
        )

    except Exception as e:
        logger.debug("trace_scoring_failed", error=str(e))


def _record_trace_error(error_message: str) -> None:
    """Record an error in the current trace.

    Args:
        error_message: Error message to record.
    """
    try:
        langfuse = get_langfuse_client()
        langfuse.score(
            name="completion_success",
            value=0.0,
            comment=f"Error: {error_message[:200]}",
        )
    except Exception:
        # Tracing is optional
        pass


# ============================================================================
# Approval Helper Functions
# ============================================================================


async def _check_approval_required(
    state: PortfolioState,
    portfolio_value: float,
) -> ApprovalInfo | None:
    """Check if recommendations require human approval.

    Args:
        state: Final workflow state with recommendations.
        portfolio_value: Total portfolio value.

    Returns:
        ApprovalInfo if recommendations exist, None otherwise.
    """
    recommendation = state.get("recommendation")
    if not recommendation:
        return None

    trades = recommendation.get("trades", [])
    if not trades:
        return ApprovalInfo(required=False)

    compliance = recommendation.get("compliance", {})

    # Import approval manager
    from src.approval import get_approval_manager

    manager = get_approval_manager()

    # Evaluate if approval is required
    evaluation = manager.evaluate_recommendation(
        trades=trades,
        portfolio_value=portfolio_value,
        compliance=compliance,
        analysis_context=state.get("analysis", {}),
    )

    if not evaluation.requires_approval:
        return ApprovalInfo(
            required=False,
            risk_level=evaluation.risk_level.value,
            triggers=[t.value for t in evaluation.triggers],
        )

    # Create approval request
    approval_request = await manager.create_approval_request(
        workflow_id=state["workflow_id"],
        trace_id=state["trace_id"],
        trades=trades,
        portfolio_value=portfolio_value,
        compliance=compliance,
        summary=recommendation.get("summary", ""),
        analysis_context=state.get("analysis", {}),
        user_id=state.get("user_id"),
    )

    return ApprovalInfo(
        required=True,
        approval_id=approval_request.approval_id,
        status=approval_request.status.value,
        risk_level=approval_request.risk_level.value,
        triggers=[t.value for t in approval_request.triggers],
    )


# ============================================================================
# Response Helper Functions
# ============================================================================


def _state_to_response(
    state: PortfolioState,
    latency_ms: float,
    approval_info: ApprovalInfo | None = None,
) -> AnalysisResponse:
    """Convert workflow state to API response.

    Args:
        state: Workflow state.
        latency_ms: Processing latency.
        approval_info: Optional approval information.

    Returns:
        AnalysisResponse.
    """
    research = None
    if state.get("research"):
        research_data = state["research"]
        research = ResearchOutput(
            market_data=research_data.get("market_data", {}),
            news=research_data.get("news", []),
            summary=research_data.get("summary", ""),
            symbols_researched=research_data.get("symbols_researched", []),
        )

    analysis = None
    if state.get("analysis"):
        analysis_data = state["analysis"]
        analysis = AnalysisOutput(
            risk_metrics=analysis_data.get("risk_metrics", {}),
            correlations=analysis_data.get("correlations", {}),
            benchmark_comparison=analysis_data.get("benchmark_comparison", {}),
            recommendations=analysis_data.get("recommendations", []),
            summary=analysis_data.get("summary", ""),
        )

    recommendations = None
    if state.get("recommendation"):
        rec_data = state["recommendation"]
        recommendations = RecommendationOutput(
            trades=rec_data.get("trades", []),
            summary=rec_data.get("summary", ""),
            total_trades=rec_data.get("total_trades", 0),
            buy_count=rec_data.get("buy_count", 0),
            sell_count=rec_data.get("sell_count", 0),
            hold_count=rec_data.get("hold_count", 0),
        )

    return AnalysisResponse(
        workflow_id=state["workflow_id"],
        trace_id=state["trace_id"],
        status=state.get("status", WorkflowStatus.PENDING.value),
        research=research,
        analysis=analysis,
        recommendations=recommendations,
        approval=approval_info,
        errors=state.get("errors", []),
        latency_ms=round(latency_ms, 2),
        started_at=state.get("started_at", datetime.now(UTC).isoformat()),
        completed_at=state.get("completed_at"),
    )


# ============================================================================
# Default Application Instance
# ============================================================================


# Create default app instance
app = create_app()
