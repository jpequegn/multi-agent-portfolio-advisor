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
from pydantic import BaseModel, Field

from src.api.health import (
    HealthService,
    ServiceStatus,
    get_health_service,
    set_health_service,
)
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

    symbol: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)
    quantity: float = Field(..., description="Number of shares", gt=0)
    cost_basis: float | None = Field(default=None, description="Average cost per share", ge=0)
    sector: str | None = Field(default=None, description="Sector classification")


class PortfolioRequest(BaseModel):
    """Request model for portfolio analysis."""

    positions: list[PositionRequest] = Field(
        ..., description="List of portfolio positions", min_length=1
    )
    user_request: str = Field(
        default="Analyze risk and suggest rebalancing",
        description="Analysis request",
        min_length=1,
        max_length=1000,
    )
    total_value: float | None = Field(default=None, description="Total portfolio value", ge=0)
    cash: float = Field(default=0.0, description="Cash balance", ge=0)
    account_type: str = Field(default="taxable", description="Account type")
    user_id: str | None = Field(default=None, description="Optional user identifier")


# ============================================================================
# Response Models
# ============================================================================


class ResearchOutput(BaseModel):
    """Research agent output."""

    market_data: dict[str, Any] = Field(default_factory=dict)
    news: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    symbols_researched: list[str] = Field(default_factory=list)


class AnalysisOutput(BaseModel):
    """Analysis agent output."""

    risk_metrics: dict[str, Any] = Field(default_factory=dict)
    correlations: dict[str, Any] = Field(default_factory=dict)
    benchmark_comparison: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    summary: str = Field(default="")


class RecommendationOutput(BaseModel):
    """Recommendation agent output."""

    trades: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = Field(default="")
    total_trades: int = Field(default=0)
    buy_count: int = Field(default=0)
    sell_count: int = Field(default=0)
    hold_count: int = Field(default=0)


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


def create_app(
    title: str = "Portfolio Advisor API",
    version: str = "1.0.0",
    description: str = "Multi-agent portfolio analysis and recommendations",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title.
        version: API version.
        description: API description.
        cors_origins: Allowed CORS origins.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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

            # Create and run workflow
            workflow = create_workflow()
            result_state: PortfolioState = await workflow.ainvoke(state)

            latency_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "analysis_completed",
                workflow_id=result_state["workflow_id"],
                status=result_state.get("status"),
                latency_ms=round(latency_ms, 2),
            )

            return _state_to_response(result_state, latency_ms)

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error("analysis_failed", error=str(e), latency_ms=round(latency_ms, 2))
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
# Helper Functions
# ============================================================================


def _state_to_response(state: PortfolioState, latency_ms: float) -> AnalysisResponse:
    """Convert workflow state to API response.

    Args:
        state: Workflow state.
        latency_ms: Processing latency.

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
