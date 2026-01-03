"""Tests for FastAPI routes."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.health import reset_health_service
from src.api.routes import (
    AnalysisOutput,
    AnalysisResponse,
    AnalysisSummary,
    ErrorResponse,
    PortfolioRequest,
    PositionRequest,
    RecommendationOutput,
    ResearchOutput,
    _state_to_response,
    create_app,
)
from src.orchestration.state import WorkflowStatus

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app():
    """Create test FastAPI app."""
    reset_health_service()
    return create_app(title="Test API", version="0.1.0")


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_position_request():
    """Sample position request data."""
    return {
        "symbol": "AAPL",
        "quantity": 100.0,
        "cost_basis": 150.0,
        "sector": "Technology",
    }


@pytest.fixture
def sample_portfolio_request(sample_position_request):
    """Sample portfolio request data."""
    return {
        "positions": [
            sample_position_request,
            {"symbol": "GOOGL", "quantity": 50.0},
            {"symbol": "MSFT", "quantity": 75.0, "cost_basis": 300.0},
        ],
        "user_request": "Analyze risk and suggest rebalancing",
        "total_value": 100000.0,
        "cash": 5000.0,
        "account_type": "taxable",
        "user_id": "test-user-123",
    }


@pytest.fixture
def sample_workflow_state():
    """Sample workflow state for testing response conversion."""
    return {
        "workflow_id": "wf-123",
        "trace_id": "trace-456",
        "status": WorkflowStatus.COMPLETED.value,
        "started_at": "2024-01-15T10:00:00Z",
        "completed_at": "2024-01-15T10:00:05Z",
        "errors": [],
        "research": {
            "market_data": {"AAPL": {"price": 180.0}},
            "news": [{"title": "Apple earnings beat"}],
            "summary": "Market conditions favorable",
            "symbols_researched": ["AAPL", "GOOGL"],
        },
        "analysis": {
            "risk_metrics": {"volatility": 0.15, "beta": 1.1},
            "correlations": {"AAPL-GOOGL": 0.7},
            "benchmark_comparison": {"vs_spy": 0.05},
            "recommendations": ["Reduce tech exposure"],
            "summary": "Portfolio has moderate risk",
        },
        "recommendation": {
            "trades": [{"symbol": "AAPL", "action": "SELL", "quantity": 10}],
            "summary": "Suggest selling some AAPL",
            "total_trades": 1,
            "buy_count": 0,
            "sell_count": 1,
            "hold_count": 2,
        },
    }


# ============================================================================
# Request Model Tests
# ============================================================================


class TestPositionRequest:
    """Tests for PositionRequest model."""

    def test_valid_position(self):
        """Test valid position creation."""
        pos = PositionRequest(
            symbol="AAPL",
            quantity=100.0,
            cost_basis=150.0,
            sector="Technology",
        )
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100.0
        assert pos.cost_basis == 150.0
        assert pos.sector == "Technology"

    def test_position_minimal(self):
        """Test position with only required fields."""
        pos = PositionRequest(symbol="MSFT", quantity=50.0)
        assert pos.symbol == "MSFT"
        assert pos.quantity == 50.0
        assert pos.cost_basis is None
        assert pos.sector is None

    def test_position_symbol_validation(self):
        """Test symbol length validation."""
        # Valid symbols
        PositionRequest(symbol="A", quantity=10.0)  # min length
        PositionRequest(symbol="ABCDEFGHIJ", quantity=10.0)  # max length

        # Invalid - empty
        with pytest.raises(ValueError):
            PositionRequest(symbol="", quantity=10.0)

        # Invalid - too long
        with pytest.raises(ValueError):
            PositionRequest(symbol="TOOLONGSYMBOL", quantity=10.0)

    def test_position_quantity_validation(self):
        """Test quantity must be positive."""
        with pytest.raises(ValueError):
            PositionRequest(symbol="AAPL", quantity=0)

        with pytest.raises(ValueError):
            PositionRequest(symbol="AAPL", quantity=-10.0)

    def test_position_cost_basis_validation(self):
        """Test cost basis must be non-negative."""
        PositionRequest(symbol="AAPL", quantity=10.0, cost_basis=0.0)  # zero is ok

        with pytest.raises(ValueError):
            PositionRequest(symbol="AAPL", quantity=10.0, cost_basis=-50.0)


class TestPortfolioRequest:
    """Tests for PortfolioRequest model."""

    def test_valid_portfolio(self, sample_portfolio_request):
        """Test valid portfolio creation."""
        req = PortfolioRequest(**sample_portfolio_request)
        assert len(req.positions) == 3
        assert req.user_request == "Analyze risk and suggest rebalancing"
        assert req.total_value == 100000.0
        assert req.cash == 5000.0
        assert req.account_type == "taxable"
        assert req.user_id == "test-user-123"

    def test_portfolio_defaults(self):
        """Test portfolio with default values."""
        req = PortfolioRequest(
            positions=[PositionRequest(symbol="AAPL", quantity=100.0)]
        )
        assert req.user_request == "Analyze risk and suggest rebalancing"
        assert req.total_value is None
        assert req.cash == 0.0
        assert req.account_type == "taxable"
        assert req.user_id is None

    def test_portfolio_empty_positions(self):
        """Test portfolio requires at least one position."""
        with pytest.raises(ValueError):
            PortfolioRequest(positions=[])

    def test_portfolio_user_request_validation(self):
        """Test user request length validation."""
        positions = [PositionRequest(symbol="AAPL", quantity=100.0)]

        # Empty request
        with pytest.raises(ValueError):
            PortfolioRequest(positions=positions, user_request="")

        # Too long request
        with pytest.raises(ValueError):
            PortfolioRequest(positions=positions, user_request="x" * 1001)


# ============================================================================
# Response Model Tests
# ============================================================================


class TestResearchOutput:
    """Tests for ResearchOutput model."""

    def test_research_output_defaults(self):
        """Test research output with defaults."""
        output = ResearchOutput()
        assert output.market_data == {}
        assert output.news == []
        assert output.summary == ""
        assert output.symbols_researched == []

    def test_research_output_with_data(self):
        """Test research output with data."""
        output = ResearchOutput(
            market_data={"AAPL": {"price": 180.0}},
            news=[{"title": "News item"}],
            summary="Test summary",
            symbols_researched=["AAPL"],
        )
        assert output.market_data == {"AAPL": {"price": 180.0}}
        assert len(output.news) == 1
        assert output.summary == "Test summary"
        assert output.symbols_researched == ["AAPL"]


class TestAnalysisOutput:
    """Tests for AnalysisOutput model."""

    def test_analysis_output_defaults(self):
        """Test analysis output with defaults."""
        output = AnalysisOutput()
        assert output.risk_metrics == {}
        assert output.correlations == {}
        assert output.benchmark_comparison == {}
        assert output.recommendations == []
        assert output.summary == ""

    def test_analysis_output_with_data(self):
        """Test analysis output with data."""
        output = AnalysisOutput(
            risk_metrics={"volatility": 0.15},
            correlations={"AAPL-MSFT": 0.7},
            benchmark_comparison={"vs_spy": 0.05},
            recommendations=["Reduce risk"],
            summary="Portfolio analysis",
        )
        assert output.risk_metrics["volatility"] == 0.15
        assert len(output.recommendations) == 1


class TestRecommendationOutput:
    """Tests for RecommendationOutput model."""

    def test_recommendation_output_defaults(self):
        """Test recommendation output with defaults."""
        output = RecommendationOutput()
        assert output.trades == []
        assert output.summary == ""
        assert output.total_trades == 0
        assert output.buy_count == 0
        assert output.sell_count == 0
        assert output.hold_count == 0

    def test_recommendation_output_with_data(self):
        """Test recommendation output with data."""
        output = RecommendationOutput(
            trades=[{"symbol": "AAPL", "action": "BUY"}],
            summary="Buy recommendations",
            total_trades=3,
            buy_count=1,
            sell_count=1,
            hold_count=1,
        )
        assert len(output.trades) == 1
        assert output.total_trades == 3
        assert output.buy_count == 1


class TestAnalysisResponse:
    """Tests for AnalysisResponse model."""

    def test_analysis_response_required_fields(self):
        """Test analysis response with required fields."""
        response = AnalysisResponse(
            workflow_id="wf-123",
            trace_id="trace-456",
            status="completed",
            latency_ms=1500.5,
            started_at="2024-01-15T10:00:00Z",
        )
        assert response.workflow_id == "wf-123"
        assert response.trace_id == "trace-456"
        assert response.status == "completed"
        assert response.latency_ms == 1500.5
        assert response.research is None
        assert response.analysis is None
        assert response.recommendations is None
        assert response.errors == []
        assert response.completed_at is None

    def test_analysis_response_full(self):
        """Test analysis response with all fields."""
        response = AnalysisResponse(
            workflow_id="wf-123",
            trace_id="trace-456",
            status="completed",
            research=ResearchOutput(summary="Research done"),
            analysis=AnalysisOutput(summary="Analysis done"),
            recommendations=RecommendationOutput(summary="Recs done"),
            errors=["Minor warning"],
            latency_ms=2000.0,
            started_at="2024-01-15T10:00:00Z",
            completed_at="2024-01-15T10:00:02Z",
        )
        assert response.research is not None
        assert response.analysis is not None
        assert response.recommendations is not None
        assert len(response.errors) == 1


class TestAnalysisSummary:
    """Tests for AnalysisSummary model."""

    def test_analysis_summary(self):
        """Test analysis summary model."""
        summary = AnalysisSummary(
            workflow_id="wf-123",
            trace_id="trace-456",
            status="completed",
            symbol_count=5,
            started_at="2024-01-15T10:00:00Z",
            completed_at="2024-01-15T10:00:02Z",
            has_errors=False,
        )
        assert summary.workflow_id == "wf-123"
        assert summary.symbol_count == 5
        assert not summary.has_errors


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_error_response_minimal(self):
        """Test error response with minimal fields."""
        error = ErrorResponse(error="Something went wrong")
        assert error.error == "Something went wrong"
        assert error.detail is None
        assert error.trace_id is None

    def test_error_response_full(self):
        """Test error response with all fields."""
        error = ErrorResponse(
            error="Analysis failed",
            detail="Connection timeout",
            trace_id="trace-789",
        )
        assert error.error == "Analysis failed"
        assert error.detail == "Connection timeout"
        assert error.trace_id == "trace-789"


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_basic(self, client):
        """Test basic /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_live(self, client):
        """Test /health/live endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_ready(self, client):
        """Test /health/ready endpoint."""
        response = client.get("/health/ready")
        # Should return ready when no checkers are registered
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ready", "degraded", "not_ready"]


# ============================================================================
# Analysis Endpoint Tests
# ============================================================================


class TestAnalyzeEndpoint:
    """Tests for /analyze endpoint."""

    @patch("src.api.routes.create_workflow")
    def test_analyze_success(self, mock_create_workflow, client, sample_portfolio_request, sample_workflow_state):
        """Test successful portfolio analysis."""
        # Mock the workflow
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = sample_workflow_state
        mock_create_workflow.return_value = mock_workflow

        response = client.post("/analyze", json=sample_portfolio_request)

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "wf-123"
        assert data["trace_id"] == "trace-456"
        assert data["status"] == "completed"
        assert data["research"] is not None
        assert data["analysis"] is not None
        assert data["recommendations"] is not None
        assert data["latency_ms"] > 0

    @patch("src.api.routes.create_workflow")
    def test_analyze_minimal_request(self, mock_create_workflow, client, sample_workflow_state):
        """Test analysis with minimal request."""
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = sample_workflow_state
        mock_create_workflow.return_value = mock_workflow

        response = client.post(
            "/analyze",
            json={
                "positions": [{"symbol": "AAPL", "quantity": 100.0}],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "wf-123"

    def test_analyze_empty_positions(self, client):
        """Test analysis with empty positions."""
        response = client.post(
            "/analyze",
            json={"positions": []},
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_invalid_position(self, client):
        """Test analysis with invalid position."""
        response = client.post(
            "/analyze",
            json={
                "positions": [{"symbol": "", "quantity": 100.0}],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_negative_quantity(self, client):
        """Test analysis with negative quantity."""
        response = client.post(
            "/analyze",
            json={
                "positions": [{"symbol": "AAPL", "quantity": -10.0}],
            },
        )

        assert response.status_code == 422  # Validation error

    @patch("src.api.routes.create_workflow")
    def test_analyze_workflow_error(self, mock_create_workflow, client, sample_portfolio_request):
        """Test analysis when workflow fails."""
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.side_effect = Exception("Workflow failed")
        mock_create_workflow.return_value = mock_workflow

        response = client.post("/analyze", json=sample_portfolio_request)

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Workflow failed" in data["error"]

    @patch("src.api.routes.create_workflow")
    def test_analyze_symbol_uppercasing(self, mock_create_workflow, client, sample_workflow_state):
        """Test that symbols are uppercased."""
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = sample_workflow_state
        mock_create_workflow.return_value = mock_workflow

        response = client.post(
            "/analyze",
            json={
                "positions": [{"symbol": "aapl", "quantity": 100.0}],
            },
        )

        assert response.status_code == 200
        # Verify the workflow was called with uppercased symbol
        call_args = mock_workflow.ainvoke.call_args[0][0]
        assert "AAPL" in call_args["symbols"]


class TestGetAnalysisEndpoint:
    """Tests for GET /analyze/{trace_id} endpoint."""

    def test_get_analysis_not_found(self, client):
        """Test getting non-existent analysis."""
        response = client.get("/analyze/non-existent-trace-id")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["error"].lower()


class TestListAnalysesEndpoint:
    """Tests for GET /analyze endpoint."""

    def test_list_analyses_empty(self, client):
        """Test listing analyses when none exist."""
        response = client.get("/analyze")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_analyses_with_params(self, client):
        """Test listing analyses with query parameters."""
        response = client.get(
            "/analyze",
            params={
                "user_id": "test-user",
                "status": "completed",
                "limit": 10,
                "offset": 0,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# ============================================================================
# State Conversion Tests
# ============================================================================


class TestStateToResponse:
    """Tests for _state_to_response helper."""

    def test_state_to_response_full(self, sample_workflow_state):
        """Test converting full state to response."""
        response = _state_to_response(sample_workflow_state, 1500.5)

        assert response.workflow_id == "wf-123"
        assert response.trace_id == "trace-456"
        assert response.status == "completed"
        assert response.latency_ms == 1500.5

        # Check research
        assert response.research is not None
        assert response.research.summary == "Market conditions favorable"
        assert len(response.research.symbols_researched) == 2

        # Check analysis
        assert response.analysis is not None
        assert response.analysis.summary == "Portfolio has moderate risk"
        assert len(response.analysis.recommendations) == 1

        # Check recommendations
        assert response.recommendations is not None
        assert response.recommendations.summary == "Suggest selling some AAPL"
        assert response.recommendations.total_trades == 1

    def test_state_to_response_minimal(self):
        """Test converting minimal state to response."""
        state = {
            "workflow_id": "wf-min",
            "trace_id": "trace-min",
            "status": "pending",
            "started_at": "2024-01-15T10:00:00Z",
        }
        response = _state_to_response(state, 100.0)

        assert response.workflow_id == "wf-min"
        assert response.trace_id == "trace-min"
        assert response.status == "pending"
        assert response.research is None
        assert response.analysis is None
        assert response.recommendations is None
        assert response.latency_ms == 100.0

    def test_state_to_response_with_errors(self):
        """Test converting state with errors."""
        state = {
            "workflow_id": "wf-err",
            "trace_id": "trace-err",
            "status": "failed",
            "errors": ["Error 1", "Error 2"],
            "started_at": "2024-01-15T10:00:00Z",
        }
        response = _state_to_response(state, 500.0)

        assert response.status == "failed"
        assert len(response.errors) == 2
        assert "Error 1" in response.errors

    def test_state_to_response_partial_research(self):
        """Test converting state with partial research data."""
        state = {
            "workflow_id": "wf-part",
            "trace_id": "trace-part",
            "status": "in_progress",
            "research": {
                "summary": "Partial research",
                # Missing other fields
            },
            "started_at": "2024-01-15T10:00:00Z",
        }
        response = _state_to_response(state, 200.0)

        assert response.research is not None
        assert response.research.summary == "Partial research"
        assert response.research.market_data == {}
        assert response.research.news == []


# ============================================================================
# App Configuration Tests
# ============================================================================


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_create_app_defaults(self):
        """Test creating app with defaults."""
        reset_health_service()
        app = create_app()

        assert app.title == "Portfolio Advisor API"
        assert app.version == "1.0.0"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_create_app_custom(self):
        """Test creating app with custom settings."""
        reset_health_service()
        app = create_app(
            title="Custom API",
            version="2.0.0",
            description="Custom description",
            cors_origins=["http://localhost:3000"],
        )

        assert app.title == "Custom API"
        assert app.version == "2.0.0"

    def test_openapi_endpoint(self, client):
        """Test OpenAPI spec endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/analyze" in data["paths"]
        assert "/health" in data["paths"]

    def test_docs_endpoint(self, client):
        """Test Swagger docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200


# ============================================================================
# CORS Tests
# ============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


# ============================================================================
# Error Handler Tests
# ============================================================================


class TestErrorHandlers:
    """Tests for error handlers."""

    def test_http_exception_handler(self, client):
        """Test HTTP exception handling."""
        response = client.get("/analyze/not-found")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_validation_error(self, client):
        """Test validation error handling."""
        response = client.post(
            "/analyze",
            json={"invalid": "data"},
        )

        assert response.status_code == 422
