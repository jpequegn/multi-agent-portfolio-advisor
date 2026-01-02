"""Tests for orchestration state management."""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestration.state import (
    AgentName,
    Portfolio,
    Position,
    StatePersistence,
    StateValidationError,
    WorkflowStatus,
    create_initial_state,
    get_state_summary,
    mark_state_completed,
    mark_state_failed,
    state_from_json,
    state_to_json,
    update_state_for_agent,
    update_state_with_result,
    validate_state,
    validate_state_or_raise,
)

# ============================================================================
# Enum Tests
# ============================================================================


class TestWorkflowStatus:
    """Tests for WorkflowStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"

    def test_status_is_string(self) -> None:
        """Test that status values are strings."""
        for status in WorkflowStatus:
            assert isinstance(status.value, str)
            assert status.value == str(status.value)


class TestAgentName:
    """Tests for AgentName enum."""

    def test_agent_names(self) -> None:
        """Test all agent names exist."""
        assert AgentName.RESEARCH.value == "research"
        assert AgentName.ANALYSIS.value == "analysis"
        assert AgentName.RECOMMENDATION.value == "recommendation"

    def test_agent_count(self) -> None:
        """Test there are exactly 3 agents."""
        assert len(AgentName) == 3


# ============================================================================
# Position Tests
# ============================================================================


class TestPosition:
    """Tests for Position model."""

    def test_minimal_position(self) -> None:
        """Test position with minimal fields."""
        position = Position(symbol="AAPL", quantity=100)
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.cost_basis is None
        assert position.current_price is None

    def test_full_position(self) -> None:
        """Test position with all fields."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            cost_basis=150.0,
            current_price=175.0,
            market_value=17500.0,
            weight=0.25,
            sector="Technology",
        )
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.cost_basis == 150.0
        assert position.current_price == 175.0
        assert position.market_value == 17500.0
        assert position.weight == 0.25
        assert position.sector == "Technology"

    def test_position_from_dict(self) -> None:
        """Test creating position from dict."""
        position = Position(symbol="GOOGL", quantity=50, cost_basis=2800.0)
        assert position.symbol == "GOOGL"
        assert position.quantity == 50
        assert position.cost_basis == 2800.0

    def test_position_serialization(self) -> None:
        """Test position can be serialized to dict."""
        position = Position(symbol="MSFT", quantity=75, cost_basis=400.0)
        data = position.model_dump()
        assert data["symbol"] == "MSFT"
        assert data["quantity"] == 75
        assert data["cost_basis"] == 400.0


# ============================================================================
# Portfolio Tests
# ============================================================================


class TestPortfolio:
    """Tests for Portfolio model."""

    def test_empty_portfolio(self) -> None:
        """Test empty portfolio."""
        portfolio = Portfolio()
        assert portfolio.positions == []
        assert portfolio.total_value == 0.0
        assert portfolio.cash == 0.0
        assert portfolio.account_type == "taxable"

    def test_portfolio_with_positions(self) -> None:
        """Test portfolio with positions."""
        positions = [
            Position(symbol="AAPL", quantity=100, market_value=17500.0),
            Position(symbol="GOOGL", quantity=50, market_value=14000.0),
        ]
        portfolio = Portfolio(
            positions=positions,
            total_value=31500.0,
            cash=5000.0,
            account_type="ira",
        )
        assert len(portfolio.positions) == 2
        assert portfolio.total_value == 31500.0
        assert portfolio.cash == 5000.0
        assert portfolio.account_type == "ira"

    def test_portfolio_symbols(self) -> None:
        """Test symbols property."""
        portfolio = Portfolio(
            positions=[
                Position(symbol="AAPL", quantity=100),
                Position(symbol="GOOGL", quantity=50),
                Position(symbol="MSFT", quantity=75),
            ]
        )
        assert portfolio.symbols == ["AAPL", "GOOGL", "MSFT"]

    def test_portfolio_weights(self) -> None:
        """Test get_weights method."""
        portfolio = Portfolio(
            positions=[
                Position(symbol="AAPL", quantity=100, market_value=5000.0),
                Position(symbol="GOOGL", quantity=50, market_value=5000.0),
            ],
            total_value=10000.0,
        )
        weights = portfolio.get_weights()
        assert weights["AAPL"] == 0.5
        assert weights["GOOGL"] == 0.5

    def test_portfolio_weights_zero_total(self) -> None:
        """Test get_weights with zero total value."""
        portfolio = Portfolio(
            positions=[Position(symbol="AAPL", quantity=100)],
            total_value=0.0,
        )
        weights = portfolio.get_weights()
        assert weights == {}

    def test_portfolio_from_dict_positions(self) -> None:
        """Test portfolio parses positions from dicts."""
        # Testing that Portfolio can parse dicts into Position objects
        positions_data: list[dict[str, int | str]] = [
            {"symbol": "AAPL", "quantity": 100},
            {"symbol": "GOOGL", "quantity": 50},
        ]
        portfolio = Portfolio(positions=positions_data)  # type: ignore[arg-type]
        assert len(portfolio.positions) == 2
        assert portfolio.positions[0].symbol == "AAPL"
        assert portfolio.positions[1].symbol == "GOOGL"

    def test_portfolio_created_at(self) -> None:
        """Test portfolio has created_at timestamp."""
        portfolio = Portfolio()
        assert portfolio.created_at is not None
        assert isinstance(portfolio.created_at, datetime)

    def test_portfolio_json_serialization(self) -> None:
        """Test portfolio can be serialized to JSON."""
        portfolio = Portfolio(
            positions=[Position(symbol="AAPL", quantity=100)],
            total_value=17500.0,
        )
        data = portfolio.model_dump(mode="json")
        # Should be JSON-serializable (datetime converted to string)
        json_str = json.dumps(data)
        assert "AAPL" in json_str


# ============================================================================
# State Creation Tests
# ============================================================================


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_with_portfolio_model(self) -> None:
        """Test creating state with Portfolio model."""
        portfolio = Portfolio(
            positions=[Position(symbol="AAPL", quantity=100)],
            total_value=17500.0,
        )
        state = create_initial_state(
            portfolio=portfolio,
            user_request="Analyze my portfolio",
        )

        assert state["workflow_id"] is not None
        assert state["trace_id"] is not None
        assert state["user_id"] is None
        assert state["portfolio"]["positions"][0]["symbol"] == "AAPL"
        assert state["user_request"] == "Analyze my portfolio"
        assert state["symbols"] == ["AAPL"]
        assert state["status"] == "pending"
        assert state["research"] is None
        assert state["analysis"] is None
        assert state["recommendation"] is None

    def test_create_with_portfolio_dict(self) -> None:
        """Test creating state with portfolio dict."""
        portfolio_dict = {
            "positions": [{"symbol": "GOOGL", "quantity": 50}],
            "total_value": 14000.0,
        }
        state = create_initial_state(
            portfolio=portfolio_dict,
            user_request="Review positions",
        )

        assert state["symbols"] == ["GOOGL"]
        assert state["portfolio"] == portfolio_dict

    def test_create_with_user_id(self) -> None:
        """Test creating state with user ID."""
        state = create_initial_state(
            portfolio=Portfolio(),
            user_request="Test",
            user_id="user-123",
        )
        assert state["user_id"] == "user-123"

    def test_create_with_trace_id(self) -> None:
        """Test creating state with custom trace ID."""
        state = create_initial_state(
            portfolio=Portfolio(),
            user_request="Test",
            trace_id="trace-abc-123",
        )
        assert state["trace_id"] == "trace-abc-123"

    def test_state_has_timestamps(self) -> None:
        """Test state has started_at timestamp."""
        state = create_initial_state(
            portfolio=Portfolio(),
            user_request="Test",
        )
        assert state["started_at"] is not None
        assert state["completed_at"] is None

    def test_state_has_empty_collections(self) -> None:
        """Test state initializes with empty collections."""
        state = create_initial_state(
            portfolio=Portfolio(),
            user_request="Test",
        )
        assert state["messages"] == []
        assert state["errors"] == []
        assert state["context"] == {}


# ============================================================================
# State Update Tests
# ============================================================================


class TestUpdateStateForAgent:
    """Tests for update_state_for_agent function."""

    def test_update_for_research(self) -> None:
        """Test updating state for research agent."""
        state = create_initial_state(Portfolio(), "Test")
        updated = update_state_for_agent(state, AgentName.RESEARCH)

        assert updated["current_agent"] == "research"
        assert updated["status"] == "running"

    def test_update_for_analysis(self) -> None:
        """Test updating state for analysis agent."""
        state = create_initial_state(Portfolio(), "Test")
        updated = update_state_for_agent(state, AgentName.ANALYSIS)

        assert updated["current_agent"] == "analysis"
        assert updated["status"] == "running"

    def test_update_for_recommendation(self) -> None:
        """Test updating state for recommendation agent."""
        state = create_initial_state(Portfolio(), "Test")
        updated = update_state_for_agent(state, AgentName.RECOMMENDATION)

        assert updated["current_agent"] == "recommendation"
        assert updated["status"] == "running"


class TestUpdateStateWithResult:
    """Tests for update_state_with_result function."""

    def test_update_with_research_result(self) -> None:
        """Test updating state with research result."""
        state = create_initial_state(Portfolio(), "Test")
        result = {
            "market_data": {"AAPL": {"price": 175.0}},
            "news": [],
            "summary": "Research complete",
            "symbols_researched": ["AAPL"],
        }
        updated = update_state_with_result(state, AgentName.RESEARCH, result)

        assert updated["research"] is not None
        assert updated["research"]["summary"] == "Research complete"

    def test_update_with_analysis_result(self) -> None:
        """Test updating state with analysis result."""
        state = create_initial_state(Portfolio(), "Test")
        result = {
            "risk_metrics": {"volatility": 0.2},
            "summary": "Analysis complete",
        }
        updated = update_state_with_result(state, AgentName.ANALYSIS, result)

        assert updated["analysis"] is not None
        assert updated["analysis"]["summary"] == "Analysis complete"

    def test_update_with_recommendation_result(self) -> None:
        """Test updating state with recommendation result."""
        state = create_initial_state(Portfolio(), "Test")
        result = {
            "trades": [],
            "summary": "Recommendations complete",
            "total_trades": 0,
        }
        updated = update_state_with_result(state, AgentName.RECOMMENDATION, result)

        assert updated["recommendation"] is not None
        assert updated["recommendation"]["summary"] == "Recommendations complete"

    def test_update_captures_errors(self) -> None:
        """Test that errors from result are captured."""
        state = create_initial_state(Portfolio(), "Test")
        result = {
            "summary": "Partial research",
            "errors": ["Failed to fetch AAPL data", "API rate limited"],
        }
        updated = update_state_with_result(state, AgentName.RESEARCH, result)

        assert "Failed to fetch AAPL data" in updated["errors"]
        assert "API rate limited" in updated["errors"]


class TestMarkStateCompleted:
    """Tests for mark_state_completed function."""

    def test_mark_completed(self) -> None:
        """Test marking state as completed."""
        state = create_initial_state(Portfolio(), "Test")
        state = update_state_for_agent(state, AgentName.RESEARCH)
        completed = mark_state_completed(state)

        assert completed["status"] == "completed"
        assert completed["completed_at"] is not None
        assert completed["current_agent"] is None


class TestMarkStateFailed:
    """Tests for mark_state_failed function."""

    def test_mark_failed(self) -> None:
        """Test marking state as failed."""
        state = create_initial_state(Portfolio(), "Test")
        failed = mark_state_failed(state, "Connection timeout")

        assert failed["status"] == "failed"
        assert failed["completed_at"] is not None
        assert failed["error"] == "Connection timeout"
        assert "Connection timeout" in failed["errors"]


# ============================================================================
# State Validation Tests
# ============================================================================


class TestValidateState:
    """Tests for validate_state function."""

    def test_valid_state(self) -> None:
        """Test validation passes for valid state."""
        state = create_initial_state(Portfolio(), "Test request")
        errors = validate_state(state)
        assert errors == []

    def test_missing_workflow_id(self) -> None:
        """Test validation catches missing workflow_id."""
        state = create_initial_state(Portfolio(), "Test")
        state["workflow_id"] = ""
        errors = validate_state(state)
        assert any("workflow_id" in e for e in errors)

    def test_missing_trace_id(self) -> None:
        """Test validation catches missing trace_id."""
        state = create_initial_state(Portfolio(), "Test")
        state["trace_id"] = ""
        errors = validate_state(state)
        assert any("trace_id" in e for e in errors)

    def test_missing_portfolio(self) -> None:
        """Test validation catches missing portfolio."""
        state = create_initial_state(Portfolio(), "Test")
        state["portfolio"] = {}
        errors = validate_state(state)
        assert any("portfolio" in e for e in errors)

    def test_missing_user_request(self) -> None:
        """Test validation catches missing user_request."""
        state = create_initial_state(Portfolio(), "Test")
        state["user_request"] = ""
        errors = validate_state(state)
        assert any("user_request" in e for e in errors)

    def test_invalid_status(self) -> None:
        """Test validation catches invalid status."""
        state = create_initial_state(Portfolio(), "Test")
        state["status"] = "invalid_status"
        errors = validate_state(state)
        assert any("Invalid status" in e for e in errors)

    def test_invalid_current_agent(self) -> None:
        """Test validation catches invalid current_agent."""
        state = create_initial_state(Portfolio(), "Test")
        state["current_agent"] = "invalid_agent"
        errors = validate_state(state)
        assert any("Invalid current_agent" in e for e in errors)

    def test_invalid_started_at_format(self) -> None:
        """Test validation catches invalid started_at format."""
        state = create_initial_state(Portfolio(), "Test")
        state["started_at"] = "not-a-date"
        errors = validate_state(state)
        assert any("started_at" in e for e in errors)

    def test_invalid_completed_at_format(self) -> None:
        """Test validation catches invalid completed_at format."""
        state = create_initial_state(Portfolio(), "Test")
        state["completed_at"] = "not-a-date"
        errors = validate_state(state)
        assert any("completed_at" in e for e in errors)

    def test_valid_iso_timestamps(self) -> None:
        """Test validation accepts valid ISO timestamps."""
        state = create_initial_state(Portfolio(), "Test")
        state["started_at"] = "2024-01-15T10:30:00+00:00"
        state["completed_at"] = "2024-01-15T10:35:00Z"
        errors = validate_state(state)
        assert errors == []


class TestValidateStateOrRaise:
    """Tests for validate_state_or_raise function."""

    def test_valid_state_no_raise(self) -> None:
        """Test no exception for valid state."""
        state = create_initial_state(Portfolio(), "Test")
        # Should not raise
        validate_state_or_raise(state)

    def test_invalid_state_raises(self) -> None:
        """Test exception raised for invalid state."""
        state = create_initial_state(Portfolio(), "Test")
        state["workflow_id"] = ""
        state["trace_id"] = ""

        with pytest.raises(StateValidationError) as exc_info:
            validate_state_or_raise(state)

        assert "workflow_id" in str(exc_info.value)
        assert "trace_id" in str(exc_info.value)


# ============================================================================
# State Serialization Tests
# ============================================================================


class TestStateSerialization:
    """Tests for state serialization functions."""

    def test_state_to_json(self) -> None:
        """Test state can be serialized to JSON."""
        state = create_initial_state(
            Portfolio(positions=[Position(symbol="AAPL", quantity=100)]),
            "Test",
        )
        json_str = state_to_json(state)
        assert isinstance(json_str, str)
        assert "AAPL" in json_str
        assert "pending" in json_str

    def test_state_from_json(self) -> None:
        """Test state can be deserialized from JSON."""
        state = create_initial_state(Portfolio(), "Test")
        json_str = state_to_json(state)
        restored = state_from_json(json_str)

        assert restored["workflow_id"] == state["workflow_id"]
        assert restored["user_request"] == state["user_request"]
        assert restored["status"] == state["status"]

    def test_roundtrip_serialization(self) -> None:
        """Test state survives JSON roundtrip."""
        portfolio = Portfolio(
            positions=[
                Position(symbol="AAPL", quantity=100, cost_basis=150.0),
                Position(symbol="GOOGL", quantity=50, cost_basis=2800.0),
            ],
            total_value=31500.0,
            cash=5000.0,
        )
        state = create_initial_state(portfolio, "Analyze portfolio", user_id="user-1")
        state = update_state_for_agent(state, AgentName.RESEARCH)
        state = update_state_with_result(
            state,
            AgentName.RESEARCH,
            {"summary": "Done", "market_data": {}, "news": []},
        )

        json_str = state_to_json(state)
        restored = state_from_json(json_str)

        assert restored["workflow_id"] == state["workflow_id"]
        assert restored["user_id"] == "user-1"
        assert restored["research"] is not None
        assert restored["research"]["summary"] == "Done"


class TestGetStateSummary:
    """Tests for get_state_summary function."""

    def test_summary_fields(self) -> None:
        """Test summary contains expected fields."""
        state = create_initial_state(
            Portfolio(positions=[Position(symbol="AAPL", quantity=100)]),
            "Test",
        )
        summary = get_state_summary(state)

        assert "workflow_id" in summary
        assert "status" in summary
        assert "current_agent" in summary
        assert "symbol_count" in summary
        assert "has_research" in summary
        assert "has_analysis" in summary
        assert "has_recommendation" in summary
        assert "error_count" in summary
        assert "started_at" in summary
        assert "completed_at" in summary

    def test_summary_values(self) -> None:
        """Test summary has correct values."""
        portfolio = Portfolio(
            positions=[
                Position(symbol="AAPL", quantity=100),
                Position(symbol="GOOGL", quantity=50),
            ]
        )
        state = create_initial_state(portfolio, "Test")
        state = update_state_with_result(
            state,
            AgentName.RESEARCH,
            {"summary": "Done"},
        )
        state["errors"] = ["Error 1", "Error 2"]

        summary = get_state_summary(state)

        assert summary["status"] == "pending"
        assert summary["symbol_count"] == 2
        assert summary["has_research"] is True
        assert summary["has_analysis"] is False
        assert summary["has_recommendation"] is False
        assert summary["error_count"] == 2


# ============================================================================
# State Persistence Tests
# ============================================================================


class TestStatePersistence:
    """Tests for StatePersistence class."""

    def test_init_with_connection_string(self) -> None:
        """Test initialization with explicit connection string."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        assert persistence.connection_string == "postgresql://localhost:5432/test"

    def test_init_with_env_var(self) -> None:
        """Test initialization with environment variable."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://db:5432/mydb"}):
            persistence = StatePersistence()
            assert persistence.connection_string == "postgresql://db:5432/mydb"

    def test_init_default(self) -> None:
        """Test initialization with default connection string."""
        with patch.dict("os.environ", {}, clear=True):
            persistence = StatePersistence()
            assert persistence.connection_string is not None
            assert "portfolio_advisor" in persistence.connection_string

    @pytest.mark.asyncio
    async def test_initialize_schema(self) -> None:
        """Test schema initialization."""
        persistence = StatePersistence("postgresql://localhost:5432/test")

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            await persistence.initialize_schema()

        mock_conn.execute.assert_called_once()
        sql = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS workflow_states" in sql

    @pytest.mark.asyncio
    async def test_save_state(self) -> None:
        """Test saving state."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        state = create_initial_state(Portfolio(), "Test request")

        mock_conn = AsyncMock()
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            await persistence.save_state(state)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_state_validates(self) -> None:
        """Test save_state validates state first."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        state = create_initial_state(Portfolio(), "Test")
        state["workflow_id"] = ""  # Invalid

        with pytest.raises(StateValidationError):
            await persistence.save_state(state)

    @pytest.mark.asyncio
    async def test_load_state_found(self) -> None:
        """Test loading existing state."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        workflow_id = "12345678-1234-1234-1234-123456789012"

        mock_row = {
            "workflow_id": workflow_id,
            "trace_id": "trace-123",
            "user_id": "user-1",
            "portfolio": '{"positions": []}',
            "user_request": "Test",
            "symbols": ["AAPL"],
            "research": None,
            "analysis": None,
            "recommendation": None,
            "status": "pending",
            "current_agent": None,
            "started_at": datetime.now(UTC),
            "completed_at": None,
            "error": None,
            "messages": "[]",
            "errors": [],
            "context": "{}",
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = mock_row
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            state = await persistence.load_state(workflow_id)

        assert state is not None
        assert state["workflow_id"] == workflow_id
        assert state["user_request"] == "Test"

    @pytest.mark.asyncio
    async def test_load_state_not_found(self) -> None:
        """Test loading non-existent state."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        workflow_id = "00000000-0000-0000-0000-000000000000"  # Valid UUID format

        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            state = await persistence.load_state(workflow_id)

        assert state is None

    @pytest.mark.asyncio
    async def test_list_states(self) -> None:
        """Test listing states."""
        persistence = StatePersistence("postgresql://localhost:5432/test")

        mock_rows: list[dict[str, Any]] = [
            {
                "workflow_id": "id-1",
                "trace_id": "trace-1",
                "user_id": "user-1",
                "portfolio": '{"positions": []}',
                "user_request": "Test 1",
                "symbols": [],
                "research": None,
                "analysis": None,
                "recommendation": None,
                "status": "completed",
                "current_agent": None,
                "started_at": datetime.now(UTC),
                "completed_at": datetime.now(UTC),
                "error": None,
                "messages": "[]",
                "errors": [],
                "context": "{}",
            }
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = mock_rows
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            states = await persistence.list_states(user_id="user-1", limit=10)

        assert len(states) == 1
        assert states[0]["user_request"] == "Test 1"

    @pytest.mark.asyncio
    async def test_list_states_with_status_filter(self) -> None:
        """Test listing states with status filter."""
        persistence = StatePersistence("postgresql://localhost:5432/test")

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            await persistence.list_states(status=WorkflowStatus.COMPLETED)

        # Verify status was included in query
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        assert "status = $" in sql

    @pytest.mark.asyncio
    async def test_delete_state_success(self) -> None:
        """Test deleting existing state."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        workflow_id = "11111111-1111-1111-1111-111111111111"

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 1"
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            result = await persistence.delete_state(workflow_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_state_not_found(self) -> None:
        """Test deleting non-existent state."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        workflow_id = "00000000-0000-0000-0000-000000000000"

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "DELETE 0"
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        with patch.object(
            persistence, "_get_pool", new=AsyncMock(return_value=mock_pool)
        ):
            result = await persistence.delete_state(workflow_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing connection pool."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        mock_pool = AsyncMock()
        persistence._pool = mock_pool

        await persistence.close()

        mock_pool.close.assert_called_once()
        assert persistence._pool is None

    @pytest.mark.asyncio
    async def test_get_pool_caches_pool(self) -> None:
        """Test connection pool is cached after creation."""
        persistence = StatePersistence("postgresql://localhost:5432/test")

        # Manually set a mock pool
        mock_pool = MagicMock()
        persistence._pool = mock_pool

        # Should return the cached pool
        pool = await persistence._get_pool()
        assert pool is mock_pool

    def test_pool_initially_none(self) -> None:
        """Test connection pool starts as None."""
        persistence = StatePersistence("postgresql://localhost:5432/test")
        assert persistence._pool is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestWorkflowStateFlow:
    """Integration tests for workflow state transitions."""

    def test_full_workflow_happy_path(self) -> None:
        """Test complete workflow from start to finish."""
        # Create initial state
        portfolio = Portfolio(
            positions=[
                Position(symbol="AAPL", quantity=100, market_value=17500.0),
                Position(symbol="GOOGL", quantity=50, market_value=14000.0),
            ],
            total_value=31500.0,
            cash=5000.0,
        )
        state = create_initial_state(
            portfolio=portfolio,
            user_request="Analyze and recommend trades",
            user_id="user-123",
        )
        assert state["status"] == "pending"

        # Research phase
        state = update_state_for_agent(state, AgentName.RESEARCH)
        assert state["status"] == "running"
        assert state["current_agent"] == "research"

        state = update_state_with_result(
            state,
            AgentName.RESEARCH,
            {
                "market_data": {"AAPL": {"price": 175.0}, "GOOGL": {"price": 2800.0}},
                "news": [{"headline": "Tech stocks rally"}],
                "summary": "Research complete",
            },
        )
        assert state["research"] is not None

        # Analysis phase
        state = update_state_for_agent(state, AgentName.ANALYSIS)
        assert state["current_agent"] == "analysis"

        state = update_state_with_result(
            state,
            AgentName.ANALYSIS,
            {
                "risk_metrics": {"volatility": 0.18, "sharpe_ratio": 1.2},
                "summary": "Analysis complete",
            },
        )
        assert state["analysis"] is not None

        # Recommendation phase
        state = update_state_for_agent(state, AgentName.RECOMMENDATION)
        assert state["current_agent"] == "recommendation"

        state = update_state_with_result(
            state,
            AgentName.RECOMMENDATION,
            {
                "trades": [{"symbol": "MSFT", "action": "buy", "quantity": 20}],
                "summary": "Recommendations ready",
                "total_trades": 1,
            },
        )
        assert state["recommendation"] is not None

        # Mark completed
        state = mark_state_completed(state)
        assert state["status"] == "completed"
        assert state["completed_at"] is not None

        # Validate final state
        errors = validate_state(state)
        assert errors == []

    def test_workflow_failure_handling(self) -> None:
        """Test workflow handles failure correctly."""
        state = create_initial_state(Portfolio(), "Test")
        state = update_state_for_agent(state, AgentName.RESEARCH)

        # Simulate failure
        state = mark_state_failed(state, "API connection failed")

        assert state["status"] == "failed"
        assert state["error"] == "API connection failed"
        assert "API connection failed" in state["errors"]
        assert state["completed_at"] is not None

    def test_state_summary_progression(self) -> None:
        """Test state summary updates through workflow."""
        state = create_initial_state(
            Portfolio(positions=[Position(symbol="AAPL", quantity=100)]),
            "Test",
        )

        # Initial summary
        summary = get_state_summary(state)
        assert summary["status"] == "pending"
        assert summary["has_research"] is False
        assert summary["has_analysis"] is False
        assert summary["has_recommendation"] is False

        # After research
        state = update_state_with_result(state, AgentName.RESEARCH, {"summary": "Done"})
        summary = get_state_summary(state)
        assert summary["has_research"] is True
        assert summary["has_analysis"] is False

        # After analysis
        state = update_state_with_result(state, AgentName.ANALYSIS, {"summary": "Done"})
        summary = get_state_summary(state)
        assert summary["has_analysis"] is True

        # After recommendation
        state = update_state_with_result(state, AgentName.RECOMMENDATION, {"summary": "Done"})
        summary = get_state_summary(state)
        assert summary["has_recommendation"] is True
