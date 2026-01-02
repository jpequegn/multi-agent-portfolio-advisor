"""Tests for LangGraph workflow orchestration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestration.state import (
    AgentName,
    Portfolio,
    PortfolioState,
    Position,
    WorkflowStatus,
    create_initial_state,
)
from src.orchestration.workflow import (
    analysis_node,
    create_workflow,
    error_handler_node,
    finalize_node,
    recommendation_node,
    research_node,
    route_after_analysis,
    route_after_recommendation,
    route_after_research,
    run_workflow,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create a sample portfolio for testing."""
    return Portfolio(
        positions=[
            Position(symbol="AAPL", quantity=100, market_value=17500.0),
            Position(symbol="GOOGL", quantity=50, market_value=14000.0),
            Position(symbol="MSFT", quantity=75, market_value=28500.0),
        ],
        total_value=60000.0,
        cash=5000.0,
    )


@pytest.fixture
def initial_state(sample_portfolio: Portfolio) -> PortfolioState:
    """Create an initial workflow state for testing."""
    return create_initial_state(
        portfolio=sample_portfolio,
        user_request="Analyze my portfolio and suggest trades",
        user_id="test-user",
    )


@pytest.fixture
def mock_agent_state() -> MagicMock:
    """Create a mock agent state with default context."""
    mock = MagicMock()
    mock.context = {
        "market_data": {"AAPL": {"price": 175.0}},
        "news_items": [],
        "summary": "Test summary",
    }
    mock.errors = []
    return mock


# ============================================================================
# Routing Function Tests
# ============================================================================


class TestRouteAfterResearch:
    """Tests for route_after_research function."""

    def test_routes_to_analysis_on_success(self, initial_state: PortfolioState) -> None:
        """Test routing to analysis when research succeeds."""
        initial_state["research"] = {"summary": "Research done", "market_data": {}}
        result = route_after_research(initial_state)
        assert result == "analysis"

    def test_routes_to_error_on_no_research(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to error handler when no research output."""
        initial_state["research"] = None
        result = route_after_research(initial_state)
        assert result == "error_handler"

    def test_routes_to_error_on_critical_error(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to error handler on critical error."""
        initial_state["research"] = {"summary": "Partial"}
        initial_state["errors"] = ["CRITICAL: API failure"]
        result = route_after_research(initial_state)
        assert result == "error_handler"

    def test_continues_with_non_critical_errors(
        self, initial_state: PortfolioState
    ) -> None:
        """Test continuing to analysis with non-critical errors."""
        initial_state["research"] = {"summary": "Done"}
        initial_state["errors"] = ["Warning: slow response"]
        result = route_after_research(initial_state)
        assert result == "analysis"


class TestRouteAfterAnalysis:
    """Tests for route_after_analysis function."""

    def test_routes_to_recommendation_on_success(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to recommendation when analysis succeeds."""
        initial_state["analysis"] = {"summary": "Analysis done", "risk_metrics": {}}
        result = route_after_analysis(initial_state)
        assert result == "recommendation"

    def test_routes_to_error_on_no_analysis(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to error handler when no analysis output."""
        initial_state["analysis"] = None
        result = route_after_analysis(initial_state)
        assert result == "error_handler"

    def test_routes_to_error_on_critical_error(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to error handler on critical error."""
        initial_state["analysis"] = {"summary": "Partial"}
        initial_state["errors"] = ["Critical calculation error"]
        result = route_after_analysis(initial_state)
        assert result == "error_handler"


class TestRouteAfterRecommendation:
    """Tests for route_after_recommendation function."""

    def test_routes_to_finalize_on_success(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to finalize when recommendation succeeds."""
        initial_state["recommendation"] = {"trades": [], "summary": "Done"}
        result = route_after_recommendation(initial_state)
        assert result == "finalize"

    def test_routes_to_finalize_even_without_trades(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to finalize even without recommendations."""
        initial_state["recommendation"] = None
        result = route_after_recommendation(initial_state)
        assert result == "finalize"

    def test_routes_to_error_on_critical_error(
        self, initial_state: PortfolioState
    ) -> None:
        """Test routing to error handler on critical error."""
        initial_state["errors"] = ["CRITICAL: Compliance failure"]
        result = route_after_recommendation(initial_state)
        assert result == "error_handler"


# ============================================================================
# Node Function Tests
# ============================================================================


class TestResearchNode:
    """Tests for research_node function."""

    @pytest.mark.asyncio
    async def test_updates_state_for_agent(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that research node updates state correctly."""
        with patch("src.orchestration.workflow.ResearchAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            # Mock agent response
            mock_result = MagicMock()
            mock_result.context = {
                "market_data": {"AAPL": {"price": 175.0}},
                "news_items": [{"title": "Test"}],
                "summary": "Research complete",
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await research_node(initial_state)

            assert result["current_agent"] == AgentName.RESEARCH.value
            assert result["status"] == WorkflowStatus.RUNNING.value

    @pytest.mark.asyncio
    async def test_captures_research_output(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that research output is captured in state."""
        with patch("src.orchestration.workflow.ResearchAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.context = {
                "research_output": {
                    "market_data": {"AAPL": {"price": 175.0}},
                    "summary": "Done",
                }
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await research_node(initial_state)

            assert result["research"] is not None
            assert result["research"]["summary"] == "Done"

    @pytest.mark.asyncio
    async def test_handles_agent_exception(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that exceptions are captured as errors."""
        with patch("src.orchestration.workflow.ResearchAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.side_effect = Exception("API timeout")

            result = await research_node(initial_state)

            assert any("Research failed" in e for e in result["errors"])


class TestAnalysisNode:
    """Tests for analysis_node function."""

    @pytest.mark.asyncio
    async def test_updates_state_for_agent(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that analysis node updates state correctly."""
        initial_state["research"] = {"summary": "Research done"}

        with patch("src.orchestration.workflow.AnalysisAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.context = {
                "risk_metrics": {"volatility": 0.2},
                "summary": "Analysis complete",
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await analysis_node(initial_state)

            assert result["current_agent"] == AgentName.ANALYSIS.value

    @pytest.mark.asyncio
    async def test_captures_analysis_output(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that analysis output is captured in state."""
        initial_state["research"] = {"summary": "Research done"}

        with patch("src.orchestration.workflow.AnalysisAgent") as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.context = {
                "analysis_output": {
                    "risk_metrics": {"volatility": 0.2},
                    "summary": "High risk",
                }
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await analysis_node(initial_state)

            assert result["analysis"] is not None
            assert result["analysis"]["summary"] == "High risk"


class TestRecommendationNode:
    """Tests for recommendation_node function."""

    @pytest.mark.asyncio
    async def test_updates_state_for_agent(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that recommendation node updates state correctly."""
        initial_state["research"] = {"summary": "Done"}
        initial_state["analysis"] = {"summary": "Done"}

        with patch(
            "src.orchestration.workflow.RecommendationAgent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.context = {
                "trades": [{"symbol": "AAPL", "action": "buy"}],
                "summary": "Recommendations ready",
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await recommendation_node(initial_state)

            assert result["current_agent"] == AgentName.RECOMMENDATION.value

    @pytest.mark.asyncio
    async def test_captures_recommendation_output(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that recommendation output is captured in state."""
        initial_state["research"] = {"summary": "Done"}
        initial_state["analysis"] = {"summary": "Done"}

        with patch(
            "src.orchestration.workflow.RecommendationAgent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.context = {
                "recommendation_output": {
                    "trades": [{"symbol": "AAPL", "action": "buy", "quantity": 10}],
                    "summary": "Buy AAPL",
                    "total_trades": 1,
                }
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await recommendation_node(initial_state)

            assert result["recommendation"] is not None
            assert result["recommendation"]["total_trades"] == 1

    @pytest.mark.asyncio
    async def test_counts_trade_actions(self, initial_state: PortfolioState) -> None:
        """Test that trade action counts are calculated."""
        initial_state["research"] = {"summary": "Done"}
        initial_state["analysis"] = {"summary": "Done"}

        with patch(
            "src.orchestration.workflow.RecommendationAgent"
        ) as mock_agent_class:
            mock_agent = AsyncMock()
            mock_agent_class.return_value = mock_agent

            mock_result = MagicMock()
            mock_result.context = {
                "trades": [
                    {"symbol": "AAPL", "action": "buy"},
                    {"symbol": "GOOGL", "action": "sell"},
                    {"symbol": "MSFT", "action": "hold"},
                    {"symbol": "AMZN", "action": "buy"},
                ],
                "summary": "Mixed recommendations",
            }
            mock_result.errors = []
            mock_agent.return_value = mock_result

            result = await recommendation_node(initial_state)

            assert result["recommendation"]["buy_count"] == 2
            assert result["recommendation"]["sell_count"] == 1
            assert result["recommendation"]["hold_count"] == 1


class TestErrorHandlerNode:
    """Tests for error_handler_node function."""

    @pytest.mark.asyncio
    async def test_marks_state_as_failed(self, initial_state: PortfolioState) -> None:
        """Test that error handler marks state as failed."""
        initial_state["errors"] = ["API failure", "Timeout error"]

        result = await error_handler_node(initial_state)

        assert result["status"] == WorkflowStatus.FAILED.value
        assert result["error"] == "Timeout error"  # Last error
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_handles_empty_errors(self, initial_state: PortfolioState) -> None:
        """Test error handler with no errors."""
        initial_state["errors"] = []

        result = await error_handler_node(initial_state)

        assert result["status"] == WorkflowStatus.FAILED.value
        assert result["error"] == "Unknown error"


class TestFinalizeNode:
    """Tests for finalize_node function."""

    @pytest.mark.asyncio
    async def test_marks_state_as_completed(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that finalize marks state as completed."""
        initial_state["research"] = {"summary": "Done"}
        initial_state["analysis"] = {"summary": "Done"}
        initial_state["recommendation"] = {"summary": "Done"}

        result = await finalize_node(initial_state)

        assert result["status"] == WorkflowStatus.COMPLETED.value
        assert result["completed_at"] is not None
        assert result["current_agent"] is None


# ============================================================================
# Workflow Creation Tests
# ============================================================================


class TestCreateWorkflow:
    """Tests for create_workflow function."""

    def test_creates_compiled_workflow(self) -> None:
        """Test that workflow is compiled successfully."""
        workflow = create_workflow()
        assert workflow is not None

    def test_workflow_has_nodes(self) -> None:
        """Test that workflow has expected nodes."""
        workflow = create_workflow()
        # Check that the workflow has the expected structure
        assert hasattr(workflow, "invoke") or hasattr(workflow, "ainvoke")

    def test_creates_new_workflow_each_time(self) -> None:
        """Test that each call creates a new workflow."""
        workflow1 = create_workflow()
        workflow2 = create_workflow()
        assert workflow1 is not workflow2


# ============================================================================
# Workflow Execution Tests
# ============================================================================


class TestRunWorkflow:
    """Tests for run_workflow function."""

    @pytest.mark.asyncio
    async def test_executes_workflow(self, initial_state: PortfolioState) -> None:
        """Test that workflow executes successfully."""
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = {
            **initial_state,
            "status": WorkflowStatus.COMPLETED.value,
            "research": {"summary": "Done"},
            "analysis": {"summary": "Done"},
            "recommendation": {"summary": "Done"},
        }

        result = await run_workflow(initial_state, workflow=mock_workflow)

        mock_workflow.ainvoke.assert_called_once()
        assert result["status"] == WorkflowStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_handles_workflow_exception(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that workflow exceptions are handled."""
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.side_effect = Exception("Workflow crashed")

        result = await run_workflow(initial_state, workflow=mock_workflow)

        assert result["status"] == WorkflowStatus.FAILED.value
        assert "Workflow execution failed" in result["error"]

    @pytest.mark.asyncio
    async def test_creates_workflow_if_not_provided(
        self, initial_state: PortfolioState
    ) -> None:
        """Test that workflow is created if not provided."""
        with patch("src.orchestration.workflow.create_workflow") as mock_create:
            mock_workflow = AsyncMock()
            mock_workflow.ainvoke.return_value = {
                **initial_state,
                "status": WorkflowStatus.COMPLETED.value,
            }
            mock_create.return_value = mock_workflow

            await run_workflow(initial_state)

            mock_create.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================


class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_mocked_agents(
        self, initial_state: PortfolioState
    ) -> None:
        """Test complete workflow with mocked agent responses."""
        with (
            patch("src.orchestration.workflow.ResearchAgent") as mock_research_class,
            patch("src.orchestration.workflow.AnalysisAgent") as mock_analysis_class,
            patch(
                "src.orchestration.workflow.RecommendationAgent"
            ) as mock_rec_class,
        ):
            # Setup research agent mock
            research_agent = AsyncMock()
            mock_research_class.return_value = research_agent
            research_result = MagicMock()
            research_result.context = {
                "market_data": {"AAPL": {"price": 175.0}},
                "summary": "Research complete",
            }
            research_result.errors = []
            research_agent.return_value = research_result

            # Setup analysis agent mock
            analysis_agent = AsyncMock()
            mock_analysis_class.return_value = analysis_agent
            analysis_result = MagicMock()
            analysis_result.context = {
                "risk_metrics": {"volatility": 0.2},
                "summary": "Analysis complete",
            }
            analysis_result.errors = []
            analysis_agent.return_value = analysis_result

            # Setup recommendation agent mock
            rec_agent = AsyncMock()
            mock_rec_class.return_value = rec_agent
            rec_result = MagicMock()
            rec_result.context = {
                "trades": [{"symbol": "AAPL", "action": "buy", "quantity": 10}],
                "summary": "Buy AAPL",
            }
            rec_result.errors = []
            rec_agent.return_value = rec_result

            # Create and run workflow
            workflow = create_workflow()
            result = await run_workflow(initial_state, workflow=workflow)

            # Verify workflow completed
            assert result["status"] == WorkflowStatus.COMPLETED.value
            assert result["research"] is not None
            assert result["analysis"] is not None
            assert result["recommendation"] is not None

    @pytest.mark.asyncio
    async def test_workflow_handles_research_failure(
        self, initial_state: PortfolioState
    ) -> None:
        """Test workflow handles research agent failure."""
        with patch("src.orchestration.workflow.ResearchAgent") as mock_research_class:
            research_agent = AsyncMock()
            mock_research_class.return_value = research_agent
            research_agent.side_effect = Exception("CRITICAL: API unavailable")

            workflow = create_workflow()
            result = await run_workflow(initial_state, workflow=workflow)

            # Should route to error handler due to no research output
            assert result["status"] == WorkflowStatus.FAILED.value
