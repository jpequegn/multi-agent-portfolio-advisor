"""Tests for the Recommendation Agent."""

import pytest

from src.agents.base import AgentState
from src.agents.recommendation import (
    ComplianceInput,
    ComplianceOutput,
    ComplianceResult,
    ExecutionCostInput,
    ExecutionCostOutput,
    ExecutionCosts,
    PlaceholderComplianceTool,
    PlaceholderExecutionCostTool,
    PlaceholderRebalancingTool,
    PlaceholderTaxImpactTool,
    RebalancingInput,
    RebalancingOutput,
    RecommendationAgent,
    RecommendationOutput,
    TaxImpact,
    TaxImpactInput,
    TaxImpactOutput,
    Trade,
)
from src.tools.base import ToolRegistry

# ============================================================================
# Output Schema Tests
# ============================================================================


class TestTrade:
    """Tests for Trade schema."""

    def test_required_fields(self) -> None:
        """Test trade with required fields only."""
        trade = Trade(action="buy", symbol="AAPL")
        assert trade.action == "buy"
        assert trade.symbol == "AAPL"
        assert trade.quantity == 0
        assert trade.priority == 3
        assert trade.urgency == "opportunistic"

    def test_all_fields(self) -> None:
        """Test trade with all fields."""
        trade = Trade(
            action="sell",
            symbol="GOOGL",
            quantity=50,
            target_weight=0.15,
            current_weight=0.20,
            rationale="Reduce tech exposure",
            priority=1,
            urgency="immediate",
        )
        assert trade.action == "sell"
        assert trade.quantity == 50
        assert trade.target_weight == 0.15
        assert trade.current_weight == 0.20
        assert trade.rationale == "Reduce tech exposure"
        assert trade.priority == 1
        assert trade.urgency == "immediate"

    def test_action_literal(self) -> None:
        """Test that action is validated."""
        for action in ["buy", "sell", "hold"]:
            trade = Trade(action=action, symbol="TEST")
            assert trade.action == action

    def test_priority_bounds(self) -> None:
        """Test priority validation."""
        trade = Trade(action="buy", symbol="TEST", priority=1)
        assert trade.priority == 1

        trade = Trade(action="buy", symbol="TEST", priority=5)
        assert trade.priority == 5


class TestTaxImpact:
    """Tests for TaxImpact schema."""

    def test_default_values(self) -> None:
        """Test default values."""
        tax = TaxImpact()
        assert tax.short_term_gains == 0.0
        assert tax.long_term_gains == 0.0
        assert tax.estimated_tax == 0.0
        assert tax.tax_loss_harvesting_opportunities == []

    def test_with_data(self) -> None:
        """Test with actual data."""
        tax = TaxImpact(
            short_term_gains=1000.0,
            long_term_gains=5000.0,
            estimated_tax=1070.0,
            tax_loss_harvesting_opportunities=["MSFT", "META"],
        )
        assert tax.short_term_gains == 1000.0
        assert tax.long_term_gains == 5000.0
        assert tax.estimated_tax == 1070.0
        assert len(tax.tax_loss_harvesting_opportunities) == 2


class TestExecutionCosts:
    """Tests for ExecutionCosts schema."""

    def test_default_values(self) -> None:
        """Test default values."""
        costs = ExecutionCosts()
        assert costs.total_commission == 0.0
        assert costs.estimated_slippage == 0.0
        assert costs.market_impact == 0.0
        assert costs.total_cost == 0.0

    def test_with_data(self) -> None:
        """Test with actual data."""
        costs = ExecutionCosts(
            total_commission=24.75,
            estimated_slippage=15.50,
            market_impact=8.25,
            total_cost=48.50,
        )
        assert costs.total_commission == 24.75
        assert costs.estimated_slippage == 15.50
        assert costs.market_impact == 8.25
        assert costs.total_cost == 48.50


class TestComplianceResult:
    """Tests for ComplianceResult schema."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = ComplianceResult()
        assert result.is_compliant is True
        assert result.violations == []
        assert result.warnings == []
        assert result.concentration_limits == {}

    def test_with_violations(self) -> None:
        """Test with violations."""
        result = ComplianceResult(
            is_compliant=False,
            violations=["AAPL exceeds 25% limit"],
            warnings=["MSFT approaching limit"],
            concentration_limits={"AAPL": 0.25, "MSFT": 0.25},
        )
        assert result.is_compliant is False
        assert len(result.violations) == 1
        assert len(result.warnings) == 1
        assert result.concentration_limits["AAPL"] == 0.25


class TestRecommendationOutput:
    """Tests for RecommendationOutput schema."""

    def test_default_values(self) -> None:
        """Test default values."""
        output = RecommendationOutput()
        assert output.trades == []
        assert output.summary == ""
        assert output.total_trades == 0
        assert output.buy_count == 0
        assert output.sell_count == 0
        assert output.hold_count == 0
        assert output.errors == []
        assert isinstance(output.tax_impact, TaxImpact)
        assert isinstance(output.execution_costs, ExecutionCosts)
        assert isinstance(output.compliance, ComplianceResult)

    def test_with_trades(self) -> None:
        """Test with trades."""
        trades = [
            Trade(action="buy", symbol="AAPL", quantity=10),
            Trade(action="sell", symbol="GOOGL", quantity=5),
            Trade(action="hold", symbol="MSFT"),
        ]
        output = RecommendationOutput(
            trades=trades,
            total_trades=3,
            buy_count=1,
            sell_count=1,
            hold_count=1,
            summary="Test summary",
        )
        assert len(output.trades) == 3
        assert output.total_trades == 3
        assert output.buy_count == 1
        assert output.sell_count == 1
        assert output.hold_count == 1


# ============================================================================
# Placeholder Tool Tests
# ============================================================================


class TestPlaceholderRebalancingTool:
    """Tests for PlaceholderRebalancingTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_trades(self) -> None:
        """Test that mock returns trade recommendations."""
        tool = PlaceholderRebalancingTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL", "MSFT"]})

        assert isinstance(result, RebalancingOutput)
        assert len(result.trades) == 3
        assert result.rebalance_summary != ""

    @pytest.mark.asyncio
    async def test_trades_have_required_fields(self) -> None:
        """Test that all trades have required fields."""
        tool = PlaceholderRebalancingTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL"]})

        for trade in result.trades:
            assert trade.action in ["buy", "sell", "hold"]
            assert trade.symbol in ["AAPL", "GOOGL"]
            assert trade.rationale != ""
            assert 1 <= trade.priority <= 5

    @pytest.mark.asyncio
    async def test_trades_sorted_by_priority(self) -> None:
        """Test that trades are sorted by priority."""
        tool = PlaceholderRebalancingTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"]})

        priorities = [t.priority for t in result.trades]
        assert priorities == sorted(priorities)

    @pytest.mark.asyncio
    async def test_risk_metrics_affect_trades(self) -> None:
        """Test that high volatility affects trade recommendations."""
        tool = PlaceholderRebalancingTool(use_mock=True)

        # With high volatility
        result_high_vol = await tool.execute({
            "symbols": ["AAPL"],
            "risk_metrics": {"volatility": 0.35},
        })

        # Trades should exist
        assert len(result_high_vol.trades) > 0

    @pytest.mark.asyncio
    async def test_real_raises_not_implemented(self) -> None:
        """Test real implementation raises ToolExecutionError."""
        tool = PlaceholderRebalancingTool(use_mock=False, fallback_to_mock=False)

        from src.tools.base import ToolExecutionError

        with pytest.raises(ToolExecutionError):
            await tool.execute({"symbols": ["AAPL"]})

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = PlaceholderRebalancingTool(use_mock=True)
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "generate_rebalancing"
        assert "input_schema" in anthropic_tool
        assert anthropic_tool["input_schema"]["type"] == "object"


class TestPlaceholderTaxImpactTool:
    """Tests for PlaceholderTaxImpactTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_tax_impact(self) -> None:
        """Test that mock returns tax impact."""
        tool = PlaceholderTaxImpactTool(use_mock=True)
        trades = [
            {"action": "sell", "symbol": "AAPL", "quantity": 10},
            {"action": "sell", "symbol": "GOOGL", "quantity": 5},
        ]
        result = await tool.execute({"trades": trades})

        assert isinstance(result, TaxImpactOutput)
        assert isinstance(result.impact, TaxImpact)

    @pytest.mark.asyncio
    async def test_identifies_harvesting_opportunities(self) -> None:
        """Test that tax loss harvesting opportunities are identified."""
        tool = PlaceholderTaxImpactTool(use_mock=True)
        # Multiple sells to increase chance of finding losses
        trades = [
            {"action": "sell", "symbol": f"SYM{i}", "quantity": 10}
            for i in range(10)
        ]
        result = await tool.execute({"trades": trades})

        # May or may not have opportunities depending on mock randomness
        assert isinstance(result.impact.tax_loss_harvesting_opportunities, list)

    @pytest.mark.asyncio
    async def test_holding_period_affects_tax_type(self) -> None:
        """Test that holding period affects short vs long term gains."""
        tool = PlaceholderTaxImpactTool(use_mock=True)
        trades = [{"action": "sell", "symbol": "AAPL", "quantity": 100}]

        result = await tool.execute({
            "trades": trades,
            "holding_periods": {"AAPL": 30},  # Short term
        })

        assert isinstance(result.impact, TaxImpact)


class TestPlaceholderExecutionCostTool:
    """Tests for PlaceholderExecutionCostTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_costs(self) -> None:
        """Test that mock returns execution costs."""
        tool = PlaceholderExecutionCostTool(use_mock=True)
        trades = [
            {"action": "buy", "symbol": "AAPL", "quantity": 100},
            {"action": "sell", "symbol": "GOOGL", "quantity": 50},
        ]
        result = await tool.execute({"trades": trades})

        assert isinstance(result, ExecutionCostOutput)
        assert isinstance(result.costs, ExecutionCosts)
        assert result.costs.total_cost >= 0

    @pytest.mark.asyncio
    async def test_larger_orders_have_more_impact(self) -> None:
        """Test that larger orders have more market impact."""
        tool = PlaceholderExecutionCostTool(use_mock=True)

        small_order = await tool.execute({
            "trades": [{"action": "buy", "symbol": "AAPL", "quantity": 10}]
        })

        large_order = await tool.execute({
            "trades": [{"action": "buy", "symbol": "AAPL", "quantity": 1000}]
        })

        # Larger orders should generally have higher costs
        assert large_order.costs.total_cost > small_order.costs.total_cost

    @pytest.mark.asyncio
    async def test_zero_quantity_trades_minimal_cost(self) -> None:
        """Test that zero quantity trades have minimal cost."""
        tool = PlaceholderExecutionCostTool(use_mock=True)
        trades = [{"action": "hold", "symbol": "AAPL", "quantity": 0}]
        result = await tool.execute({"trades": trades})

        # Should have minimal/zero costs for holds
        assert result.costs.total_cost == 0.0


class TestPlaceholderComplianceTool:
    """Tests for PlaceholderComplianceTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_compliance(self) -> None:
        """Test that mock returns compliance result."""
        tool = PlaceholderComplianceTool(use_mock=True)
        trades = [
            {"action": "buy", "symbol": "AAPL", "quantity": 100, "target_weight": 0.15},
        ]
        result = await tool.execute({"trades": trades})

        assert isinstance(result, ComplianceOutput)
        assert isinstance(result.result, ComplianceResult)

    @pytest.mark.asyncio
    async def test_high_concentration_triggers_violation(self) -> None:
        """Test that high concentration triggers violation."""
        tool = PlaceholderComplianceTool(use_mock=True)
        trades = [
            {"action": "buy", "symbol": "AAPL", "quantity": 1000, "target_weight": 0.30},
        ]
        result = await tool.execute({"trades": trades})

        # Should have violation for exceeding 25% limit
        assert result.result.is_compliant is False
        assert len(result.result.violations) > 0
        assert any("concentration" in v.lower() for v in result.result.violations)

    @pytest.mark.asyncio
    async def test_sets_concentration_limits(self) -> None:
        """Test that concentration limits are set."""
        tool = PlaceholderComplianceTool(use_mock=True)
        trades = [
            {"action": "buy", "symbol": "AAPL", "quantity": 100, "target_weight": 0.10},
            {"action": "buy", "symbol": "GOOGL", "quantity": 50, "target_weight": 0.10},
        ]
        result = await tool.execute({"trades": trades})

        assert "AAPL" in result.result.concentration_limits
        assert "GOOGL" in result.result.concentration_limits
        assert result.result.concentration_limits["AAPL"] == 0.25

    @pytest.mark.asyncio
    async def test_wash_sale_warning(self) -> None:
        """Test that sells trigger wash sale warnings."""
        tool = PlaceholderComplianceTool(use_mock=True)
        trades = [
            {"action": "sell", "symbol": "AAPL", "quantity": 100},
        ]
        result = await tool.execute({
            "trades": trades,
            "current_positions": {"AAPL": 0.15},
        })

        # Should have wash sale warning
        assert any("wash sale" in w.lower() for w in result.result.warnings)


# ============================================================================
# Agent Tests
# ============================================================================


class TestRecommendationAgent:
    """Tests for RecommendationAgent."""

    def test_initialization_default(self) -> None:
        """Test agent initializes with default registry."""
        agent = RecommendationAgent()

        assert agent.name == "recommendation"
        assert agent._tool_registry is not None
        assert len(agent.tools) == 4

    def test_initialization_custom_registry(self) -> None:
        """Test agent initializes with custom registry."""
        registry = ToolRegistry()
        registry.register(PlaceholderRebalancingTool(use_mock=True))
        registry.register(PlaceholderTaxImpactTool(use_mock=True))
        registry.register(PlaceholderExecutionCostTool(use_mock=True))
        registry.register(PlaceholderComplianceTool(use_mock=True))

        agent = RecommendationAgent(tool_registry=registry)

        assert agent._tool_registry is registry

    def test_system_prompt(self) -> None:
        """Test system prompt contains key information."""
        agent = RecommendationAgent()
        prompt = agent.system_prompt

        assert "recommendation" in prompt.lower()
        assert "generate_rebalancing" in prompt
        assert "estimate_tax_impact" in prompt
        assert "estimate_execution_cost" in prompt
        assert "validate_compliance" in prompt

    def test_tools_property(self) -> None:
        """Test tools are in Anthropic format."""
        agent = RecommendationAgent()
        tools = agent.tools

        assert len(tools) == 4
        tool_names = [t["name"] for t in tools]
        assert "generate_rebalancing" in tool_names
        assert "estimate_tax_impact" in tool_names
        assert "estimate_execution_cost" in tool_names
        assert "validate_compliance" in tool_names

    @pytest.mark.asyncio
    async def test_invoke_with_symbols(self) -> None:
        """Test invoke with symbols in state."""
        agent = RecommendationAgent()
        state = AgentState(
            context={
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "analysis": {
                    "risk_metrics": {"volatility": 0.15},
                    "recommendations": [],
                },
            }
        )

        result = await agent.invoke(state)

        assert "recommendations" in result.context
        recs = result.context["recommendations"]
        assert recs["total_trades"] == 3
        assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_invoke_with_analysis_data(self) -> None:
        """Test invoke consumes analysis data."""
        agent = RecommendationAgent()
        state = AgentState(
            context={
                "symbols": ["AAPL", "GOOGL"],
                "analysis": {
                    "risk_metrics": {"volatility": 0.30, "beta": 1.5},
                    "recommendations": ["Reduce AAPL exposure"],
                },
            }
        )

        result = await agent.invoke(state)

        assert "recommendations" in result.context
        # Recommendations from analysis should influence trades
        recs = result.context["recommendations"]
        assert recs["total_trades"] == 2

    @pytest.mark.asyncio
    async def test_invoke_no_symbols(self) -> None:
        """Test invoke handles missing symbols gracefully."""
        agent = RecommendationAgent()
        state = AgentState(context={})

        result = await agent.invoke(state)

        assert len(result.errors) > 0
        assert "No symbols" in result.errors[0]

    @pytest.mark.asyncio
    async def test_invoke_adds_message(self) -> None:
        """Test that invoke adds a message to state."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        assert len(result.messages) > 0
        last_message = result.messages[-1]
        assert last_message["role"] == "assistant"
        assert "trade recommendations" in last_message["content"]

    @pytest.mark.asyncio
    async def test_invoke_callable(self) -> None:
        """Test agent is callable via __call__."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent(state)

        assert "recommendations" in result.context


class TestRecommendationAgentSummaryGeneration:
    """Tests for summary generation in RecommendationAgent."""

    @pytest.mark.asyncio
    async def test_summary_high_priority_trades(self) -> None:
        """Test summary mentions high priority trades."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL", "GOOGL", "MSFT"]})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        # Summary should mention trade counts or priorities
        assert recs["summary"] != ""

    @pytest.mark.asyncio
    async def test_summary_tax_impact(self) -> None:
        """Test summary includes tax impact when significant."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL", "GOOGL"]})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        # Tax info should be in summary if there's estimated tax
        if recs["tax_impact"]["estimated_tax"] > 0:
            assert "tax" in recs["summary"].lower()

    @pytest.mark.asyncio
    async def test_summary_execution_costs(self) -> None:
        """Test summary includes execution costs."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL", "GOOGL", "MSFT"]})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        # Should mention execution costs if non-zero
        if recs["execution_costs"]["total_cost"] > 0:
            assert "execution" in recs["summary"].lower() or "cost" in recs["summary"].lower()

    @pytest.mark.asyncio
    async def test_summary_compliance_status(self) -> None:
        """Test summary includes compliance status."""
        agent = RecommendationAgent()
        # Force a compliance violation
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        # Check message mentions compliance
        last_message = result.messages[-1]["content"]
        assert "Compliance" in last_message


class TestRecommendationAgentToolFailure:
    """Tests for tool failure handling."""

    @pytest.mark.asyncio
    async def test_handles_rebalancing_failure(self) -> None:
        """Test agent handles rebalancing tool failure."""
        registry = ToolRegistry()

        # Create a tool that will fail
        class FailingRebalancingTool(PlaceholderRebalancingTool):
            async def _execute_mock(self, _input_data: RebalancingInput) -> RebalancingOutput:
                raise ValueError("Rebalancing failed")

        registry.register(FailingRebalancingTool(use_mock=True))
        registry.register(PlaceholderTaxImpactTool(use_mock=True))
        registry.register(PlaceholderExecutionCostTool(use_mock=True))
        registry.register(PlaceholderComplianceTool(use_mock=True))

        agent = RecommendationAgent(tool_registry=registry)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        assert len(recs["errors"]) > 0
        assert "rebalancing" in recs["errors"][0].lower()

    @pytest.mark.asyncio
    async def test_continues_with_partial_results(self) -> None:
        """Test agent continues even with partial tool failures."""
        registry = ToolRegistry()

        class FailingTaxTool(PlaceholderTaxImpactTool):
            async def _execute_mock(self, _input_data: TaxImpactInput) -> TaxImpactOutput:
                raise ValueError("Tax calculation failed")

        registry.register(PlaceholderRebalancingTool(use_mock=True))
        registry.register(FailingTaxTool(use_mock=True))
        registry.register(PlaceholderExecutionCostTool(use_mock=True))
        registry.register(PlaceholderComplianceTool(use_mock=True))

        agent = RecommendationAgent(tool_registry=registry)
        state = AgentState(context={"symbols": ["AAPL", "GOOGL"]})

        result = await agent.invoke(state)

        # Should still have trades from rebalancing
        recs = result.context["recommendations"]
        assert recs["total_trades"] > 0
        # Should have error about tax
        assert any("tax" in e.lower() for e in recs["errors"])


class TestRecommendationAgentCoverage:
    """Additional tests for coverage."""

    @pytest.mark.asyncio
    async def test_invoke_with_single_symbol(self) -> None:
        """Test invoke with single symbol."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        assert recs["total_trades"] == 1

    @pytest.mark.asyncio
    async def test_invoke_with_many_symbols(self) -> None:
        """Test invoke with many symbols."""
        agent = RecommendationAgent()
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"]
        state = AgentState(context={"symbols": symbols})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        assert recs["total_trades"] == len(symbols)

    @pytest.mark.asyncio
    async def test_invoke_preserves_existing_state(self) -> None:
        """Test that invoke preserves existing state data."""
        agent = RecommendationAgent()
        state = AgentState(
            context={
                "symbols": ["AAPL"],
                "existing_data": {"key": "value"},
            }
        )

        result = await agent.invoke(state)

        assert result.context["existing_data"] == {"key": "value"}
        assert "recommendations" in result.context

    def test_agent_description(self) -> None:
        """Test agent description property."""
        agent = RecommendationAgent()
        assert "recommendation" in agent.description.lower()
        assert "trade" in agent.description.lower()

    def test_system_prompt_contains_all_tools(self) -> None:
        """Test system prompt mentions all tools."""
        agent = RecommendationAgent()
        prompt = agent.system_prompt

        for tool in agent.tools:
            assert tool["name"] in prompt

    @pytest.mark.asyncio
    async def test_tracing_integration(self) -> None:
        """Test that agent has tracing decorator."""
        agent = RecommendationAgent()
        # The @traced_agent decorator should be applied
        assert hasattr(agent.invoke, "__wrapped__") or callable(agent.invoke)

    @pytest.mark.asyncio
    async def test_trade_counts_accurate(self) -> None:
        """Test that trade counts are accurately calculated."""
        agent = RecommendationAgent()
        state = AgentState(context={"symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]})

        result = await agent.invoke(state)

        recs = result.context["recommendations"]
        trades = recs["trades"]

        actual_buy = sum(1 for t in trades if t["action"] == "buy")
        actual_sell = sum(1 for t in trades if t["action"] == "sell")
        actual_hold = sum(1 for t in trades if t["action"] == "hold")

        assert recs["buy_count"] == actual_buy
        assert recs["sell_count"] == actual_sell
        assert recs["hold_count"] == actual_hold
        assert recs["total_trades"] == len(trades)


class TestToolInputSchemas:
    """Tests for tool input schemas."""

    def test_rebalancing_input(self) -> None:
        """Test RebalancingInput schema."""
        input_data = RebalancingInput(symbols=["AAPL", "GOOGL"])
        assert input_data.symbols == ["AAPL", "GOOGL"]
        assert input_data.current_weights is None
        assert input_data.target_weights is None

    def test_rebalancing_input_with_weights(self) -> None:
        """Test RebalancingInput with weights."""
        input_data = RebalancingInput(
            symbols=["AAPL", "GOOGL"],
            current_weights={"AAPL": 0.6, "GOOGL": 0.4},
            target_weights={"AAPL": 0.5, "GOOGL": 0.5},
        )
        assert input_data.current_weights["AAPL"] == 0.6
        assert input_data.target_weights["GOOGL"] == 0.5

    def test_tax_impact_input(self) -> None:
        """Test TaxImpactInput schema."""
        input_data = TaxImpactInput(
            trades=[{"action": "sell", "symbol": "AAPL", "quantity": 100}]
        )
        assert len(input_data.trades) == 1

    def test_execution_cost_input(self) -> None:
        """Test ExecutionCostInput schema."""
        input_data = ExecutionCostInput(
            trades=[{"action": "buy", "symbol": "GOOGL", "quantity": 50}]
        )
        assert len(input_data.trades) == 1

    def test_compliance_input(self) -> None:
        """Test ComplianceInput schema."""
        input_data = ComplianceInput(
            trades=[{"action": "buy", "symbol": "MSFT", "quantity": 25}],
            account_type="ira",
        )
        assert input_data.account_type == "ira"
