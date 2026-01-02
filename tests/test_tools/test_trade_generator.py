"""Tests for the Trade Generation Tools."""

import pytest

from src.tools.base import ToolExecutionError
from src.tools.trade_generator import (
    ComplianceInput,
    ComplianceOutput,
    ComplianceResult,
    ComplianceTool,
    ExecutionCostInput,
    ExecutionCostOutput,
    ExecutionCosts,
    ExecutionCostTool,
    RebalancingInput,
    RebalancingOutput,
    RebalancingTool,
    TaxImpact,
    TaxImpactInput,
    TaxImpactOutput,
    TaxImpactTool,
    Trade,
    TradeList,
    _calculate_shares_to_trade,
    _generate_mock_prices,
    _generate_mock_weights,
)

# ============================================================================
# Output Schema Tests
# ============================================================================


class TestTrade:
    """Tests for Trade model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        trade = Trade(action="buy", symbol="AAPL")
        assert trade.action == "buy"
        assert trade.symbol == "AAPL"
        assert trade.quantity == 0
        assert trade.priority == 3
        assert trade.urgency == "opportunistic"

    def test_all_fields(self) -> None:
        """Test with all fields populated."""
        trade = Trade(
            action="sell",
            symbol="MSFT",
            quantity=100,
            estimated_value=42000.0,
            target_weight=0.15,
            current_weight=0.20,
            rationale="Reduce allocation",
            priority=2,
            urgency="soon",
            tax_lot_id="LOT123",
        )
        assert trade.quantity == 100
        assert trade.estimated_value == 42000.0
        assert trade.tax_lot_id == "LOT123"


class TestTradeList:
    """Tests for TradeList model."""

    def test_default_values(self) -> None:
        """Test default values."""
        trade_list = TradeList()
        assert trade_list.trades == []
        assert trade_list.total_value == 0.0
        assert trade_list.total_buys == 0
        assert trade_list.total_sells == 0
        assert trade_list.total_holds == 0
        assert trade_list.generated_at is not None

    def test_with_trades(self) -> None:
        """Test with populated trades."""
        trades = [
            Trade(action="buy", symbol="AAPL"),
            Trade(action="sell", symbol="MSFT"),
        ]
        trade_list = TradeList(
            trades=trades,
            total_value=10000.0,
            total_buys=1,
            total_sells=1,
            net_cash_flow=-5000.0,
        )
        assert len(trade_list.trades) == 2
        assert trade_list.net_cash_flow == -5000.0


class TestTaxImpact:
    """Tests for TaxImpact model."""

    def test_default_values(self) -> None:
        """Test default values."""
        impact = TaxImpact()
        assert impact.short_term_gains == 0.0
        assert impact.long_term_gains == 0.0
        assert impact.estimated_tax == 0.0
        assert impact.tax_loss_harvesting_opportunities == []

    def test_with_data(self) -> None:
        """Test with populated data."""
        impact = TaxImpact(
            short_term_gains=1000.0,
            long_term_gains=5000.0,
            short_term_losses=200.0,
            long_term_losses=0.0,
            net_gains=5800.0,
            estimated_tax=1150.0,
            effective_tax_rate=0.20,
            tax_loss_harvesting_opportunities=["XYZ"],
            wash_sale_warnings=["ABC: potential wash sale"],
        )
        assert impact.net_gains == 5800.0
        assert len(impact.wash_sale_warnings) == 1


class TestExecutionCosts:
    """Tests for ExecutionCosts model."""

    def test_default_values(self) -> None:
        """Test default values."""
        costs = ExecutionCosts()
        assert costs.total_commission == 0.0
        assert costs.total_cost == 0.0
        assert costs.recommended_execution == "market"

    def test_with_data(self) -> None:
        """Test with populated data."""
        costs = ExecutionCosts(
            total_commission=9.90,
            estimated_spread_cost=25.0,
            estimated_slippage=15.0,
            market_impact=10.0,
            total_cost=59.90,
            cost_as_percent=0.06,
            recommended_execution="vwap",
        )
        assert costs.total_cost == 59.90
        assert costs.recommended_execution == "vwap"


class TestComplianceResult:
    """Tests for ComplianceResult model."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = ComplianceResult()
        assert result.is_compliant is True
        assert result.violations == []
        assert result.warnings == []
        assert result.requires_approval is False

    def test_with_violations(self) -> None:
        """Test with compliance violations."""
        result = ComplianceResult(
            is_compliant=False,
            violations=["AAPL: exceeds concentration limit"],
            warnings=["MSFT: approaching limit"],
            concentration_limits={"AAPL": 0.25, "MSFT": 0.25},
            restricted_symbols=["RESTRICTED1"],
            requires_approval=True,
        )
        assert result.is_compliant is False
        assert len(result.violations) == 1
        assert result.requires_approval is True


# ============================================================================
# Input Schema Tests
# ============================================================================


class TestRebalancingInput:
    """Tests for RebalancingInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = RebalancingInput(symbols=["AAPL", "MSFT"])
        assert input_data.symbols == ["AAPL", "MSFT"]
        assert input_data.portfolio_value == 100000.0
        assert input_data.min_trade_value == 100.0
        assert input_data.rebalance_threshold == 0.05

    def test_with_weights(self) -> None:
        """Test with weights provided."""
        input_data = RebalancingInput(
            symbols=["AAPL", "MSFT"],
            current_weights={"AAPL": 0.6, "MSFT": 0.4},
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
        )
        assert input_data.current_weights["AAPL"] == 0.6
        assert input_data.target_weights["MSFT"] == 0.5


class TestTaxImpactInput:
    """Tests for TaxImpactInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = TaxImpactInput(trades=[{"action": "sell", "symbol": "AAPL"}])
        assert len(input_data.trades) == 1
        assert input_data.tax_bracket == 0.32

    def test_with_cost_basis(self) -> None:
        """Test with cost basis provided."""
        input_data = TaxImpactInput(
            trades=[{"action": "sell", "symbol": "AAPL", "quantity": 100}],
            cost_basis={"AAPL": 150.0},
            current_prices={"AAPL": 185.0},
            holding_periods={"AAPL": 400},
        )
        assert input_data.cost_basis["AAPL"] == 150.0
        assert input_data.holding_periods["AAPL"] == 400


class TestExecutionCostInput:
    """Tests for ExecutionCostInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = ExecutionCostInput(trades=[{"action": "buy", "symbol": "AAPL"}])
        assert len(input_data.trades) == 1
        assert input_data.commission_per_trade == 0.0

    def test_with_market_data(self) -> None:
        """Test with market data provided."""
        input_data = ExecutionCostInput(
            trades=[{"action": "buy", "symbol": "AAPL", "quantity": 100}],
            current_prices={"AAPL": 185.0},
            average_daily_volume={"AAPL": 50000000},
            bid_ask_spreads={"AAPL": 0.02},
        )
        assert input_data.average_daily_volume["AAPL"] == 50000000


class TestComplianceInput:
    """Tests for ComplianceInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = ComplianceInput(trades=[{"action": "buy", "symbol": "AAPL"}])
        assert input_data.account_type == "taxable"
        assert input_data.concentration_limit == 0.25
        assert input_data.sector_limit == 0.40

    def test_with_account_type(self) -> None:
        """Test with different account type."""
        input_data = ComplianceInput(
            trades=[{"action": "buy", "symbol": "AAPL"}],
            account_type="ira",
            portfolio_value=500000.0,
        )
        assert input_data.account_type == "ira"
        assert input_data.portfolio_value == 500000.0


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestGenerateMockPrices:
    """Tests for _generate_mock_prices function."""

    def test_known_symbols(self) -> None:
        """Test that known symbols return expected prices."""
        prices = _generate_mock_prices(["AAPL", "MSFT", "GOOGL"])
        assert prices["AAPL"] == 185.0
        assert prices["MSFT"] == 420.0
        assert prices["GOOGL"] == 175.0

    def test_unknown_symbols(self) -> None:
        """Test that unknown symbols get generated prices."""
        prices = _generate_mock_prices(["UNKNOWN1", "UNKNOWN2"])
        assert "UNKNOWN1" in prices
        assert "UNKNOWN2" in prices
        assert 20 <= prices["UNKNOWN1"] <= 500
        assert 20 <= prices["UNKNOWN2"] <= 500

    def test_deterministic(self) -> None:
        """Test that prices are deterministic."""
        prices1 = _generate_mock_prices(["UNKNOWN"])
        prices2 = _generate_mock_prices(["UNKNOWN"])
        assert prices1 == prices2


class TestGenerateMockWeights:
    """Tests for _generate_mock_weights function."""

    def test_weights_sum_to_one(self) -> None:
        """Test that weights sum to approximately 1.0."""
        weights = _generate_mock_weights(["AAPL", "MSFT", "GOOGL", "AMZN"])
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_all_positive(self) -> None:
        """Test that all weights are positive."""
        weights = _generate_mock_weights(["A", "B", "C", "D", "E"])
        assert all(w > 0 for w in weights.values())

    def test_deterministic(self) -> None:
        """Test that weights are deterministic."""
        weights1 = _generate_mock_weights(["AAPL", "MSFT"])
        weights2 = _generate_mock_weights(["AAPL", "MSFT"])
        assert weights1 == weights2


class TestCalculateSharesToTrade:
    """Tests for _calculate_shares_to_trade function."""

    def test_buy_shares(self) -> None:
        """Test calculating shares to buy."""
        shares = _calculate_shares_to_trade(
            current_value=10000,
            target_value=15000,
            price=100,
        )
        assert shares == 50

    def test_sell_shares(self) -> None:
        """Test calculating shares to sell."""
        shares = _calculate_shares_to_trade(
            current_value=15000,
            target_value=10000,
            price=100,
        )
        assert shares == -50

    def test_zero_price(self) -> None:
        """Test with zero price."""
        shares = _calculate_shares_to_trade(
            current_value=10000,
            target_value=15000,
            price=0,
        )
        assert shares == 0

    def test_fractional_shares_truncated(self) -> None:
        """Test that fractional shares are truncated."""
        shares = _calculate_shares_to_trade(
            current_value=10000,
            target_value=10150,
            price=100,
        )
        assert shares == 1  # 150/100 = 1.5 -> 1


# ============================================================================
# Rebalancing Tool Tests
# ============================================================================


class TestRebalancingTool:
    """Tests for RebalancingTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = RebalancingTool()
        assert tool.name == "generate_rebalancing"
        assert "rebalancing" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = RebalancingTool()
        anthropic_tool = tool.to_anthropic_tool()
        assert anthropic_tool["name"] == "generate_rebalancing"
        assert "symbols" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_basic(self) -> None:
        """Test basic execution."""
        tool = RebalancingTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "GOOGL"]})

        assert isinstance(result, RebalancingOutput)
        assert result.success is True
        assert len(result.trade_list.trades) == 3

    @pytest.mark.asyncio
    async def test_execute_with_weights(self) -> None:
        """Test execution with specified weights."""
        tool = RebalancingTool(use_mock=True)
        result = await tool.execute({
            "symbols": ["AAPL", "MSFT"],
            "current_weights": {"AAPL": 0.60, "MSFT": 0.40},
            "target_weights": {"AAPL": 0.50, "MSFT": 0.50},
        })

        assert result.success is True
        # Should have trades to rebalance
        trades = result.trade_list.trades
        assert any(t.action in ["buy", "sell"] for t in trades)

    @pytest.mark.asyncio
    async def test_hold_when_within_threshold(self) -> None:
        """Test that positions within threshold result in hold."""
        tool = RebalancingTool(use_mock=True)
        result = await tool.execute({
            "symbols": ["AAPL"],
            "current_weights": {"AAPL": 0.50},
            "target_weights": {"AAPL": 0.52},  # Only 2% deviation
            "rebalance_threshold": 0.05,
        })

        assert result.success is True
        assert result.trade_list.trades[0].action == "hold"

    @pytest.mark.asyncio
    async def test_trade_when_exceeds_threshold(self) -> None:
        """Test that positions exceeding threshold result in trade."""
        tool = RebalancingTool(use_mock=True)
        result = await tool.execute({
            "symbols": ["AAPL"],
            "current_weights": {"AAPL": 0.30},
            "target_weights": {"AAPL": 0.50},  # 20% deviation
            "rebalance_threshold": 0.05,
        })

        assert result.success is True
        trade = result.trade_list.trades[0]
        assert trade.action == "buy"
        assert trade.urgency in ["immediate", "soon"]

    @pytest.mark.asyncio
    async def test_summary_generated(self) -> None:
        """Test that summary is generated."""
        tool = RebalancingTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT"]})

        assert result.rebalance_summary != ""
        assert "Rebalancing" in result.rebalance_summary


# ============================================================================
# Tax Impact Tool Tests
# ============================================================================


class TestTaxImpactTool:
    """Tests for TaxImpactTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = TaxImpactTool()
        assert tool.name == "estimate_tax_impact"
        assert "tax" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = TaxImpactTool()
        anthropic_tool = tool.to_anthropic_tool()
        assert anthropic_tool["name"] == "estimate_tax_impact"
        assert "trades" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_no_sells(self) -> None:
        """Test execution with no sells (no tax impact)."""
        tool = TaxImpactTool(use_mock=True)
        result = await tool.execute({
            "trades": [{"action": "buy", "symbol": "AAPL", "quantity": 100}]
        })

        assert isinstance(result, TaxImpactOutput)
        assert result.success is True
        # No sells = no gains
        assert result.impact.short_term_gains == 0.0
        assert result.impact.long_term_gains == 0.0

    @pytest.mark.asyncio
    async def test_execute_with_sells(self) -> None:
        """Test execution with sells."""
        tool = TaxImpactTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "sell", "symbol": "AAPL", "quantity": 100},
                {"action": "sell", "symbol": "MSFT", "quantity": 50},
            ]
        })

        assert result.success is True
        # Should have some gains or losses calculated
        total_gains = (
            result.impact.short_term_gains
            + result.impact.long_term_gains
            - result.impact.short_term_losses
            - result.impact.long_term_losses
        )
        assert total_gains != 0.0 or result.impact.net_gains != 0.0

    @pytest.mark.asyncio
    async def test_short_term_vs_long_term(self) -> None:
        """Test classification of short vs long term gains."""
        tool = TaxImpactTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "sell", "symbol": "AAPL", "quantity": 100},
            ],
            "holding_periods": {"AAPL": 100},  # Short-term
            "cost_basis": {"AAPL": 150.0},
            "current_prices": {"AAPL": 185.0},
        })

        assert result.success is True
        # Should have short-term gain (185-150)*100 = 3500
        assert result.impact.short_term_gains > 0

    @pytest.mark.asyncio
    async def test_wash_sale_warning(self) -> None:
        """Test wash sale warning detection."""
        tool = TaxImpactTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "sell", "symbol": "AAPL", "quantity": 100},
                {"action": "buy", "symbol": "AAPL", "quantity": 50},  # Same symbol
            ]
        })

        assert result.success is True
        assert len(result.impact.wash_sale_warnings) > 0
        assert "AAPL" in result.impact.wash_sale_warnings[0]


# ============================================================================
# Execution Cost Tool Tests
# ============================================================================


class TestExecutionCostTool:
    """Tests for ExecutionCostTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = ExecutionCostTool()
        assert tool.name == "estimate_execution_cost"
        assert "execution" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = ExecutionCostTool()
        anthropic_tool = tool.to_anthropic_tool()
        assert anthropic_tool["name"] == "estimate_execution_cost"
        assert "trades" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_basic(self) -> None:
        """Test basic execution."""
        tool = ExecutionCostTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL", "quantity": 100},
                {"action": "sell", "symbol": "MSFT", "quantity": 50},
            ]
        })

        assert isinstance(result, ExecutionCostOutput)
        assert result.success is True
        assert result.costs.total_cost > 0

    @pytest.mark.asyncio
    async def test_holds_have_no_cost(self) -> None:
        """Test that hold trades have no execution cost."""
        tool = ExecutionCostTool(use_mock=True)
        result = await tool.execute({
            "trades": [{"action": "hold", "symbol": "AAPL", "quantity": 0}]
        })

        assert result.success is True
        assert result.costs.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_zero_quantity_no_cost(self) -> None:
        """Test that zero quantity trades have no cost."""
        tool = ExecutionCostTool(use_mock=True)
        result = await tool.execute({
            "trades": [{"action": "buy", "symbol": "AAPL", "quantity": 0}]
        })

        assert result.success is True
        assert result.costs.total_cost == 0.0

    @pytest.mark.asyncio
    async def test_recommended_execution_strategy(self) -> None:
        """Test that execution strategy is recommended."""
        tool = ExecutionCostTool(use_mock=True)
        result = await tool.execute({
            "trades": [{"action": "buy", "symbol": "AAPL", "quantity": 100}]
        })

        assert result.success is True
        assert result.costs.recommended_execution in ["market", "vwap", "twap"]

    @pytest.mark.asyncio
    async def test_with_commission(self) -> None:
        """Test with commission per trade."""
        tool = ExecutionCostTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL", "quantity": 100},
                {"action": "sell", "symbol": "MSFT", "quantity": 50},
            ],
            "commission_per_trade": 4.95,
        })

        assert result.success is True
        assert result.costs.total_commission == 9.90  # 2 trades * $4.95


# ============================================================================
# Compliance Tool Tests
# ============================================================================


class TestComplianceTool:
    """Tests for ComplianceTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = ComplianceTool()
        assert tool.name == "validate_compliance"
        assert "compliance" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = ComplianceTool()
        anthropic_tool = tool.to_anthropic_tool()
        assert anthropic_tool["name"] == "validate_compliance"
        assert "trades" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_compliant(self) -> None:
        """Test execution with compliant trades."""
        tool = ComplianceTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL", "target_weight": 0.10},
                {"action": "buy", "symbol": "MSFT", "target_weight": 0.10},
            ]
        })

        assert isinstance(result, ComplianceOutput)
        assert result.success is True
        assert result.result.is_compliant is True

    @pytest.mark.asyncio
    async def test_concentration_violation(self) -> None:
        """Test detection of concentration limit violation."""
        tool = ComplianceTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL", "target_weight": 0.30},
            ],
            "concentration_limit": 0.25,
        })

        assert result.success is True
        assert result.result.is_compliant is False
        assert len(result.result.violations) > 0
        assert "concentration" in result.result.violations[0].lower()

    @pytest.mark.asyncio
    async def test_sector_concentration(self) -> None:
        """Test detection of sector concentration."""
        tool = ComplianceTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL", "target_weight": 0.20},
                {"action": "buy", "symbol": "MSFT", "target_weight": 0.20},
                {"action": "buy", "symbol": "GOOGL", "target_weight": 0.20},
            ],
            "sector_limit": 0.40,  # 60% tech should violate
        })

        assert result.success is True
        # Should have sector violation or warning
        has_sector_issue = (
            any("sector" in v.lower() for v in result.result.violations)
            or any("sector" in w.lower() for w in result.result.warnings)
        )
        assert has_sector_issue

    @pytest.mark.asyncio
    async def test_ira_short_sell_violation(self) -> None:
        """Test that short selling in IRA is flagged."""
        tool = ComplianceTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "sell", "symbol": "AAPL", "estimated_value": 10000},
            ],
            "current_positions": {"AAPL": 5000},  # Less than sell value
            "account_type": "ira",
        })

        assert result.success is True
        assert result.result.is_compliant is False
        assert any("short" in v.lower() for v in result.result.violations)

    @pytest.mark.asyncio
    async def test_large_trade_approval(self) -> None:
        """Test that large trades require approval."""
        tool = ComplianceTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL", "estimated_value": 30000, "target_weight": 0.20},
            ],
            "portfolio_value": 100000,
        })

        assert result.success is True
        assert result.result.requires_approval is True

    @pytest.mark.asyncio
    async def test_concentration_limits_set(self) -> None:
        """Test that concentration limits are set for each symbol."""
        tool = ComplianceTool(use_mock=True)
        result = await tool.execute({
            "trades": [
                {"action": "buy", "symbol": "AAPL"},
                {"action": "buy", "symbol": "MSFT"},
            ],
            "concentration_limit": 0.25,
        })

        assert result.success is True
        assert "AAPL" in result.result.concentration_limits
        assert "MSFT" in result.result.concentration_limits
        assert result.result.concentration_limits["AAPL"] == 0.25


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation across tools."""

    @pytest.mark.asyncio
    async def test_rebalancing_missing_symbols(self) -> None:
        """Test that missing symbols raises error."""
        tool = RebalancingTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_tax_impact_missing_trades(self) -> None:
        """Test that missing trades raises error."""
        tool = TaxImpactTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_execution_cost_missing_trades(self) -> None:
        """Test that missing trades raises error."""
        tool = ExecutionCostTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_compliance_missing_trades(self) -> None:
        """Test that missing trades raises error."""
        tool = ComplianceTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_accepts_model_input(self) -> None:
        """Test that validated model can be passed directly."""
        tool = RebalancingTool(use_mock=True)
        input_data = RebalancingInput(symbols=["AAPL", "MSFT"])

        result = await tool.execute(input_data)

        assert result.success is True
