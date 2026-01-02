"""Trade Generation Tools for portfolio recommendations.

This module provides tools for generating and validating trade recommendations:
- Rebalancing trade generation
- Tax impact estimation
- Execution cost estimation
- Compliance validation

Note: For learning purposes, these use simplified calculations.
Production implementations would integrate with real tax and compliance systems.
"""

import hashlib
import random
from datetime import UTC, datetime
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolInput, ToolOutput

logger = structlog.get_logger(__name__)


# ============================================================================
# Output Schemas
# ============================================================================


class Trade(BaseModel):
    """A single trade recommendation."""

    action: Literal["buy", "sell", "hold"] = Field(
        ..., description="Trade action to take"
    )
    symbol: str = Field(..., description="Stock ticker symbol")
    quantity: int = Field(default=0, description="Number of shares")
    estimated_value: float = Field(default=0.0, description="Estimated trade value in dollars")
    target_weight: float | None = Field(
        default=None, description="Target portfolio weight (0-1)"
    )
    current_weight: float | None = Field(
        default=None, description="Current portfolio weight (0-1)"
    )
    rationale: str = Field(default="", description="Reason for the trade")
    priority: int = Field(
        default=3, ge=1, le=5, description="Priority (1=highest, 5=lowest)"
    )
    urgency: Literal["immediate", "soon", "opportunistic"] = Field(
        default="opportunistic", description="Trade urgency level"
    )
    tax_lot_id: str | None = Field(default=None, description="Specific tax lot to sell")


class TaxImpact(BaseModel):
    """Estimated tax impact of trades."""

    short_term_gains: float = Field(default=0.0, description="Short-term capital gains")
    long_term_gains: float = Field(default=0.0, description="Long-term capital gains")
    short_term_losses: float = Field(default=0.0, description="Short-term capital losses")
    long_term_losses: float = Field(default=0.0, description="Long-term capital losses")
    net_gains: float = Field(default=0.0, description="Net capital gains")
    estimated_tax: float = Field(default=0.0, description="Estimated tax liability")
    effective_tax_rate: float = Field(default=0.0, description="Effective tax rate on gains")
    tax_loss_harvesting_opportunities: list[str] = Field(
        default_factory=list, description="Symbols with harvesting opportunities"
    )
    wash_sale_warnings: list[str] = Field(
        default_factory=list, description="Potential wash sale violations"
    )


class ExecutionCosts(BaseModel):
    """Estimated execution costs."""

    total_commission: float = Field(default=0.0, description="Total commission cost")
    estimated_spread_cost: float = Field(default=0.0, description="Bid-ask spread cost")
    estimated_slippage: float = Field(default=0.0, description="Estimated slippage")
    market_impact: float = Field(default=0.0, description="Estimated market impact")
    total_cost: float = Field(default=0.0, description="Total execution cost")
    cost_as_percent: float = Field(default=0.0, description="Cost as percentage of trade value")
    recommended_execution: str = Field(
        default="market", description="Recommended execution strategy"
    )


class ComplianceResult(BaseModel):
    """Compliance validation result."""

    is_compliant: bool = Field(default=True, description="Whether trades are compliant")
    violations: list[str] = Field(default_factory=list, description="Compliance violations")
    warnings: list[str] = Field(default_factory=list, description="Compliance warnings")
    concentration_limits: dict[str, float] = Field(
        default_factory=dict, description="Position concentration limits"
    )
    restricted_symbols: list[str] = Field(
        default_factory=list, description="Symbols that cannot be traded"
    )
    requires_approval: bool = Field(
        default=False, description="Whether trades require additional approval"
    )


class TradeList(BaseModel):
    """Collection of trades with summary statistics."""

    trades: list[Trade] = Field(default_factory=list)
    total_value: float = Field(default=0.0, description="Total value of all trades")
    total_buys: int = Field(default=0, description="Number of buy trades")
    total_sells: int = Field(default=0, description="Number of sell trades")
    total_holds: int = Field(default=0, description="Number of hold recommendations")
    net_cash_flow: float = Field(default=0.0, description="Net cash impact (positive = inflow)")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ============================================================================
# Tool Input/Output Schemas
# ============================================================================


class RebalancingInput(ToolInput):
    """Input for rebalancing recommendations."""

    symbols: list[str] = Field(..., description="List of symbols in portfolio")
    current_weights: dict[str, float] | None = Field(
        default=None, description="Current portfolio weights"
    )
    target_weights: dict[str, float] | None = Field(
        default=None, description="Target portfolio weights"
    )
    portfolio_value: float = Field(
        default=100000.0, description="Total portfolio value"
    )
    current_prices: dict[str, float] | None = Field(
        default=None, description="Current prices per symbol"
    )
    min_trade_value: float = Field(
        default=100.0, description="Minimum trade value threshold"
    )
    rebalance_threshold: float = Field(
        default=0.05, description="Minimum weight deviation to trigger trade"
    )


class RebalancingOutput(ToolOutput):
    """Output from rebalancing tool."""

    trade_list: TradeList = Field(default_factory=lambda: TradeList())
    rebalance_summary: str = ""


class TaxImpactInput(ToolInput):
    """Input for tax impact estimation."""

    trades: list[dict[str, Any]] = Field(..., description="Proposed trades")
    cost_basis: dict[str, float] | None = Field(
        default=None, description="Cost basis per symbol"
    )
    current_prices: dict[str, float] | None = Field(
        default=None, description="Current prices per symbol"
    )
    holding_periods: dict[str, int] | None = Field(
        default=None, description="Days held per symbol"
    )
    tax_bracket: float = Field(
        default=0.32, description="Marginal tax bracket for short-term gains"
    )
    state_tax_rate: float = Field(
        default=0.05, description="State tax rate"
    )


class TaxImpactOutput(ToolOutput):
    """Output from tax impact estimation."""

    impact: TaxImpact = Field(default_factory=lambda: TaxImpact())


class ExecutionCostInput(ToolInput):
    """Input for execution cost estimation."""

    trades: list[dict[str, Any]] = Field(..., description="Proposed trades")
    current_prices: dict[str, float] | None = Field(
        default=None, description="Current prices per symbol"
    )
    average_daily_volume: dict[str, int] | None = Field(
        default=None, description="ADV per symbol"
    )
    bid_ask_spreads: dict[str, float] | None = Field(
        default=None, description="Bid-ask spread per symbol"
    )
    commission_per_trade: float = Field(
        default=0.0, description="Commission per trade (0 for commission-free)"
    )


class ExecutionCostOutput(ToolOutput):
    """Output from execution cost estimation."""

    costs: ExecutionCosts = Field(default_factory=lambda: ExecutionCosts())


class ComplianceInput(ToolInput):
    """Input for compliance validation."""

    trades: list[dict[str, Any]] = Field(..., description="Proposed trades")
    current_positions: dict[str, float] | None = Field(
        default=None, description="Current position values"
    )
    portfolio_value: float = Field(
        default=100000.0, description="Total portfolio value"
    )
    account_type: Literal["taxable", "ira", "roth_ira", "401k"] = Field(
        default="taxable", description="Account type"
    )
    concentration_limit: float = Field(
        default=0.25, description="Maximum single position concentration"
    )
    sector_limit: float = Field(
        default=0.40, description="Maximum sector concentration"
    )


class ComplianceOutput(ToolOutput):
    """Output from compliance validation."""

    result: ComplianceResult = Field(default_factory=lambda: ComplianceResult())


# ============================================================================
# Utility Functions
# ============================================================================


def _generate_mock_prices(symbols: list[str]) -> dict[str, float]:
    """Generate deterministic mock prices for symbols.

    Args:
        symbols: List of stock symbols.

    Returns:
        Dictionary mapping symbols to prices.
    """
    # Base prices for known symbols
    base_prices = {
        "AAPL": 185.0,
        "MSFT": 420.0,
        "GOOGL": 175.0,
        "GOOG": 176.0,
        "AMZN": 185.0,
        "META": 500.0,
        "NVDA": 875.0,
        "TSLA": 250.0,
        "JPM": 195.0,
        "JNJ": 160.0,
        "UNH": 525.0,
        "V": 280.0,
        "PG": 165.0,
        "XOM": 105.0,
        "HD": 380.0,
        "CVX": 150.0,
        "BAC": 35.0,
        "ABBV": 175.0,
        "PFE": 28.0,
        "KO": 62.0,
        "SPY": 475.0,
        "QQQ": 420.0,
    }

    prices = {}
    for symbol in symbols:
        if symbol in base_prices:
            prices[symbol] = base_prices[symbol]
        else:
            # Generate deterministic price based on symbol hash
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            prices[symbol] = round(random.uniform(20, 500), 2)

    return prices


def _generate_mock_weights(symbols: list[str]) -> dict[str, float]:
    """Generate deterministic mock portfolio weights.

    Args:
        symbols: List of stock symbols.

    Returns:
        Dictionary mapping symbols to weights (sum to 1.0).
    """
    weights = {}
    remaining = 1.0

    for i, symbol in enumerate(symbols):
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        if i == len(symbols) - 1:
            weights[symbol] = round(remaining, 4)
        else:
            max_weight = remaining - (len(symbols) - i - 1) * 0.02
            weight = round(random.uniform(0.02, min(0.30, max_weight)), 4)
            weights[symbol] = weight
            remaining -= weight

    return weights


def _calculate_shares_to_trade(
    current_value: float,
    target_value: float,
    price: float,
) -> int:
    """Calculate number of shares to trade.

    Args:
        current_value: Current position value.
        target_value: Target position value.
        price: Current share price.

    Returns:
        Number of shares (positive for buy, negative for sell).
    """
    if price <= 0:
        return 0
    value_diff = target_value - current_value
    shares = int(value_diff / price)
    return shares


# ============================================================================
# Rebalancing Tool
# ============================================================================


class RebalancingTool(BaseTool[RebalancingInput, RebalancingOutput]):
    """Tool for generating portfolio rebalancing trades.

    Calculates trades needed to move from current weights to target weights,
    considering minimum trade thresholds and rebalancing bands.
    """

    name = "generate_rebalancing"
    description = (
        "Generates rebalancing trade recommendations to align portfolio with target allocations. "
        "Considers minimum trade sizes and rebalancing thresholds."
    )

    @property
    def input_schema(self) -> type[RebalancingInput]:
        return RebalancingInput

    @property
    def output_schema(self) -> type[RebalancingOutput]:
        return RebalancingOutput

    async def _execute_real(self, input_data: RebalancingInput) -> RebalancingOutput:
        """Calculate rebalancing trades from provided data."""
        return await self._calculate_rebalancing(input_data)

    async def _execute_mock(self, input_data: RebalancingInput) -> RebalancingOutput:
        """Calculate rebalancing trades with mock data."""
        return await self._calculate_rebalancing(input_data)

    async def _calculate_rebalancing(
        self, input_data: RebalancingInput
    ) -> RebalancingOutput:
        """Core rebalancing calculation logic.

        Args:
            input_data: Rebalancing parameters.

        Returns:
            Trade recommendations.
        """
        symbols = input_data.symbols
        portfolio_value = input_data.portfolio_value

        # Get or generate prices
        prices = input_data.current_prices or _generate_mock_prices(symbols)

        # Get or generate current weights
        current_weights = input_data.current_weights or _generate_mock_weights(symbols)

        # Get or generate target weights (equal weight if not specified)
        if input_data.target_weights:
            target_weights = input_data.target_weights
        else:
            equal_weight = 1.0 / len(symbols)
            target_weights = dict.fromkeys(symbols, equal_weight)

        trades: list[Trade] = []
        total_value = 0.0
        net_cash_flow = 0.0

        for symbol in symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            price = prices.get(symbol, 100.0)

            # Calculate weight deviation
            deviation = target_weight - current_weight

            # Check if deviation exceeds threshold
            if abs(deviation) < input_data.rebalance_threshold:
                # Hold - deviation too small
                trades.append(
                    Trade(
                        action="hold",
                        symbol=symbol,
                        quantity=0,
                        estimated_value=0.0,
                        current_weight=round(current_weight, 4),
                        target_weight=round(target_weight, 4),
                        rationale=f"Weight deviation ({deviation:.1%}) below threshold",
                        priority=5,
                        urgency="opportunistic",
                    )
                )
                continue

            # Calculate trade value
            current_value = current_weight * portfolio_value
            target_value = target_weight * portfolio_value
            trade_value = abs(target_value - current_value)

            # Check minimum trade value
            if trade_value < input_data.min_trade_value:
                trades.append(
                    Trade(
                        action="hold",
                        symbol=symbol,
                        quantity=0,
                        estimated_value=0.0,
                        current_weight=round(current_weight, 4),
                        target_weight=round(target_weight, 4),
                        rationale=f"Trade value (${trade_value:.0f}) below minimum",
                        priority=5,
                        urgency="opportunistic",
                    )
                )
                continue

            # Calculate shares
            shares = _calculate_shares_to_trade(current_value, target_value, price)

            if shares == 0:
                trades.append(
                    Trade(
                        action="hold",
                        symbol=symbol,
                        quantity=0,
                        estimated_value=0.0,
                        current_weight=round(current_weight, 4),
                        target_weight=round(target_weight, 4),
                        rationale="No whole shares to trade",
                        priority=5,
                        urgency="opportunistic",
                    )
                )
                continue

            # Determine action and priority
            action: Literal["buy", "sell", "hold"]
            if shares > 0:
                action = "buy"
                rationale = f"Increase {symbol} allocation by {deviation:.1%}"
                priority = 3 if deviation < 0.10 else 2
                net_cash_flow -= trade_value
            else:
                action = "sell"
                rationale = f"Reduce {symbol} allocation by {abs(deviation):.1%}"
                priority = 3 if abs(deviation) < 0.10 else 2
                net_cash_flow += trade_value

            # Determine urgency
            urgency: Literal["immediate", "soon", "opportunistic"]
            if abs(deviation) > 0.15:
                urgency = "immediate"
            elif abs(deviation) > 0.08:
                urgency = "soon"
            else:
                urgency = "opportunistic"

            trades.append(
                Trade(
                    action=action,
                    symbol=symbol,
                    quantity=abs(shares),
                    estimated_value=round(trade_value, 2),
                    current_weight=round(current_weight, 4),
                    target_weight=round(target_weight, 4),
                    rationale=rationale,
                    priority=priority,
                    urgency=urgency,
                )
            )
            total_value += trade_value

        # Sort by priority
        trades.sort(key=lambda t: (t.priority, -t.estimated_value))

        # Count trade types
        total_buys = sum(1 for t in trades if t.action == "buy")
        total_sells = sum(1 for t in trades if t.action == "sell")
        total_holds = sum(1 for t in trades if t.action == "hold")

        trade_list = TradeList(
            trades=trades,
            total_value=round(total_value, 2),
            total_buys=total_buys,
            total_sells=total_sells,
            total_holds=total_holds,
            net_cash_flow=round(net_cash_flow, 2),
        )

        summary = (
            f"Rebalancing: {total_buys} buys, {total_sells} sells, {total_holds} holds. "
            f"Total trade value: ${total_value:,.2f}. Net cash: ${net_cash_flow:+,.2f}"
        )

        return RebalancingOutput(trade_list=trade_list, rebalance_summary=summary)


# ============================================================================
# Tax Impact Tool
# ============================================================================


class TaxImpactTool(BaseTool[TaxImpactInput, TaxImpactOutput]):
    """Tool for estimating tax impact of trades.

    Calculates short-term and long-term capital gains/losses,
    estimated tax liability, and identifies tax-loss harvesting opportunities.
    """

    name = "estimate_tax_impact"
    description = (
        "Estimates tax impact of proposed trades including short-term and long-term "
        "capital gains, wash sale warnings, and tax-loss harvesting opportunities."
    )

    @property
    def input_schema(self) -> type[TaxImpactInput]:
        return TaxImpactInput

    @property
    def output_schema(self) -> type[TaxImpactOutput]:
        return TaxImpactOutput

    async def _execute_real(self, input_data: TaxImpactInput) -> TaxImpactOutput:
        """Calculate tax impact from provided data."""
        return await self._calculate_tax_impact(input_data)

    async def _execute_mock(self, input_data: TaxImpactInput) -> TaxImpactOutput:
        """Calculate tax impact with mock data."""
        return await self._calculate_tax_impact(input_data)

    async def _calculate_tax_impact(
        self, input_data: TaxImpactInput
    ) -> TaxImpactOutput:
        """Core tax impact calculation logic.

        Args:
            input_data: Tax calculation parameters.

        Returns:
            Tax impact estimation.
        """
        trades = input_data.trades
        cost_basis = input_data.cost_basis or {}
        current_prices = input_data.current_prices or {}
        holding_periods = input_data.holding_periods or {}

        # Tax rates
        short_term_rate = input_data.tax_bracket + input_data.state_tax_rate
        long_term_rate = 0.15 + input_data.state_tax_rate  # Assume 15% LTCG rate

        short_term_gains = 0.0
        long_term_gains = 0.0
        short_term_losses = 0.0
        long_term_losses = 0.0
        harvesting_opportunities: list[str] = []
        wash_sale_warnings: list[str] = []

        # Track symbols being sold for wash sale detection
        sold_symbols: set[str] = set()

        for trade in trades:
            if trade.get("action") != "sell":
                continue

            symbol = trade.get("symbol", "")
            quantity = trade.get("quantity", 0)
            sold_symbols.add(symbol)

            # Get or generate values
            if symbol in current_prices:
                price = current_prices[symbol]
            else:
                seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
                random.seed(seed)
                price = random.uniform(50, 300)

            if symbol in cost_basis:
                basis = cost_basis[symbol]
            else:
                # Generate mock cost basis (typically slightly different from current price)
                seed = int(hashlib.md5((symbol + "basis").encode()).hexdigest()[:8], 16)
                random.seed(seed)
                basis = price * random.uniform(0.7, 1.3)

            if symbol in holding_periods:
                days_held = holding_periods[symbol]
            else:
                # Generate mock holding period
                seed = int(hashlib.md5((symbol + "days").encode()).hexdigest()[:8], 16)
                random.seed(seed)
                days_held = random.randint(30, 800)

            # Calculate gain/loss per share
            gain_per_share = price - basis
            total_gain = gain_per_share * quantity

            # Classify as short-term or long-term
            is_long_term = days_held >= 365

            if total_gain >= 0:
                if is_long_term:
                    long_term_gains += total_gain
                else:
                    short_term_gains += total_gain
            else:
                if is_long_term:
                    long_term_losses += abs(total_gain)
                else:
                    short_term_losses += abs(total_gain)

                # Identify tax-loss harvesting opportunity
                if abs(total_gain) > 500:
                    harvesting_opportunities.append(symbol)

        # Check for potential wash sales (buying after selling)
        for trade in trades:
            if trade.get("action") == "buy":
                symbol = trade.get("symbol", "")
                if symbol in sold_symbols:
                    wash_sale_warnings.append(
                        f"{symbol}: Buying after selling may trigger wash sale rule"
                    )

        # Calculate net gains
        net_gains = (short_term_gains - short_term_losses) + (long_term_gains - long_term_losses)

        # Calculate estimated tax
        taxable_short_term = max(0, short_term_gains - short_term_losses)
        taxable_long_term = max(0, long_term_gains - long_term_losses)

        estimated_tax = (
            taxable_short_term * short_term_rate + taxable_long_term * long_term_rate
        )

        # Calculate effective tax rate
        total_gains = short_term_gains + long_term_gains
        effective_rate = estimated_tax / total_gains if total_gains > 0 else 0.0

        return TaxImpactOutput(
            impact=TaxImpact(
                short_term_gains=round(short_term_gains, 2),
                long_term_gains=round(long_term_gains, 2),
                short_term_losses=round(short_term_losses, 2),
                long_term_losses=round(long_term_losses, 2),
                net_gains=round(net_gains, 2),
                estimated_tax=round(estimated_tax, 2),
                effective_tax_rate=round(effective_rate, 4),
                tax_loss_harvesting_opportunities=harvesting_opportunities,
                wash_sale_warnings=wash_sale_warnings,
            )
        )


# ============================================================================
# Execution Cost Tool
# ============================================================================


class ExecutionCostTool(BaseTool[ExecutionCostInput, ExecutionCostOutput]):
    """Tool for estimating trade execution costs.

    Calculates commissions, spread costs, slippage, and market impact.
    """

    name = "estimate_execution_cost"
    description = (
        "Estimates execution costs including commissions, bid-ask spread, slippage, "
        "and market impact. Recommends optimal execution strategy."
    )

    @property
    def input_schema(self) -> type[ExecutionCostInput]:
        return ExecutionCostInput

    @property
    def output_schema(self) -> type[ExecutionCostOutput]:
        return ExecutionCostOutput

    async def _execute_real(self, input_data: ExecutionCostInput) -> ExecutionCostOutput:
        """Calculate execution costs from provided data."""
        return await self._calculate_costs(input_data)

    async def _execute_mock(self, input_data: ExecutionCostInput) -> ExecutionCostOutput:
        """Calculate execution costs with mock data."""
        return await self._calculate_costs(input_data)

    async def _calculate_costs(
        self, input_data: ExecutionCostInput
    ) -> ExecutionCostOutput:
        """Core execution cost calculation logic.

        Args:
            input_data: Cost calculation parameters.

        Returns:
            Execution cost estimation.
        """
        trades = input_data.trades
        current_prices = input_data.current_prices or {}
        adv = input_data.average_daily_volume or {}
        spreads = input_data.bid_ask_spreads or {}
        commission_rate = input_data.commission_per_trade

        total_commission = 0.0
        total_spread_cost = 0.0
        total_slippage = 0.0
        total_impact = 0.0
        total_trade_value = 0.0

        for trade in trades:
            action = trade.get("action", "")
            if action == "hold":
                continue

            symbol = trade.get("symbol", "")
            quantity = trade.get("quantity", 0)

            if quantity == 0:
                continue

            # Get or generate price
            if symbol in current_prices:
                price = current_prices[symbol]
            else:
                seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
                random.seed(seed)
                price = random.uniform(50, 300)

            trade_value = quantity * price
            total_trade_value += trade_value

            # Commission
            if commission_rate > 0:
                total_commission += commission_rate

            # Bid-ask spread cost
            if symbol in spreads:
                spread = spreads[symbol]
            else:
                # Generate mock spread based on price (lower price = wider relative spread)
                seed = int(hashlib.md5((symbol + "spread").encode()).hexdigest()[:8], 16)
                random.seed(seed)
                spread = price * random.uniform(0.0005, 0.003)  # 0.05% to 0.3%

            spread_cost = quantity * spread / 2  # Half spread on each side
            total_spread_cost += spread_cost

            # Slippage estimation (depends on urgency)
            urgency = trade.get("urgency", "opportunistic")
            if urgency == "immediate":
                slippage_factor = 0.002  # 0.2% for immediate execution
            elif urgency == "soon":
                slippage_factor = 0.001  # 0.1%
            else:
                slippage_factor = 0.0005  # 0.05% for patient execution

            slippage = trade_value * slippage_factor
            total_slippage += slippage

            # Market impact (based on size relative to ADV)
            if symbol in adv:
                daily_volume = adv[symbol]
            else:
                seed = int(hashlib.md5((symbol + "adv").encode()).hexdigest()[:8], 16)
                random.seed(seed)
                daily_volume = int(random.uniform(500000, 10000000))

            # Impact = square root of (shares / ADV) * price
            if daily_volume > 0:
                participation = quantity / daily_volume
                impact_factor = (participation ** 0.5) * 0.1  # Simplified impact model
                impact = trade_value * min(impact_factor, 0.01)  # Cap at 1%
            else:
                impact = 0.0

            total_impact += impact

        total_cost = total_commission + total_spread_cost + total_slippage + total_impact
        cost_percent = total_cost / total_trade_value if total_trade_value > 0 else 0.0

        # Recommend execution strategy
        if cost_percent > 0.01:
            recommended = "twap"  # Time-weighted for large orders
        elif cost_percent > 0.005:
            recommended = "vwap"  # Volume-weighted for medium orders
        else:
            recommended = "market"  # Market order for small orders

        return ExecutionCostOutput(
            costs=ExecutionCosts(
                total_commission=round(total_commission, 2),
                estimated_spread_cost=round(total_spread_cost, 2),
                estimated_slippage=round(total_slippage, 2),
                market_impact=round(total_impact, 2),
                total_cost=round(total_cost, 2),
                cost_as_percent=round(cost_percent * 100, 4),
                recommended_execution=recommended,
            )
        )


# ============================================================================
# Compliance Tool
# ============================================================================


# Sector mapping for concentration limits
DEFAULT_SECTOR_MAPPING = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "WMT": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PG": "Consumer Staples",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "V": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
}


class ComplianceTool(BaseTool[ComplianceInput, ComplianceOutput]):
    """Tool for validating trade compliance.

    Checks position concentration limits, sector limits, and account-specific rules.
    """

    name = "validate_compliance"
    description = (
        "Validates proposed trades against compliance rules including position "
        "concentration limits, sector limits, wash sale rules, and account restrictions."
    )

    @property
    def input_schema(self) -> type[ComplianceInput]:
        return ComplianceInput

    @property
    def output_schema(self) -> type[ComplianceOutput]:
        return ComplianceOutput

    async def _execute_real(self, input_data: ComplianceInput) -> ComplianceOutput:
        """Validate compliance from provided data."""
        return await self._validate_compliance(input_data)

    async def _execute_mock(self, input_data: ComplianceInput) -> ComplianceOutput:
        """Validate compliance with mock data."""
        return await self._validate_compliance(input_data)

    async def _validate_compliance(
        self, input_data: ComplianceInput
    ) -> ComplianceOutput:
        """Core compliance validation logic.

        Args:
            input_data: Compliance check parameters.

        Returns:
            Compliance validation result.
        """
        trades = input_data.trades
        current_positions = input_data.current_positions or {}
        portfolio_value = input_data.portfolio_value
        account_type = input_data.account_type
        concentration_limit = input_data.concentration_limit
        sector_limit = input_data.sector_limit

        violations: list[str] = []
        warnings: list[str] = []
        concentration_limits: dict[str, float] = {}
        restricted_symbols: list[str] = []
        requires_approval = False

        # Track projected positions after trades
        projected_positions = dict(current_positions)
        sector_weights: dict[str, float] = {}

        for trade in trades:
            symbol = trade.get("symbol", "")
            action = trade.get("action", "")
            target_weight = trade.get("target_weight", 0.0)
            estimated_value = trade.get("estimated_value", 0.0)

            # Set concentration limit for this symbol
            concentration_limits[symbol] = concentration_limit

            # Update projected position
            if action == "buy":
                projected_positions[symbol] = projected_positions.get(symbol, 0) + estimated_value
            elif action == "sell":
                projected_positions[symbol] = max(0, projected_positions.get(symbol, 0) - estimated_value)

            # Check single position concentration
            if target_weight and target_weight > concentration_limit:
                violations.append(
                    f"{symbol}: Target weight {target_weight:.1%} exceeds "
                    f"{concentration_limit:.0%} concentration limit"
                )

            # Check for near-limit positions
            if target_weight and target_weight > concentration_limit * 0.8:
                warnings.append(
                    f"{symbol}: Position at {target_weight:.1%} approaching concentration limit"
                )

            # Track sector exposure
            sector = DEFAULT_SECTOR_MAPPING.get(symbol, "Other")
            sector_weights[sector] = sector_weights.get(sector, 0) + (target_weight or 0)

        # Check sector concentration
        for sector, weight in sector_weights.items():
            if weight > sector_limit:
                violations.append(
                    f"{sector} sector: Combined weight {weight:.1%} exceeds "
                    f"{sector_limit:.0%} sector limit"
                )
            elif weight > sector_limit * 0.8:
                warnings.append(
                    f"{sector} sector: Combined weight {weight:.1%} approaching limit"
                )

        # Account-specific rules
        if account_type in ["ira", "roth_ira", "401k"]:
            # Check for margin/short selling in retirement accounts
            for trade in trades:
                if trade.get("action") == "sell":
                    symbol = trade.get("symbol", "")
                    current = current_positions.get(symbol, 0)
                    sell_value = trade.get("estimated_value", 0)
                    if sell_value > current:
                        violations.append(
                            f"{symbol}: Cannot short sell in {account_type.upper()} account"
                        )

            # Day trading restrictions
            buy_count = sum(1 for t in trades if t.get("action") == "buy")
            sell_count = sum(1 for t in trades if t.get("action") == "sell")
            if buy_count >= 3 and sell_count >= 3:
                warnings.append(
                    "Multiple round-trip trades may trigger pattern day trading rules"
                )

        # Check for restricted symbols (example: company insiders)
        for trade in trades:
            symbol = trade.get("symbol", "")
            # Example restricted list (would be populated from real data)
            if symbol in ["RESTRICTED1", "RESTRICTED2"]:
                restricted_symbols.append(symbol)
                violations.append(f"{symbol}: Trading restricted due to compliance policy")

        # Large trades may require approval
        total_value = sum(t.get("estimated_value", 0) for t in trades)
        if total_value > portfolio_value * 0.20:
            requires_approval = True
            warnings.append(
                f"Total trade value ${total_value:,.0f} exceeds 20% of portfolio - "
                "may require additional approval"
            )

        is_compliant = len(violations) == 0

        return ComplianceOutput(
            result=ComplianceResult(
                is_compliant=is_compliant,
                violations=violations,
                warnings=warnings,
                concentration_limits=concentration_limits,
                restricted_symbols=restricted_symbols,
                requires_approval=requires_approval,
            )
        )
