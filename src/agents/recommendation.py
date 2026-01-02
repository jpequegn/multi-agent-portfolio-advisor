"""Recommendation Agent for generating portfolio trade recommendations.

This module implements the RecommendationAgent that generates actionable
portfolio recommendations based on analysis from the AnalysisAgent.
"""

from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from src.agents.base import AgentState, BaseAgent
from src.observability.tracing import traced_agent
from src.tools.base import BaseTool, ToolInput, ToolOutput, ToolRegistry

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


class TaxImpact(BaseModel):
    """Estimated tax impact of trades."""

    short_term_gains: float = Field(default=0.0, description="Short-term capital gains")
    long_term_gains: float = Field(default=0.0, description="Long-term capital gains")
    estimated_tax: float = Field(default=0.0, description="Estimated tax liability")
    tax_loss_harvesting_opportunities: list[str] = Field(
        default_factory=list, description="Symbols with harvesting opportunities"
    )


class ExecutionCosts(BaseModel):
    """Estimated execution costs."""

    total_commission: float = Field(default=0.0, description="Total commission cost")
    estimated_slippage: float = Field(default=0.0, description="Estimated slippage")
    market_impact: float = Field(default=0.0, description="Estimated market impact")
    total_cost: float = Field(default=0.0, description="Total execution cost")


class ComplianceResult(BaseModel):
    """Compliance validation result."""

    is_compliant: bool = Field(default=True, description="Whether trades are compliant")
    violations: list[str] = Field(default_factory=list, description="Compliance violations")
    warnings: list[str] = Field(default_factory=list, description="Compliance warnings")
    concentration_limits: dict[str, float] = Field(
        default_factory=dict, description="Position concentration limits"
    )


class RecommendationOutput(BaseModel):
    """Structured output from the Recommendation Agent.

    Contains trade recommendations with cost and compliance analysis.
    """

    trades: list[Trade] = Field(default_factory=list)
    tax_impact: TaxImpact = Field(default_factory=lambda: TaxImpact())
    execution_costs: ExecutionCosts = Field(default_factory=lambda: ExecutionCosts())
    compliance: ComplianceResult = Field(default_factory=lambda: ComplianceResult())
    summary: str = ""
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    errors: list[str] = Field(default_factory=list)


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
    risk_metrics: dict[str, Any] | None = Field(
        default=None, description="Risk metrics from analysis"
    )
    recommendations: list[str] | None = Field(
        default=None, description="Recommendations from analysis"
    )


class RebalancingOutput(ToolOutput):
    """Output from rebalancing tool."""

    trades: list[Trade] = Field(default_factory=list)
    rebalance_summary: str = ""


class TaxImpactInput(ToolInput):
    """Input for tax impact estimation."""

    trades: list[dict[str, Any]] = Field(..., description="Proposed trades")
    holding_periods: dict[str, int] | None = Field(
        default=None, description="Days held for each symbol"
    )
    cost_basis: dict[str, float] | None = Field(
        default=None, description="Cost basis for each symbol"
    )


class TaxImpactOutput(ToolOutput):
    """Output from tax impact estimation."""

    impact: TaxImpact = Field(default_factory=lambda: TaxImpact())


class ExecutionCostInput(ToolInput):
    """Input for execution cost estimation."""

    trades: list[dict[str, Any]] = Field(..., description="Proposed trades")
    average_daily_volume: dict[str, int] | None = Field(
        default=None, description="ADV for each symbol"
    )


class ExecutionCostOutput(ToolOutput):
    """Output from execution cost estimation."""

    costs: ExecutionCosts = Field(default_factory=lambda: ExecutionCosts())


class ComplianceInput(ToolInput):
    """Input for compliance validation."""

    trades: list[dict[str, Any]] = Field(..., description="Proposed trades")
    current_positions: dict[str, float] | None = Field(
        default=None, description="Current position sizes"
    )
    account_type: str = Field(default="taxable", description="Account type")


class ComplianceOutput(ToolOutput):
    """Output from compliance validation."""

    result: ComplianceResult = Field(default_factory=lambda: ComplianceResult())


# ============================================================================
# Placeholder Tools
# ============================================================================


class PlaceholderRebalancingTool(BaseTool[RebalancingInput, RebalancingOutput]):
    """Placeholder tool for generating rebalancing trades.

    Generates mock trade recommendations based on symbols and risk analysis.
    """

    name = "generate_rebalancing"
    description = (
        "Generates rebalancing trade recommendations based on current portfolio "
        "composition and target allocation. Returns a list of trades to execute."
    )

    @property
    def input_schema(self) -> type[RebalancingInput]:
        return RebalancingInput

    @property
    def output_schema(self) -> type[RebalancingOutput]:
        return RebalancingOutput

    async def _execute_real(self, input_data: RebalancingInput) -> RebalancingOutput:
        """Real implementation would connect to portfolio optimization service."""
        raise NotImplementedError("Real rebalancing not yet implemented")

    async def _execute_mock(self, input_data: RebalancingInput) -> RebalancingOutput:
        """Generate mock rebalancing trades."""
        import hashlib
        import random

        trades: list[Trade] = []
        symbols = input_data.symbols
        recommendations = input_data.recommendations or []
        risk_metrics = input_data.risk_metrics or {}

        # Generate deterministic but varied trades based on symbols
        for symbol in symbols:
            # Use hash for deterministic randomness
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            random.seed(seed)

            # Determine action based on various factors
            action: Literal["buy", "sell", "hold"]
            rationale = ""
            priority = 3
            urgency: Literal["immediate", "soon", "opportunistic"] = "opportunistic"

            # Check if recommendations mention this symbol
            symbol_mentioned = any(symbol in rec for rec in recommendations)

            # High volatility symbols get reduced
            volatility = risk_metrics.get("volatility", 0.15)
            if volatility > 0.25 and random.random() > 0.5:
                action = "sell"
                rationale = f"Reduce position due to high volatility ({volatility:.1%})"
                priority = 2
                urgency = "soon"
            # Random variation for demo
            elif random.random() > 0.6:
                action = "buy"
                rationale = f"Increase allocation to {symbol} for diversification"
                priority = 3
            elif random.random() > 0.5:
                action = "sell"
                rationale = f"Take profits on {symbol} position"
                priority = 4
            else:
                action = "hold"
                rationale = f"Maintain current {symbol} position"
                priority = 5

            if symbol_mentioned:
                priority = min(priority, 2)
                urgency = "soon"
                rationale = f"Action recommended by analysis: {rationale}"

            current_weight = round(random.uniform(0.05, 0.25), 3)
            target_weight = current_weight
            quantity = 0

            if action == "buy":
                target_weight = min(current_weight + random.uniform(0.02, 0.08), 0.30)
                quantity = random.randint(10, 100)
            elif action == "sell":
                target_weight = max(current_weight - random.uniform(0.02, 0.08), 0.02)
                quantity = random.randint(5, 50)

            trades.append(
                Trade(
                    action=action,
                    symbol=symbol,
                    quantity=quantity,
                    current_weight=current_weight,
                    target_weight=round(target_weight, 3),
                    rationale=rationale,
                    priority=priority,
                    urgency=urgency,
                )
            )

        # Sort by priority
        trades.sort(key=lambda t: t.priority)

        # Generate summary
        buy_count = sum(1 for t in trades if t.action == "buy")
        sell_count = sum(1 for t in trades if t.action == "sell")
        hold_count = sum(1 for t in trades if t.action == "hold")

        summary = (
            f"Generated {len(trades)} trade recommendations: "
            f"{buy_count} buys, {sell_count} sells, {hold_count} holds"
        )

        return RebalancingOutput(trades=trades, rebalance_summary=summary)


class PlaceholderTaxImpactTool(BaseTool[TaxImpactInput, TaxImpactOutput]):
    """Placeholder tool for estimating tax impact.

    Generates mock tax impact estimates for proposed trades.
    """

    name = "estimate_tax_impact"
    description = (
        "Estimates the tax impact of proposed trades including short-term and "
        "long-term capital gains, and identifies tax loss harvesting opportunities."
    )

    @property
    def input_schema(self) -> type[TaxImpactInput]:
        return TaxImpactInput

    @property
    def output_schema(self) -> type[TaxImpactOutput]:
        return TaxImpactOutput

    async def _execute_real(self, input_data: TaxImpactInput) -> TaxImpactOutput:
        """Real implementation would connect to tax calculation service."""
        raise NotImplementedError("Real tax impact estimation not yet implemented")

    async def _execute_mock(self, input_data: TaxImpactInput) -> TaxImpactOutput:
        """Generate mock tax impact estimates."""
        import hashlib
        import random

        trades = input_data.trades
        holding_periods = input_data.holding_periods or {}

        # Generate deterministic values based on trades
        trade_str = str(sorted([t.get("symbol", "") for t in trades]))
        seed = int(hashlib.md5(trade_str.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Calculate mock gains
        sell_trades = [t for t in trades if t.get("action") == "sell"]

        short_term_gains = 0.0
        long_term_gains = 0.0
        harvesting_opportunities: list[str] = []

        for trade in sell_trades:
            symbol = trade.get("symbol", "")
            days_held = holding_periods.get(symbol, random.randint(30, 500))

            # Mock gain/loss calculation
            gain = random.uniform(-500, 1500)

            if days_held < 365:
                short_term_gains += gain
            else:
                long_term_gains += gain

            # Identify harvesting opportunities (losses)
            if gain < 0 and symbol not in harvesting_opportunities:
                harvesting_opportunities.append(symbol)

        # Estimate tax (simplified)
        short_term_tax_rate = 0.32  # Assume high bracket
        long_term_tax_rate = 0.15
        estimated_tax = max(0, short_term_gains * short_term_tax_rate) + max(
            0, long_term_gains * long_term_tax_rate
        )

        return TaxImpactOutput(
            impact=TaxImpact(
                short_term_gains=round(short_term_gains, 2),
                long_term_gains=round(long_term_gains, 2),
                estimated_tax=round(estimated_tax, 2),
                tax_loss_harvesting_opportunities=harvesting_opportunities,
            )
        )


class PlaceholderExecutionCostTool(BaseTool[ExecutionCostInput, ExecutionCostOutput]):
    """Placeholder tool for estimating execution costs.

    Generates mock execution cost estimates including commissions and slippage.
    """

    name = "estimate_execution_cost"
    description = (
        "Estimates execution costs for proposed trades including commissions, "
        "slippage, and market impact based on order sizes and market conditions."
    )

    @property
    def input_schema(self) -> type[ExecutionCostInput]:
        return ExecutionCostInput

    @property
    def output_schema(self) -> type[ExecutionCostOutput]:
        return ExecutionCostOutput

    async def _execute_real(self, input_data: ExecutionCostInput) -> ExecutionCostOutput:
        """Real implementation would connect to execution analysis service."""
        raise NotImplementedError("Real execution cost estimation not yet implemented")

    async def _execute_mock(self, input_data: ExecutionCostInput) -> ExecutionCostOutput:
        """Generate mock execution cost estimates."""
        import hashlib
        import random

        trades = input_data.trades

        # Generate deterministic values
        trade_str = str(sorted([t.get("symbol", "") for t in trades]))
        seed = int(hashlib.md5(trade_str.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Calculate costs per trade
        total_commission = 0.0
        total_slippage = 0.0
        total_impact = 0.0

        for trade in trades:
            quantity = trade.get("quantity", 0)
            if quantity == 0:
                continue

            # Commission: flat fee per trade
            commission = 4.95  # Flat commission
            total_commission += commission

            # Slippage: based on quantity
            slippage = quantity * random.uniform(0.01, 0.05)
            total_slippage += slippage

            # Market impact: larger orders have more impact
            impact = quantity * random.uniform(0.005, 0.02)
            total_impact += impact

        total_cost = total_commission + total_slippage + total_impact

        return ExecutionCostOutput(
            costs=ExecutionCosts(
                total_commission=round(total_commission, 2),
                estimated_slippage=round(total_slippage, 2),
                market_impact=round(total_impact, 2),
                total_cost=round(total_cost, 2),
            )
        )


class PlaceholderComplianceTool(BaseTool[ComplianceInput, ComplianceOutput]):
    """Placeholder tool for compliance validation.

    Validates proposed trades against compliance rules and concentration limits.
    """

    name = "validate_compliance"
    description = (
        "Validates proposed trades against compliance rules including position "
        "concentration limits, restricted securities, and account-specific rules."
    )

    @property
    def input_schema(self) -> type[ComplianceInput]:
        return ComplianceInput

    @property
    def output_schema(self) -> type[ComplianceOutput]:
        return ComplianceOutput

    async def _execute_real(self, input_data: ComplianceInput) -> ComplianceOutput:
        """Real implementation would connect to compliance service."""
        raise NotImplementedError("Real compliance validation not yet implemented")

    async def _execute_mock(self, input_data: ComplianceInput) -> ComplianceOutput:
        """Generate mock compliance validation results."""
        import hashlib
        import random

        trades = input_data.trades
        current_positions = input_data.current_positions or {}
        account_type = input_data.account_type

        # Generate deterministic results
        trade_str = str(sorted([t.get("symbol", "") for t in trades]))
        seed = int(hashlib.md5(trade_str.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        violations: list[str] = []
        warnings: list[str] = []
        concentration_limits: dict[str, float] = {}

        # Check each trade
        for trade in trades:
            symbol = trade.get("symbol", "")
            action = trade.get("action", "")
            target_weight = trade.get("target_weight", 0.0)

            # Set concentration limit
            concentration_limits[symbol] = 0.25  # 25% max

            # Check concentration
            if target_weight and target_weight > 0.25:
                violations.append(
                    f"{symbol}: Target weight {target_weight:.1%} exceeds 25% concentration limit"
                )

            # Random warnings for demo
            if random.random() > 0.8:
                warnings.append(f"{symbol}: Near concentration limit, monitor closely")

            # Check for wash sale risk on sells followed by buys
            if action == "sell" and symbol in current_positions:
                warnings.append(
                    f"{symbol}: Potential wash sale risk if repurchased within 30 days"
                )

        # Account-specific rules
        if account_type == "ira":
            for trade in trades:
                if trade.get("action") == "sell" and trade.get("symbol", "").startswith("OPT"):
                    warnings.append("Options trades in IRA require additional approval")

        is_compliant = len(violations) == 0

        return ComplianceOutput(
            result=ComplianceResult(
                is_compliant=is_compliant,
                violations=violations,
                warnings=warnings,
                concentration_limits=concentration_limits,
            )
        )


# ============================================================================
# Recommendation Agent
# ============================================================================


class RecommendationAgent(BaseAgent):
    """Agent for generating actionable portfolio recommendations.

    Consumes analysis output from the AnalysisAgent and generates
    trade recommendations with tax, cost, and compliance considerations.

    Tools:
        - generate_rebalancing: Generates rebalancing trades
        - estimate_tax_impact: Estimates tax implications
        - estimate_execution_cost: Estimates execution costs
        - validate_compliance: Validates compliance rules

    Output:
        RecommendationOutput containing trades and cost analysis
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        use_mock_tools: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Recommendation Agent.

        Args:
            tool_registry: Optional custom tool registry. If not provided,
                creates a new one with placeholder tools.
            use_mock_tools: Whether to use mock mode for tools.
            **kwargs: Additional arguments passed to BaseAgent.
        """
        super().__init__(**kwargs)

        # Set up tool registry
        if tool_registry is not None:
            self._tool_registry = tool_registry
        else:
            self._tool_registry = ToolRegistry()
            self._tool_registry.register(
                PlaceholderRebalancingTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderTaxImpactTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderExecutionCostTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderComplianceTool(use_mock=use_mock_tools)
            )

    @property
    def name(self) -> str:
        return "recommendation"

    @property
    def description(self) -> str:
        return "Generates actionable portfolio recommendations with trade list"

    @property
    def system_prompt(self) -> str:
        """System prompt for the Recommendation Agent."""
        return """You are a portfolio recommendation agent specializing in generating
actionable trade recommendations.

Your role is to:
1. Analyze portfolio analysis results from the previous agent
2. Generate specific trade recommendations (buy, sell, hold)
3. Consider tax implications of trades
4. Estimate execution costs
5. Ensure compliance with investment guidelines

Available tools:
- generate_rebalancing: Generate rebalancing trades based on analysis
- estimate_tax_impact: Estimate tax implications of proposed trades
- estimate_execution_cost: Estimate execution costs and market impact
- validate_compliance: Check trades against compliance rules

When generating recommendations:
- Prioritize risk reduction when volatility is high
- Consider tax efficiency (prefer long-term gains, harvest losses)
- Minimize execution costs by using appropriate order types
- Ensure all trades are compliant with guidelines
- Provide clear rationale for each recommendation

Output should include:
- List of trades with action, symbol, quantity, and rationale
- Tax impact summary
- Execution cost estimate
- Compliance status
- Overall recommendation summary"""

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return tools in Anthropic format."""
        return self._tool_registry.to_anthropic_tools()

    def get_tool(self, name: str) -> BaseTool[Any, Any]:
        """Get a tool by name from the registry.

        Args:
            name: Name of the tool.

        Returns:
            The tool instance.
        """
        return self._tool_registry.get(name)

    @traced_agent("recommendation_agent")
    async def invoke(self, state: AgentState) -> AgentState:
        """Execute the recommendation agent workflow.

        Args:
            state: Current workflow state containing:
                - context.analysis: Analysis data from AnalysisAgent
                - context.symbols: List of symbols to consider

        Returns:
            Updated state with recommendations in context.recommendations
        """
        self._logger.info("recommendation_invoke_start", context_keys=list(state.context.keys()))

        # Extract data from state
        symbols: list[str] = state.context.get("symbols", [])
        analysis_data = state.context.get("analysis", {})

        if not symbols:
            # Try to get symbols from analysis data
            symbols = list(analysis_data.get("risk_metrics", {}).keys()) or []

        if not symbols:
            self._logger.warning("no_symbols_for_recommendations")
            state.errors.append("RecommendationAgent: No symbols provided for recommendations")
            return state

        # Initialize output
        output = RecommendationOutput()

        # Extract analysis insights
        risk_metrics = analysis_data.get("risk_metrics", {})
        recommendations_from_analysis = analysis_data.get("recommendations", [])

        # Generate rebalancing trades
        rebalancing_tool = self.get_tool("generate_rebalancing")
        try:
            rebal_result = await rebalancing_tool.execute({
                "symbols": symbols,
                "risk_metrics": risk_metrics,
                "recommendations": recommendations_from_analysis,
            })
            output.trades = rebal_result.trades
            self._logger.debug("rebalancing_generated", trade_count=len(output.trades))
        except Exception as e:
            error_msg = f"Failed to generate rebalancing: {e}"
            output.errors.append(error_msg)
            self._logger.warning("rebalancing_failed", error=str(e))

        # Convert trades to dict for other tools
        trades_dict = [t.model_dump() for t in output.trades]

        # Estimate tax impact
        tax_tool = self.get_tool("estimate_tax_impact")
        try:
            tax_result = await tax_tool.execute({"trades": trades_dict})
            output.tax_impact = tax_result.impact
            self._logger.debug("tax_impact_estimated")
        except Exception as e:
            error_msg = f"Failed to estimate tax impact: {e}"
            output.errors.append(error_msg)
            self._logger.warning("tax_impact_failed", error=str(e))

        # Estimate execution costs
        cost_tool = self.get_tool("estimate_execution_cost")
        try:
            cost_result = await cost_tool.execute({"trades": trades_dict})
            output.execution_costs = cost_result.costs
            self._logger.debug("execution_costs_estimated")
        except Exception as e:
            error_msg = f"Failed to estimate execution costs: {e}"
            output.errors.append(error_msg)
            self._logger.warning("execution_costs_failed", error=str(e))

        # Validate compliance
        compliance_tool = self.get_tool("validate_compliance")
        try:
            compliance_result = await compliance_tool.execute({"trades": trades_dict})
            output.compliance = compliance_result.result
            self._logger.debug("compliance_validated")
        except Exception as e:
            error_msg = f"Failed to validate compliance: {e}"
            output.errors.append(error_msg)
            self._logger.warning("compliance_validation_failed", error=str(e))

        # Calculate trade counts
        output.total_trades = len(output.trades)
        output.buy_count = sum(1 for t in output.trades if t.action == "buy")
        output.sell_count = sum(1 for t in output.trades if t.action == "sell")
        output.hold_count = sum(1 for t in output.trades if t.action == "hold")

        # Generate summary
        output.summary = self._generate_summary(output)

        # Update state with recommendations
        state.context["recommendations"] = output.model_dump()
        state.messages.append({
            "role": "assistant",
            "content": f"Generated {output.total_trades} trade recommendations: "
                       f"{output.buy_count} buys, {output.sell_count} sells, {output.hold_count} holds. "
                       f"Compliance: {'Passed' if output.compliance.is_compliant else 'Failed'}.",
        })

        self._logger.info(
            "recommendation_invoke_complete",
            total_trades=output.total_trades,
            buy_count=output.buy_count,
            sell_count=output.sell_count,
            hold_count=output.hold_count,
            is_compliant=output.compliance.is_compliant,
            error_count=len(output.errors),
        )

        return state

    def _generate_summary(self, output: RecommendationOutput) -> str:
        """Generate a summary of recommendations.

        Args:
            output: The recommendation output.

        Returns:
            Summary string.
        """
        parts = []

        # Trade summary
        if output.trades:
            high_priority = [t for t in output.trades if t.priority <= 2]
            if high_priority:
                parts.append(
                    f"{len(high_priority)} high-priority trades requiring attention"
                )

            immediate = [t for t in output.trades if t.urgency == "immediate"]
            if immediate:
                symbols = ", ".join(t.symbol for t in immediate)
                parts.append(f"Immediate action recommended for: {symbols}")

        # Tax considerations
        if output.tax_impact.estimated_tax > 0:
            parts.append(
                f"Estimated tax impact: ${output.tax_impact.estimated_tax:,.2f}"
            )

        if output.tax_impact.tax_loss_harvesting_opportunities:
            parts.append(
                f"Tax loss harvesting available: "
                f"{', '.join(output.tax_impact.tax_loss_harvesting_opportunities)}"
            )

        # Execution costs
        if output.execution_costs.total_cost > 0:
            parts.append(
                f"Estimated execution cost: ${output.execution_costs.total_cost:,.2f}"
            )

        # Compliance
        if not output.compliance.is_compliant:
            parts.append(
                f"COMPLIANCE ALERT: {len(output.compliance.violations)} violation(s)"
            )
        elif output.compliance.warnings:
            parts.append(f"{len(output.compliance.warnings)} compliance warning(s)")

        # Errors
        if output.errors:
            parts.append(f"Encountered {len(output.errors)} error(s)")

        return ". ".join(parts) if parts else "Recommendations generated successfully."
