"""Analysis Agent for portfolio risk and performance analysis.

This module implements the AnalysisAgent that analyzes portfolio data
from the ResearchAgent and produces risk metrics and recommendations.
"""

from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import AgentState, BaseAgent
from src.observability.tracing import traced_agent
from src.tools.base import BaseTool, ToolInput, ToolOutput, ToolRegistry

logger = structlog.get_logger(__name__)


# ============================================================================
# Output Schemas
# ============================================================================


class RiskMetrics(BaseModel):
    """Portfolio risk metrics."""

    var_95: float | None = Field(default=None, description="Value at Risk at 95% confidence")
    var_99: float | None = Field(default=None, description="Value at Risk at 99% confidence")
    sharpe_ratio: float | None = Field(default=None, description="Risk-adjusted return metric")
    sortino_ratio: float | None = Field(default=None, description="Downside risk-adjusted return")
    beta: float | None = Field(default=None, description="Market sensitivity")
    alpha: float | None = Field(default=None, description="Excess return over benchmark")
    max_drawdown: float | None = Field(default=None, description="Maximum peak-to-trough decline")
    volatility: float | None = Field(default=None, description="Annualized standard deviation")


class CorrelationResult(BaseModel):
    """Correlation analysis results."""

    matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    highly_correlated_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    diversification_score: float | None = None


class BenchmarkComparison(BaseModel):
    """Benchmark comparison results."""

    benchmark_name: str = "S&P 500"
    portfolio_return: float | None = None
    benchmark_return: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None
    relative_performance: float | None = None


class AttributionResult(BaseModel):
    """Performance attribution results."""

    sector_attribution: dict[str, float] = Field(default_factory=dict)
    factor_attribution: dict[str, float] = Field(default_factory=dict)
    selection_effect: float | None = None
    allocation_effect: float | None = None
    interaction_effect: float | None = None


class AnalysisOutput(BaseModel):
    """Structured output from the Analysis Agent.

    Contains all risk metrics, correlations, and analysis results.
    """

    risk_metrics: RiskMetrics = Field(default_factory=lambda: RiskMetrics())
    correlations: CorrelationResult = Field(default_factory=lambda: CorrelationResult())
    benchmark_comparison: BenchmarkComparison = Field(default_factory=lambda: BenchmarkComparison())
    attribution: AttributionResult = Field(default_factory=lambda: AttributionResult())
    risk_summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ============================================================================
# Tool Input/Output Schemas
# ============================================================================


class RiskMetricsInput(ToolInput):
    """Input for risk metrics calculation."""

    symbols: list[str] = Field(..., description="List of symbols in the portfolio")
    weights: list[float] | None = Field(None, description="Portfolio weights")
    period_days: int = Field(default=252, description="Historical period in trading days")


class RiskMetricsOutput(ToolOutput):
    """Output from risk metrics calculation."""

    metrics: RiskMetrics = Field(default_factory=lambda: RiskMetrics())


class CorrelationInput(ToolInput):
    """Input for correlation analysis."""

    symbols: list[str] = Field(..., description="Symbols to analyze")
    period_days: int = Field(default=252, description="Historical period")


class CorrelationOutput(ToolOutput):
    """Output from correlation analysis."""

    result: CorrelationResult = Field(default_factory=CorrelationResult)


class BenchmarkInput(ToolInput):
    """Input for benchmark comparison."""

    symbols: list[str] = Field(..., description="Portfolio symbols")
    weights: list[float] | None = Field(None, description="Portfolio weights")
    benchmark: str = Field(default="SPY", description="Benchmark symbol")


class BenchmarkOutput(ToolOutput):
    """Output from benchmark comparison."""

    result: BenchmarkComparison = Field(default_factory=BenchmarkComparison)


class AttributionInput(ToolInput):
    """Input for performance attribution."""

    symbols: list[str] = Field(..., description="Portfolio symbols")
    weights: list[float] | None = Field(None, description="Portfolio weights")


class AttributionOutput(ToolOutput):
    """Output from performance attribution."""

    result: AttributionResult = Field(default_factory=AttributionResult)


# ============================================================================
# Placeholder Tools
# ============================================================================


class PlaceholderRiskMetricsTool(BaseTool[RiskMetricsInput, RiskMetricsOutput]):
    """Placeholder risk metrics tool for testing.

    Will be replaced by actual implementation in issue #16.
    """

    name = "calculate_risk_metrics"
    description = "Calculates portfolio risk metrics including VaR, Sharpe ratio, and volatility"

    @property
    def input_schema(self) -> type[RiskMetricsInput]:
        return RiskMetricsInput

    @property
    def output_schema(self) -> type[RiskMetricsOutput]:
        return RiskMetricsOutput

    async def _execute_real(self, input_data: RiskMetricsInput) -> RiskMetricsOutput:
        """Real implementation - placeholder raises to trigger mock fallback."""
        raise NotImplementedError("Real risk metrics calculation not yet implemented")

    async def _execute_mock(self, input_data: RiskMetricsInput) -> RiskMetricsOutput:
        """Mock implementation with sample data."""
        # Generate mock metrics based on number of symbols
        num_symbols = len(input_data.symbols)
        diversification_factor = min(1.0, num_symbols / 10)

        return RiskMetricsOutput(
            metrics=RiskMetrics(
                var_95=0.02 * (1 - diversification_factor * 0.3),
                var_99=0.035 * (1 - diversification_factor * 0.3),
                sharpe_ratio=1.2 + diversification_factor * 0.3,
                sortino_ratio=1.5 + diversification_factor * 0.3,
                beta=0.95 + (num_symbols % 3) * 0.05,
                alpha=0.02,
                max_drawdown=0.15,
                volatility=0.18 * (1 - diversification_factor * 0.2),
            )
        )


class PlaceholderCorrelationTool(BaseTool[CorrelationInput, CorrelationOutput]):
    """Placeholder correlation analysis tool for testing.

    Will be replaced by actual implementation in issue #16.
    """

    name = "correlation_analysis"
    description = "Analyzes correlations between portfolio holdings"

    @property
    def input_schema(self) -> type[CorrelationInput]:
        return CorrelationInput

    @property
    def output_schema(self) -> type[CorrelationOutput]:
        return CorrelationOutput

    async def _execute_real(self, input_data: CorrelationInput) -> CorrelationOutput:
        """Real implementation - placeholder raises to trigger mock fallback."""
        raise NotImplementedError("Real correlation analysis not yet implemented")

    async def _execute_mock(self, input_data: CorrelationInput) -> CorrelationOutput:
        """Mock implementation with sample data."""
        symbols = input_data.symbols
        matrix: dict[str, dict[str, float]] = {}
        highly_correlated: list[tuple[str, str, float]] = []

        # Generate mock correlation matrix
        for i, sym1 in enumerate(symbols):
            matrix[sym1] = {}
            for j, sym2 in enumerate(symbols):
                if i == j:
                    matrix[sym1][sym2] = 1.0
                elif i < j:
                    # Mock correlation between 0.3 and 0.8
                    corr = 0.3 + ((i + j) % 5) * 0.1
                    matrix[sym1][sym2] = round(corr, 2)
                    if corr > 0.7:
                        highly_correlated.append((sym1, sym2, corr))
                else:
                    matrix[sym1][sym2] = matrix[sym2][sym1]

        # Diversification score: lower average correlation = higher score
        all_corrs = [
            matrix[s1][s2] for s1 in symbols for s2 in symbols if s1 != s2
        ]
        avg_corr = sum(all_corrs) / len(all_corrs) if all_corrs else 0
        div_score = round(1 - avg_corr, 2)

        return CorrelationOutput(
            result=CorrelationResult(
                matrix=matrix,
                highly_correlated_pairs=highly_correlated,
                diversification_score=div_score,
            )
        )


class PlaceholderBenchmarkTool(BaseTool[BenchmarkInput, BenchmarkOutput]):
    """Placeholder benchmark comparison tool for testing.

    Will be replaced by actual implementation in issue #16.
    """

    name = "benchmark_comparison"
    description = "Compares portfolio performance against a benchmark"

    @property
    def input_schema(self) -> type[BenchmarkInput]:
        return BenchmarkInput

    @property
    def output_schema(self) -> type[BenchmarkOutput]:
        return BenchmarkOutput

    async def _execute_real(self, input_data: BenchmarkInput) -> BenchmarkOutput:
        """Real implementation - placeholder raises to trigger mock fallback."""
        raise NotImplementedError("Real benchmark comparison not yet implemented")

    async def _execute_mock(self, input_data: BenchmarkInput) -> BenchmarkOutput:
        """Mock implementation with sample data."""
        return BenchmarkOutput(
            result=BenchmarkComparison(
                benchmark_name=f"{input_data.benchmark} Index",
                portfolio_return=0.12,
                benchmark_return=0.10,
                tracking_error=0.05,
                information_ratio=0.4,
                relative_performance=0.02,
            )
        )


class PlaceholderAttributionTool(BaseTool[AttributionInput, AttributionOutput]):
    """Placeholder performance attribution tool for testing.

    Will be replaced by actual implementation in issue #16.
    """

    name = "attribution_analysis"
    description = "Analyzes sources of portfolio performance"

    @property
    def input_schema(self) -> type[AttributionInput]:
        return AttributionInput

    @property
    def output_schema(self) -> type[AttributionOutput]:
        return AttributionOutput

    async def _execute_real(self, input_data: AttributionInput) -> AttributionOutput:
        """Real implementation - placeholder raises to trigger mock fallback."""
        raise NotImplementedError("Real attribution analysis not yet implemented")

    async def _execute_mock(self, input_data: AttributionInput) -> AttributionOutput:
        """Mock implementation with sample data."""
        # Generate mock sector attribution based on symbols
        sector_map = {
            "AAPL": "Technology",
            "GOOGL": "Technology",
            "MSFT": "Technology",
            "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
            "JPM": "Financials",
            "BAC": "Financials",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "XOM": "Energy",
        }

        sector_attr: dict[str, float] = {}
        for symbol in input_data.symbols:
            sector = sector_map.get(symbol.upper(), "Other")
            sector_attr[sector] = sector_attr.get(sector, 0) + 0.01

        return AttributionOutput(
            result=AttributionResult(
                sector_attribution=sector_attr,
                factor_attribution={
                    "Market": 0.08,
                    "Size": 0.01,
                    "Value": -0.005,
                    "Momentum": 0.015,
                },
                selection_effect=0.015,
                allocation_effect=0.008,
                interaction_effect=0.002,
            )
        )


# ============================================================================
# Analysis Agent
# ============================================================================


class AnalysisAgent(BaseAgent):
    """Agent that analyzes portfolio risk and performance.

    The Analysis Agent is the second step in the portfolio analysis workflow.
    It consumes data from the ResearchAgent and produces comprehensive
    risk analysis and recommendations.

    Tools:
        - calculate_risk_metrics: Calculates VaR, Sharpe, volatility
        - correlation_analysis: Analyzes correlations between holdings
        - benchmark_comparison: Compares performance to benchmark
        - attribution_analysis: Analyzes performance attribution

    Output:
        AnalysisOutput containing risk metrics, correlations, and recommendations
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        use_mock_tools: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Analysis Agent.

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
                PlaceholderRiskMetricsTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderCorrelationTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderBenchmarkTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderAttributionTool(use_mock=use_mock_tools)
            )

    @property
    def name(self) -> str:
        return "analysis"

    @property
    def description(self) -> str:
        return "Analyzes portfolio risk and performance metrics"

    @property
    def system_prompt(self) -> str:
        return """You are a quantitative analysis agent specialized in portfolio risk assessment.

Your role is to:
1. Calculate comprehensive risk metrics for the portfolio
2. Analyze correlations between holdings
3. Compare performance against benchmarks
4. Perform attribution analysis to understand return sources

Available tools:
- calculate_risk_metrics: Calculates VaR, Sharpe ratio, beta, volatility, and other risk metrics
- correlation_analysis: Analyzes correlations between portfolio holdings
- benchmark_comparison: Compares portfolio performance against a benchmark index
- attribution_analysis: Breaks down performance by sector and factor exposures

Guidelines:
- Use all available tools to provide comprehensive analysis
- Identify potential risk concentrations
- Flag highly correlated positions that may reduce diversification
- Compare performance objectively against benchmark
- Provide actionable recommendations based on findings

Output your findings in a structured format with:
- Complete risk metrics
- Correlation analysis with diversification assessment
- Benchmark comparison
- Performance attribution
- Risk summary with key findings
- Specific recommendations for improvement"""

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

    @traced_agent("analysis_agent")
    async def invoke(self, state: AgentState) -> AgentState:
        """Execute the analysis agent workflow.

        Args:
            state: Current workflow state containing:
                - context.research: Research data from ResearchAgent
                - context.symbols: List of symbols to analyze

        Returns:
            Updated state with analysis results in context.analysis
        """
        self._logger.info("analysis_invoke_start", context_keys=list(state.context.keys()))

        # Extract symbols and research data from state
        symbols: list[str] = state.context.get("symbols", [])
        research_data = state.context.get("research", {})

        if not symbols:
            # Try to get symbols from research data
            symbols = research_data.get("symbols_analyzed", [])

        if not symbols:
            self._logger.warning("no_symbols_for_analysis")
            state.errors.append("AnalysisAgent: No symbols provided for analysis")
            return state

        # Initialize output
        output = AnalysisOutput()

        # Calculate risk metrics
        risk_tool = self.get_tool("calculate_risk_metrics")
        try:
            risk_result = await risk_tool.execute({"symbols": symbols})
            output.risk_metrics = risk_result.metrics
            self._logger.debug("risk_metrics_calculated")
        except Exception as e:
            error_msg = f"Failed to calculate risk metrics: {e}"
            output.errors.append(error_msg)
            self._logger.warning("risk_metrics_failed", error=str(e))

        # Analyze correlations
        correlation_tool = self.get_tool("correlation_analysis")
        try:
            corr_result = await correlation_tool.execute({"symbols": symbols})
            output.correlations = corr_result.result
            self._logger.debug("correlations_calculated")
        except Exception as e:
            error_msg = f"Failed to analyze correlations: {e}"
            output.errors.append(error_msg)
            self._logger.warning("correlation_analysis_failed", error=str(e))

        # Compare against benchmark
        benchmark_tool = self.get_tool("benchmark_comparison")
        try:
            bench_result = await benchmark_tool.execute({"symbols": symbols})
            output.benchmark_comparison = bench_result.result
            self._logger.debug("benchmark_compared")
        except Exception as e:
            error_msg = f"Failed to compare benchmark: {e}"
            output.errors.append(error_msg)
            self._logger.warning("benchmark_comparison_failed", error=str(e))

        # Perform attribution analysis
        attribution_tool = self.get_tool("attribution_analysis")
        try:
            attr_result = await attribution_tool.execute({"symbols": symbols})
            output.attribution = attr_result.result
            self._logger.debug("attribution_analyzed")
        except Exception as e:
            error_msg = f"Failed to analyze attribution: {e}"
            output.errors.append(error_msg)
            self._logger.warning("attribution_analysis_failed", error=str(e))

        # Generate summary and recommendations
        output.risk_summary = self._generate_risk_summary(output)
        output.recommendations = self._generate_recommendations(output, symbols)
        output.warnings = self._generate_warnings(output)

        # Update state with analysis results
        state.context["analysis"] = output.model_dump()
        state.messages.append({
            "role": "assistant",
            "content": f"Analysis completed for {len(symbols)} symbols. "
                       f"Risk metrics calculated. {len(output.recommendations)} recommendations generated.",
        })

        self._logger.info(
            "analysis_invoke_complete",
            symbols_count=len(symbols),
            recommendations_count=len(output.recommendations),
            warnings_count=len(output.warnings),
            error_count=len(output.errors),
        )

        return state

    def _generate_risk_summary(self, output: AnalysisOutput) -> str:
        """Generate a summary of risk findings.

        Args:
            output: The analysis output to summarize.

        Returns:
            Summary string.
        """
        parts = []

        # Risk metrics summary
        metrics = output.risk_metrics
        if metrics.sharpe_ratio is not None:
            if metrics.sharpe_ratio >= 1.5:
                parts.append(f"Strong risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
            elif metrics.sharpe_ratio >= 1.0:
                parts.append(f"Good risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
            else:
                parts.append(f"Below-average risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")

        if metrics.volatility is not None:
            if metrics.volatility > 0.25:
                parts.append(f"High volatility ({metrics.volatility:.1%})")
            elif metrics.volatility < 0.15:
                parts.append(f"Low volatility ({metrics.volatility:.1%})")

        if metrics.beta is not None:
            if metrics.beta > 1.1:
                parts.append(f"Higher market sensitivity (Beta: {metrics.beta:.2f})")
            elif metrics.beta < 0.9:
                parts.append(f"Lower market sensitivity (Beta: {metrics.beta:.2f})")

        # Diversification summary
        if output.correlations.diversification_score is not None:
            score = output.correlations.diversification_score
            if score >= 0.6:
                parts.append(f"Well diversified (score: {score:.2f})")
            elif score < 0.4:
                parts.append(f"Poorly diversified (score: {score:.2f})")

        # Benchmark summary
        bench = output.benchmark_comparison
        if bench.relative_performance is not None:
            if bench.relative_performance > 0:
                parts.append(f"Outperforming benchmark by {bench.relative_performance:.1%}")
            else:
                parts.append(f"Underperforming benchmark by {abs(bench.relative_performance):.1%}")

        # Error summary
        if output.errors:
            parts.append(f"Encountered {len(output.errors)} error(s) during analysis")

        return ". ".join(parts) if parts else "Analysis completed successfully."

    def _generate_recommendations(
        self, output: AnalysisOutput, _symbols: list[str]
    ) -> list[str]:
        """Generate recommendations based on analysis.

        Args:
            output: The analysis output.
            symbols: List of analyzed symbols.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Risk-based recommendations
        metrics = output.risk_metrics
        if metrics.volatility is not None and metrics.volatility > 0.25:
            recommendations.append(
                "Consider reducing position sizes or adding defensive holdings to lower portfolio volatility"
            )

        if metrics.beta is not None and metrics.beta > 1.2:
            recommendations.append(
                "Portfolio has high market sensitivity. Consider adding low-beta assets for downside protection"
            )

        if metrics.max_drawdown is not None and metrics.max_drawdown > 0.2:
            recommendations.append(
                "Historical drawdowns are significant. Consider implementing stop-loss strategies"
            )

        # Correlation-based recommendations
        if output.correlations.highly_correlated_pairs:
            pairs_str = ", ".join(
                f"{p[0]}-{p[1]}" for p in output.correlations.highly_correlated_pairs[:3]
            )
            recommendations.append(
                f"Highly correlated positions detected ({pairs_str}). Consider reducing overlap"
            )

        if (
            output.correlations.diversification_score is not None
            and output.correlations.diversification_score < 0.4
        ):
            recommendations.append(
                "Low diversification score. Consider adding uncorrelated assets to the portfolio"
            )

        # Benchmark-based recommendations
        bench = output.benchmark_comparison
        if bench.tracking_error is not None and bench.tracking_error > 0.1:
            recommendations.append(
                "High tracking error vs benchmark. Review if active risk is intentional"
            )

        if bench.information_ratio is not None and bench.information_ratio < 0:
            recommendations.append(
                "Negative information ratio suggests active decisions are not adding value"
            )

        # If no specific recommendations, provide general advice
        if not recommendations:
            recommendations.append(
                "Portfolio metrics are within acceptable ranges. Continue monitoring for changes"
            )

        return recommendations

    def _generate_warnings(self, output: AnalysisOutput) -> list[str]:
        """Generate warnings based on analysis.

        Args:
            output: The analysis output.

        Returns:
            List of warning strings.
        """
        warnings = []

        metrics = output.risk_metrics
        if metrics.var_99 is not None and metrics.var_99 > 0.05:
            warnings.append(
                f"VaR(99%) of {metrics.var_99:.1%} indicates significant tail risk"
            )

        if metrics.max_drawdown is not None and metrics.max_drawdown > 0.25:
            warnings.append(
                f"Historical max drawdown of {metrics.max_drawdown:.1%} exceeds typical tolerance"
            )

        if (
            output.correlations.highly_correlated_pairs
            and len(output.correlations.highly_correlated_pairs) > 3
        ):
            warnings.append(
                f"Multiple highly correlated pairs ({len(output.correlations.highly_correlated_pairs)}) "
                "may amplify losses during market stress"
            )

        return warnings
