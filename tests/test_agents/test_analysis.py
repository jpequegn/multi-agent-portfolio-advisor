"""Tests for the Analysis Agent."""

import pytest

from src.agents.analysis import (
    AnalysisAgent,
    AnalysisOutput,
    AttributionInput,
    AttributionResult,
    BenchmarkComparison,
    BenchmarkInput,
    CorrelationInput,
    CorrelationOutput,
    CorrelationResult,
    PlaceholderAttributionTool,
    PlaceholderBenchmarkTool,
    PlaceholderCorrelationTool,
    PlaceholderRiskMetricsTool,
    RiskMetrics,
    RiskMetricsInput,
    RiskMetricsOutput,
)
from src.agents.base import AgentState
from src.tools.base import BaseTool, ToolRegistry


class TestRiskMetrics:
    """Tests for RiskMetrics model."""

    def test_default_values(self) -> None:
        """Test that all fields default to None."""
        metrics = RiskMetrics()
        assert metrics.var_95 is None
        assert metrics.sharpe_ratio is None
        assert metrics.beta is None
        assert metrics.volatility is None

    def test_all_fields(self) -> None:
        """Test with all fields populated."""
        metrics = RiskMetrics(
            var_95=0.02,
            var_99=0.035,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=1.1,
            alpha=0.02,
            max_drawdown=0.15,
            volatility=0.18,
        )
        assert metrics.var_95 == 0.02
        assert metrics.sharpe_ratio == 1.5
        assert metrics.beta == 1.1


class TestCorrelationResult:
    """Tests for CorrelationResult model."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = CorrelationResult()
        assert result.matrix == {}
        assert result.highly_correlated_pairs == []
        assert result.diversification_score is None

    def test_with_data(self) -> None:
        """Test with populated data."""
        result = CorrelationResult(
            matrix={"AAPL": {"AAPL": 1.0, "GOOGL": 0.7}},
            highly_correlated_pairs=[("AAPL", "GOOGL", 0.7)],
            diversification_score=0.65,
        )
        assert "AAPL" in result.matrix
        assert len(result.highly_correlated_pairs) == 1


class TestBenchmarkComparison:
    """Tests for BenchmarkComparison model."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = BenchmarkComparison()
        assert result.benchmark_name == "S&P 500"
        assert result.portfolio_return is None

    def test_with_data(self) -> None:
        """Test with populated data."""
        result = BenchmarkComparison(
            benchmark_name="SPY Index",
            portfolio_return=0.12,
            benchmark_return=0.10,
            relative_performance=0.02,
        )
        assert result.benchmark_name == "SPY Index"
        assert result.relative_performance == 0.02


class TestAttributionResult:
    """Tests for AttributionResult model."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = AttributionResult()
        assert result.sector_attribution == {}
        assert result.factor_attribution == {}

    def test_with_data(self) -> None:
        """Test with populated data."""
        result = AttributionResult(
            sector_attribution={"Technology": 0.05, "Healthcare": 0.02},
            factor_attribution={"Market": 0.08, "Value": 0.01},
            selection_effect=0.015,
        )
        assert result.sector_attribution["Technology"] == 0.05
        assert result.selection_effect == 0.015


class TestAnalysisOutput:
    """Tests for AnalysisOutput model."""

    def test_default_values(self) -> None:
        """Test default values."""
        output = AnalysisOutput()
        assert output.risk_metrics is not None
        assert output.correlations is not None
        assert output.risk_summary == ""
        assert output.recommendations == []
        assert output.errors == []

    def test_with_data(self) -> None:
        """Test with populated data."""
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(sharpe_ratio=1.5),
            risk_summary="Good risk-adjusted returns",
            recommendations=["Consider diversifying"],
        )
        assert output.risk_metrics.sharpe_ratio == 1.5
        assert len(output.recommendations) == 1


# ============================================================================
# Placeholder Tool Tests
# ============================================================================


class TestPlaceholderRiskMetricsTool:
    """Tests for PlaceholderRiskMetricsTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_metrics(self) -> None:
        """Test mock returns valid risk metrics."""
        tool = PlaceholderRiskMetricsTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL", "MSFT"]})

        assert result.success is True
        assert result.metrics.var_95 is not None
        assert result.metrics.sharpe_ratio is not None
        assert result.metrics.volatility is not None

    @pytest.mark.asyncio
    async def test_mock_diversification_affects_metrics(self) -> None:
        """Test that more symbols improve diversification metrics."""
        tool = PlaceholderRiskMetricsTool(use_mock=True)

        result_few = await tool.execute({"symbols": ["AAPL"]})
        result_many = await tool.execute(
            {"symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]}
        )

        # More diversified portfolio should have better Sharpe
        assert result_many.metrics.sharpe_ratio > result_few.metrics.sharpe_ratio

    @pytest.mark.asyncio
    async def test_real_raises_not_implemented(self) -> None:
        """Test real implementation raises NotImplementedError."""
        tool = PlaceholderRiskMetricsTool(use_mock=False, fallback_to_mock=False)

        from src.tools.base import ToolExecutionError

        with pytest.raises(ToolExecutionError):
            await tool.execute({"symbols": ["AAPL"]})

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = PlaceholderRiskMetricsTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "calculate_risk_metrics"
        assert "risk" in anthropic_tool["description"].lower()


class TestPlaceholderCorrelationTool:
    """Tests for PlaceholderCorrelationTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_correlations(self) -> None:
        """Test mock returns valid correlation data."""
        tool = PlaceholderCorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL", "MSFT"]})

        assert result.success is True
        assert "AAPL" in result.result.matrix
        assert result.result.diversification_score is not None

    @pytest.mark.asyncio
    async def test_correlation_matrix_diagonal(self) -> None:
        """Test correlation matrix has 1.0 on diagonal."""
        tool = PlaceholderCorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL"]})

        assert result.result.matrix["AAPL"]["AAPL"] == 1.0
        assert result.result.matrix["GOOGL"]["GOOGL"] == 1.0

    @pytest.mark.asyncio
    async def test_correlation_matrix_symmetric(self) -> None:
        """Test correlation matrix is symmetric."""
        tool = PlaceholderCorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL"]})

        assert result.result.matrix["AAPL"]["GOOGL"] == result.result.matrix["GOOGL"]["AAPL"]


class TestPlaceholderBenchmarkTool:
    """Tests for PlaceholderBenchmarkTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_benchmark_data(self) -> None:
        """Test mock returns valid benchmark comparison."""
        tool = PlaceholderBenchmarkTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL"], "benchmark": "SPY"})

        assert result.success is True
        assert result.result.benchmark_name == "SPY Index"
        assert result.result.portfolio_return is not None
        assert result.result.relative_performance is not None

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = PlaceholderBenchmarkTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "benchmark_comparison"
        assert "benchmark" in anthropic_tool["description"].lower()


class TestPlaceholderAttributionTool:
    """Tests for PlaceholderAttributionTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_attribution_data(self) -> None:
        """Test mock returns valid attribution data."""
        tool = PlaceholderAttributionTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "JPM", "JNJ"]})

        assert result.success is True
        assert len(result.result.sector_attribution) > 0
        assert len(result.result.factor_attribution) > 0

    @pytest.mark.asyncio
    async def test_sector_attribution_varies_by_symbols(self) -> None:
        """Test that different symbols produce different sector attributions."""
        tool = PlaceholderAttributionTool(use_mock=True)

        result_tech = await tool.execute({"symbols": ["AAPL", "GOOGL", "MSFT"]})
        result_mixed = await tool.execute({"symbols": ["AAPL", "JPM", "JNJ"]})

        assert result_tech.result.sector_attribution != result_mixed.result.sector_attribution


# ============================================================================
# Analysis Agent Tests
# ============================================================================


class TestAnalysisAgent:
    """Tests for AnalysisAgent."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        agent = AnalysisAgent()

        assert agent.name == "analysis"
        assert "risk" in agent.description.lower()
        assert agent.get_tool("calculate_risk_metrics") is not None
        assert agent.get_tool("correlation_analysis") is not None
        assert agent.get_tool("benchmark_comparison") is not None
        assert agent.get_tool("attribution_analysis") is not None

    def test_initialization_custom_registry(self) -> None:
        """Test initialization with custom tool registry."""
        registry = ToolRegistry()
        registry.register(PlaceholderRiskMetricsTool(use_mock=True))
        registry.register(PlaceholderCorrelationTool(use_mock=True))
        registry.register(PlaceholderBenchmarkTool(use_mock=True))
        registry.register(PlaceholderAttributionTool(use_mock=True))

        agent = AnalysisAgent(tool_registry=registry)

        assert agent.get_tool("calculate_risk_metrics") is not None

    def test_system_prompt(self) -> None:
        """Test system prompt contains expected content."""
        agent = AnalysisAgent()

        assert "risk" in agent.system_prompt.lower()
        assert "calculate_risk_metrics" in agent.system_prompt
        assert "correlation_analysis" in agent.system_prompt
        assert "benchmark_comparison" in agent.system_prompt

    def test_tools_property(self) -> None:
        """Test tools are in Anthropic format."""
        agent = AnalysisAgent()
        tools = agent.tools

        assert len(tools) == 4
        tool_names = [t["name"] for t in tools]
        assert "calculate_risk_metrics" in tool_names
        assert "correlation_analysis" in tool_names
        assert "benchmark_comparison" in tool_names
        assert "attribution_analysis" in tool_names

    @pytest.mark.asyncio
    async def test_invoke_with_symbols(self) -> None:
        """Test invoke analyzes provided symbols."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL", "GOOGL", "MSFT"]})

        result = await agent.invoke(state)

        assert "analysis" in result.context
        analysis = result.context["analysis"]

        assert analysis["risk_metrics"]["sharpe_ratio"] is not None
        assert analysis["correlations"]["diversification_score"] is not None
        assert analysis["benchmark_comparison"]["relative_performance"] is not None
        assert len(analysis["recommendations"]) > 0
        assert analysis["risk_summary"] != ""

    @pytest.mark.asyncio
    async def test_invoke_with_research_data(self) -> None:
        """Test invoke can extract symbols from research data."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(
            context={
                "research": {
                    "symbols_analyzed": ["AAPL", "GOOGL"],
                    "market_data": {},
                }
            }
        )

        result = await agent.invoke(state)

        assert "analysis" in result.context
        analysis = result.context["analysis"]
        assert analysis["risk_metrics"]["sharpe_ratio"] is not None

    @pytest.mark.asyncio
    async def test_invoke_no_symbols(self) -> None:
        """Test invoke handles missing symbols gracefully."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(context={})

        result = await agent.invoke(state)

        assert len(result.errors) == 1
        assert "No symbols provided" in result.errors[0]
        assert "analysis" not in result.context

    @pytest.mark.asyncio
    async def test_invoke_adds_message(self) -> None:
        """Test invoke adds completion message."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "assistant"
        assert "Analysis completed" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_invoke_callable(self) -> None:
        """Test agent can be called as a function."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent(state)

        assert "analysis" in result.context


# ============================================================================
# Summary and Recommendation Generation Tests
# ============================================================================


class TestAnalysisAgentSummaryGeneration:
    """Tests for risk summary generation."""

    def test_summary_strong_sharpe(self) -> None:
        """Test summary with strong Sharpe ratio."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(sharpe_ratio=1.8),
        )

        summary = agent._generate_risk_summary(output)

        assert "Strong" in summary
        assert "1.80" in summary

    def test_summary_high_volatility(self) -> None:
        """Test summary with high volatility."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(volatility=0.30),
        )

        summary = agent._generate_risk_summary(output)

        assert "High volatility" in summary

    def test_summary_low_diversification(self) -> None:
        """Test summary with low diversification."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            correlations=CorrelationResult(diversification_score=0.3),
        )

        summary = agent._generate_risk_summary(output)

        assert "Poorly diversified" in summary

    def test_summary_outperforming_benchmark(self) -> None:
        """Test summary when outperforming benchmark."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            benchmark_comparison=BenchmarkComparison(relative_performance=0.05),
        )

        summary = agent._generate_risk_summary(output)

        assert "Outperforming" in summary

    def test_summary_with_errors(self) -> None:
        """Test summary mentions errors."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(errors=["Error 1", "Error 2"])

        summary = agent._generate_risk_summary(output)

        assert "2 error" in summary.lower()

    def test_summary_empty_output(self) -> None:
        """Test summary for empty output."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput()

        summary = agent._generate_risk_summary(output)

        assert summary == "Analysis completed successfully."


class TestAnalysisAgentRecommendations:
    """Tests for recommendation generation."""

    def test_recommendation_high_volatility(self) -> None:
        """Test recommendation for high volatility."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(volatility=0.30),
        )

        recommendations = agent._generate_recommendations(output, ["AAPL"])

        assert any("volatility" in r.lower() for r in recommendations)

    def test_recommendation_high_beta(self) -> None:
        """Test recommendation for high beta."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(beta=1.3),
        )

        recommendations = agent._generate_recommendations(output, ["AAPL"])

        assert any("beta" in r.lower() for r in recommendations)

    def test_recommendation_high_correlations(self) -> None:
        """Test recommendation for highly correlated positions."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            correlations=CorrelationResult(
                highly_correlated_pairs=[
                    ("AAPL", "GOOGL", 0.85),
                    ("MSFT", "GOOGL", 0.82),
                ]
            ),
        )

        recommendations = agent._generate_recommendations(output, ["AAPL", "GOOGL", "MSFT"])

        assert any("correlated" in r.lower() for r in recommendations)

    def test_recommendation_low_diversification(self) -> None:
        """Test recommendation for low diversification."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            correlations=CorrelationResult(diversification_score=0.3),
        )

        recommendations = agent._generate_recommendations(output, ["AAPL"])

        assert any("diversification" in r.lower() for r in recommendations)

    def test_recommendation_negative_information_ratio(self) -> None:
        """Test recommendation for negative information ratio."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            benchmark_comparison=BenchmarkComparison(information_ratio=-0.2),
        )

        recommendations = agent._generate_recommendations(output, ["AAPL"])

        assert any("information ratio" in r.lower() for r in recommendations)

    def test_default_recommendation(self) -> None:
        """Test default recommendation when no issues."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput()

        recommendations = agent._generate_recommendations(output, ["AAPL"])

        assert len(recommendations) == 1
        assert "acceptable" in recommendations[0].lower()


class TestAnalysisAgentWarnings:
    """Tests for warning generation."""

    def test_warning_high_var(self) -> None:
        """Test warning for high VaR."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(var_99=0.08),
        )

        warnings = agent._generate_warnings(output)

        assert any("tail risk" in w.lower() for w in warnings)

    def test_warning_high_drawdown(self) -> None:
        """Test warning for high max drawdown."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            risk_metrics=RiskMetrics(max_drawdown=0.30),
        )

        warnings = agent._generate_warnings(output)

        assert any("drawdown" in w.lower() for w in warnings)

    def test_warning_many_correlated_pairs(self) -> None:
        """Test warning for many highly correlated pairs."""
        agent = AnalysisAgent(use_mock_tools=True)
        output = AnalysisOutput(
            correlations=CorrelationResult(
                highly_correlated_pairs=[
                    ("A", "B", 0.9),
                    ("C", "D", 0.85),
                    ("E", "F", 0.82),
                    ("G", "H", 0.80),
                ]
            ),
        )

        warnings = agent._generate_warnings(output)

        assert any("market stress" in w.lower() for w in warnings)


# ============================================================================
# Tool Failure Handling Tests
# ============================================================================


class FailingRiskTool(BaseTool[RiskMetricsInput, RiskMetricsOutput]):
    """Tool that always fails for testing error handling."""

    name = "calculate_risk_metrics"
    description = "Failing risk metrics tool"

    @property
    def input_schema(self) -> type[RiskMetricsInput]:
        return RiskMetricsInput

    @property
    def output_schema(self) -> type[RiskMetricsOutput]:
        return RiskMetricsOutput

    async def _execute_real(self, _input_data: RiskMetricsInput) -> RiskMetricsOutput:
        raise RuntimeError("Simulated failure")

    async def _execute_mock(self, _input_data: RiskMetricsInput) -> RiskMetricsOutput:
        raise RuntimeError("Simulated mock failure")


class TestAnalysisAgentToolFailure:
    """Tests for Analysis Agent handling tool failures gracefully."""

    @pytest.mark.asyncio
    async def test_handles_risk_metrics_failure(self) -> None:
        """Test agent continues when risk metrics tool fails."""
        registry = ToolRegistry()
        registry.register(FailingRiskTool(use_mock=True))
        registry.register(PlaceholderCorrelationTool(use_mock=True))
        registry.register(PlaceholderBenchmarkTool(use_mock=True))
        registry.register(PlaceholderAttributionTool(use_mock=True))

        agent = AnalysisAgent(tool_registry=registry)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        # Agent should complete despite failure
        assert "analysis" in result.context
        analysis = result.context["analysis"]

        # Should have errors recorded
        assert len(analysis["errors"]) >= 1
        assert any("risk metrics" in e.lower() for e in analysis["errors"])

        # Other tools should still work
        assert analysis["correlations"]["diversification_score"] is not None

    @pytest.mark.asyncio
    async def test_continues_with_partial_analysis(self) -> None:
        """Test agent continues with partial data when some tools fail."""

        class FailingCorrelationTool(BaseTool[CorrelationInput, CorrelationOutput]):
            name = "correlation_analysis"
            description = "Failing correlation tool"

            @property
            def input_schema(self) -> type[CorrelationInput]:
                return CorrelationInput

            @property
            def output_schema(self) -> type[CorrelationOutput]:
                return CorrelationOutput

            async def _execute_real(
                self, _input_data: CorrelationInput
            ) -> CorrelationOutput:
                raise RuntimeError("Correlation failed")

            async def _execute_mock(
                self, _input_data: CorrelationInput
            ) -> CorrelationOutput:
                raise RuntimeError("Correlation mock failed")

        registry = ToolRegistry()
        registry.register(PlaceholderRiskMetricsTool(use_mock=True))
        registry.register(FailingCorrelationTool(use_mock=True))
        registry.register(PlaceholderBenchmarkTool(use_mock=True))
        registry.register(PlaceholderAttributionTool(use_mock=True))

        agent = AnalysisAgent(tool_registry=registry)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        analysis = result.context["analysis"]

        # Risk metrics should still work
        assert analysis["risk_metrics"]["sharpe_ratio"] is not None

        # Should have correlation error
        assert any("correlation" in e.lower() for e in analysis["errors"])


# ============================================================================
# Coverage Enhancement Tests
# ============================================================================


class TestAnalysisAgentCoverage:
    """Additional tests to ensure comprehensive coverage."""

    @pytest.mark.asyncio
    async def test_invoke_with_single_symbol(self) -> None:
        """Test invoke with a single symbol."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        assert "analysis" in result.context

    @pytest.mark.asyncio
    async def test_invoke_with_many_symbols(self) -> None:
        """Test invoke with many symbols."""
        agent = AnalysisAgent(use_mock_tools=True)
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "JPM", "JNJ"]
        state = AgentState(context={"symbols": symbols})

        result = await agent.invoke(state)

        analysis = result.context["analysis"]
        assert analysis["risk_metrics"]["sharpe_ratio"] is not None

    @pytest.mark.asyncio
    async def test_invoke_preserves_existing_state(self) -> None:
        """Test that invoke preserves existing state context."""
        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(
            context={
                "symbols": ["AAPL"],
                "research": {"market_data": {"AAPL": {}}},
                "user_request": "analyze my portfolio",
            }
        )

        result = await agent.invoke(state)

        # Existing context should be preserved
        assert result.context["research"] == {"market_data": {"AAPL": {}}}
        assert result.context["user_request"] == "analyze my portfolio"
        # New analysis data should be added
        assert "analysis" in result.context

    def test_agent_description(self) -> None:
        """Test agent description property."""
        agent = AnalysisAgent()
        assert "risk" in agent.description.lower()
        assert "performance" in agent.description.lower()

    def test_system_prompt_contains_all_tools(self) -> None:
        """Test system prompt mentions all available tools."""
        agent = AnalysisAgent()
        prompt = agent.system_prompt

        assert "calculate_risk_metrics" in prompt
        assert "correlation_analysis" in prompt
        assert "benchmark_comparison" in prompt
        assert "attribution_analysis" in prompt

    @pytest.mark.asyncio
    async def test_tracing_integration(self) -> None:
        """Test that tracing works with agent invocation."""
        from src.observability.tracing import TraceContext

        agent = AnalysisAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL"]})

        with TraceContext(session_id="test-analysis", user_id="test-user"):
            result = await agent.invoke(state)

        assert "analysis" in result.context


# ============================================================================
# Input Schema Tests
# ============================================================================


class TestToolInputSchemas:
    """Tests for tool input schemas."""

    def test_risk_metrics_input(self) -> None:
        """Test RiskMetricsInput model."""
        input_data = RiskMetricsInput(symbols=["AAPL", "GOOGL"])
        assert input_data.symbols == ["AAPL", "GOOGL"]
        assert input_data.period_days == 252  # default

    def test_risk_metrics_input_with_weights(self) -> None:
        """Test RiskMetricsInput with weights."""
        input_data = RiskMetricsInput(
            symbols=["AAPL", "GOOGL"],
            weights=[0.6, 0.4],
        )
        assert input_data.weights == [0.6, 0.4]

    def test_correlation_input(self) -> None:
        """Test CorrelationInput model."""
        input_data = CorrelationInput(symbols=["AAPL", "GOOGL"])
        assert input_data.symbols == ["AAPL", "GOOGL"]

    def test_benchmark_input(self) -> None:
        """Test BenchmarkInput model."""
        input_data = BenchmarkInput(symbols=["AAPL"], benchmark="QQQ")
        assert input_data.benchmark == "QQQ"

    def test_attribution_input(self) -> None:
        """Test AttributionInput model."""
        input_data = AttributionInput(symbols=["AAPL", "JPM"])
        assert input_data.symbols == ["AAPL", "JPM"]
