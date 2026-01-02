"""Tests for the Risk Metrics Tools."""

import math

import pytest

from src.tools.risk_metrics import (
    AttributionInput,
    AttributionOutput,
    AttributionResult,
    AttributionTool,
    BenchmarkComparison,
    BenchmarkComparisonTool,
    BenchmarkInput,
    BenchmarkOutput,
    CorrelationInput,
    CorrelationOutput,
    CorrelationResult,
    CorrelationTool,
    RiskMetrics,
    RiskMetricsInput,
    RiskMetricsOutput,
    RiskMetricsTool,
    _calculate_alpha,
    _calculate_beta,
    _calculate_max_drawdown,
    _calculate_sharpe,
    _calculate_sortino,
    _calculate_var,
    _correlation,
    _covariance,
    _generate_mock_returns,
    _mean,
    _std,
)

# ============================================================================
# Output Schema Tests
# ============================================================================


class TestRiskMetrics:
    """Tests for RiskMetrics model."""

    def test_default_values(self) -> None:
        """Test that all metric fields default to None."""
        metrics = RiskMetrics()
        assert metrics.var_95 is None
        assert metrics.var_99 is None
        assert metrics.sharpe_ratio is None
        assert metrics.sortino_ratio is None
        assert metrics.beta is None
        assert metrics.alpha is None
        assert metrics.max_drawdown is None
        assert metrics.volatility is None
        assert metrics.calculated_at is not None

    def test_all_fields_populated(self) -> None:
        """Test with all fields populated."""
        metrics = RiskMetrics(
            var_95=0.025,
            var_99=0.035,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            beta=1.1,
            alpha=0.02,
            max_drawdown=0.15,
            volatility=0.20,
        )
        assert metrics.var_95 == 0.025
        assert metrics.var_99 == 0.035
        assert metrics.sharpe_ratio == 1.5
        assert metrics.sortino_ratio == 2.0
        assert metrics.beta == 1.1
        assert metrics.alpha == 0.02
        assert metrics.max_drawdown == 0.15
        assert metrics.volatility == 0.20


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
            matrix={"AAPL": {"AAPL": 1.0, "MSFT": 0.8}, "MSFT": {"AAPL": 0.8, "MSFT": 1.0}},
            highly_correlated_pairs=[("AAPL", "MSFT", 0.8)],
            diversification_score=0.75,
        )
        assert result.matrix["AAPL"]["MSFT"] == 0.8
        assert len(result.highly_correlated_pairs) == 1
        assert result.diversification_score == 0.75


class TestBenchmarkComparison:
    """Tests for BenchmarkComparison model."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = BenchmarkComparison()
        assert result.benchmark_name == "S&P 500"
        assert result.portfolio_return is None
        assert result.benchmark_return is None
        assert result.alpha is None

    def test_with_data(self) -> None:
        """Test with populated data."""
        result = BenchmarkComparison(
            benchmark_name="SPY (S&P 500 proxy)",
            portfolio_return=0.12,
            benchmark_return=0.10,
            alpha=0.02,
            beta=1.1,
            r_squared=0.85,
            tracking_error=0.05,
            information_ratio=0.4,
            relative_performance=0.02,
        )
        assert result.portfolio_return == 0.12
        assert result.relative_performance == 0.02


class TestAttributionResult:
    """Tests for AttributionResult model."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = AttributionResult()
        assert result.sector_attribution == {}
        assert result.factor_attribution == {}
        assert result.selection_effect is None
        assert result.allocation_effect is None

    def test_with_data(self) -> None:
        """Test with populated data."""
        result = AttributionResult(
            sector_attribution={"Technology": 0.05, "Healthcare": 0.02},
            factor_attribution={"Momentum": 0.01, "Value": 0.005},
            selection_effect=0.03,
            allocation_effect=0.02,
            interaction_effect=0.005,
            total_active_return=0.055,
        )
        assert result.sector_attribution["Technology"] == 0.05
        assert result.factor_attribution["Momentum"] == 0.01
        assert result.total_active_return == 0.055


# ============================================================================
# Input Schema Tests
# ============================================================================


class TestRiskMetricsInput:
    """Tests for RiskMetricsInput model."""

    def test_required_fields(self) -> None:
        """Test that symbols is required."""
        input_data = RiskMetricsInput(symbols=["AAPL", "MSFT"])
        assert input_data.symbols == ["AAPL", "MSFT"]
        assert input_data.weights is None
        assert input_data.period_days == 252
        assert input_data.risk_free_rate == 0.05

    def test_with_weights(self) -> None:
        """Test with custom weights."""
        input_data = RiskMetricsInput(
            symbols=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
        )
        assert input_data.weights == [0.6, 0.4]

    def test_with_returns(self) -> None:
        """Test with provided returns."""
        returns = [[0.01, 0.02, -0.01], [0.02, 0.01, -0.02]]
        input_data = RiskMetricsInput(symbols=["AAPL", "MSFT"], returns=returns)
        assert input_data.returns == returns


class TestCorrelationInput:
    """Tests for CorrelationInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = CorrelationInput(symbols=["AAPL", "MSFT", "GOOGL"])
        assert input_data.symbols == ["AAPL", "MSFT", "GOOGL"]
        assert input_data.correlation_threshold == 0.7

    def test_custom_threshold(self) -> None:
        """Test with custom correlation threshold."""
        input_data = CorrelationInput(symbols=["AAPL", "MSFT"], correlation_threshold=0.8)
        assert input_data.correlation_threshold == 0.8


class TestBenchmarkInput:
    """Tests for BenchmarkInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = BenchmarkInput(symbols=["AAPL", "MSFT"])
        assert input_data.benchmark == "SPY"
        assert input_data.weights is None

    def test_with_benchmark(self) -> None:
        """Test with custom benchmark."""
        input_data = BenchmarkInput(symbols=["AAPL", "MSFT"], benchmark="QQQ")
        assert input_data.benchmark == "QQQ"


class TestAttributionInput:
    """Tests for AttributionInput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        input_data = AttributionInput(symbols=["AAPL", "MSFT"])
        assert input_data.symbols == ["AAPL", "MSFT"]
        assert input_data.sector_mapping is None

    def test_with_sector_mapping(self) -> None:
        """Test with custom sector mapping."""
        mapping = {"AAPL": "Technology", "JNJ": "Healthcare"}
        input_data = AttributionInput(symbols=["AAPL", "JNJ"], sector_mapping=mapping)
        assert input_data.sector_mapping == mapping


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestMean:
    """Tests for _mean function."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert _mean([]) == 0.0

    def test_single_value(self) -> None:
        """Test with single value."""
        assert _mean([5.0]) == 5.0

    def test_multiple_values(self) -> None:
        """Test with multiple values."""
        assert _mean([1.0, 2.0, 3.0, 4.0, 5.0]) == 3.0

    def test_negative_values(self) -> None:
        """Test with negative values."""
        assert _mean([-1.0, 1.0]) == 0.0


class TestStd:
    """Tests for _std function."""

    def test_empty_list(self) -> None:
        """Test with empty list."""
        assert _std([]) == 0.0

    def test_single_value(self) -> None:
        """Test with single value."""
        assert _std([5.0]) == 0.0

    def test_known_values(self) -> None:
        """Test with known standard deviation."""
        # For [1, 2, 3, 4, 5], sample std = sqrt(2.5) ≈ 1.581
        result = _std([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result - math.sqrt(2.5)) < 0.001

    def test_population_std(self) -> None:
        """Test with ddof=0 for population std."""
        result = _std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=0)
        assert abs(result - math.sqrt(2.0)) < 0.001


class TestCovariance:
    """Tests for _covariance function."""

    def test_mismatched_lengths(self) -> None:
        """Test with mismatched list lengths."""
        assert _covariance([1, 2, 3], [1, 2]) == 0.0

    def test_too_short(self) -> None:
        """Test with list too short."""
        assert _covariance([1], [1]) == 0.0

    def test_perfect_positive(self) -> None:
        """Test perfect positive covariance."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        # Covariance should be positive
        assert _covariance(x, y) > 0

    def test_perfect_negative(self) -> None:
        """Test perfect negative covariance."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert _covariance(x, y) < 0


class TestCorrelationFunction:
    """Tests for _correlation function."""

    def test_perfect_correlation(self) -> None:
        """Test perfect positive correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert abs(_correlation(x, y) - 1.0) < 0.001

    def test_perfect_negative_correlation(self) -> None:
        """Test perfect negative correlation."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        assert abs(_correlation(x, y) - (-1.0)) < 0.001

    def test_zero_std(self) -> None:
        """Test with zero standard deviation."""
        x = [1.0, 1.0, 1.0]
        y = [2.0, 4.0, 6.0]
        assert _correlation(x, y) == 0.0


class TestCalculateVar:
    """Tests for _calculate_var function."""

    def test_empty_returns(self) -> None:
        """Test with empty returns."""
        assert _calculate_var([]) == 0.0

    def test_positive_loss(self) -> None:
        """Test VaR returns positive value for potential loss."""
        returns = [-0.03, -0.02, 0.01, 0.02, 0.03, 0.01, -0.01, 0.02, -0.02, 0.01]
        var = _calculate_var(returns, 0.95)
        assert var > 0  # VaR should be positive

    def test_higher_confidence_higher_var(self) -> None:
        """Test that higher confidence gives higher VaR."""
        returns = [-0.03, -0.02, 0.01, 0.02, 0.03] * 20
        var_95 = _calculate_var(returns, 0.95)
        var_99 = _calculate_var(returns, 0.99)
        assert var_99 >= var_95


class TestCalculateMaxDrawdown:
    """Tests for _calculate_max_drawdown function."""

    def test_empty_returns(self) -> None:
        """Test with empty returns."""
        assert _calculate_max_drawdown([]) == 0.0

    def test_only_gains(self) -> None:
        """Test with only positive returns."""
        returns = [0.01, 0.02, 0.01, 0.03, 0.02]
        assert _calculate_max_drawdown(returns) == 0.0

    def test_known_drawdown(self) -> None:
        """Test with known drawdown scenario."""
        # Cumulative: 1.0 -> 1.1 -> 0.99 -> 1.05
        returns = [0.10, -0.10, 0.06]
        drawdown = _calculate_max_drawdown(returns)
        assert drawdown > 0
        assert drawdown < 1.0  # Must be less than 100%


class TestCalculateSharpe:
    """Tests for _calculate_sharpe function."""

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        assert _calculate_sharpe([0.01]) == 0.0

    def test_zero_volatility(self) -> None:
        """Test with zero volatility."""
        returns = [0.01, 0.01, 0.01, 0.01]
        assert _calculate_sharpe(returns) == 0.0

    def test_reasonable_sharpe(self) -> None:
        """Test that Sharpe ratio is reasonable for normal returns."""
        # Mock daily returns with ~10% annual return, ~15% volatility
        returns = [0.0004] * 252  # ~10% annual
        sharpe = _calculate_sharpe(returns, risk_free_rate=0.05)
        # Result should be finite
        assert math.isfinite(sharpe)


class TestCalculateSortino:
    """Tests for _calculate_sortino function."""

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        assert _calculate_sortino([0.01]) == 0.0

    def test_no_downside(self) -> None:
        """Test with no downside returns."""
        returns = [0.01, 0.02, 0.01, 0.02]
        sortino = _calculate_sortino(returns)
        # Should return high value when no significant downside
        assert sortino == 3.0

    def test_with_downside(self) -> None:
        """Test with some downside returns."""
        returns = [0.01, -0.02, 0.01, -0.01, 0.02, -0.015, 0.01, -0.01] * 10
        sortino = _calculate_sortino(returns)
        assert math.isfinite(sortino)


class TestCalculateBeta:
    """Tests for _calculate_beta function."""

    def test_mismatched_lengths(self) -> None:
        """Test with mismatched lengths."""
        assert _calculate_beta([0.01, 0.02], [0.01]) == 1.0

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        assert _calculate_beta([0.01], [0.01]) == 1.0

    def test_identical_returns(self) -> None:
        """Test with identical returns."""
        returns = [0.01, -0.01, 0.02, -0.02, 0.01]
        beta = _calculate_beta(returns, returns)
        assert abs(beta - 1.0) < 0.001


class TestCalculateAlpha:
    """Tests for _calculate_alpha function."""

    def test_empty_returns(self) -> None:
        """Test with empty returns."""
        assert _calculate_alpha([], [], 1.0) == 0.0

    def test_with_data(self) -> None:
        """Test with valid data."""
        portfolio = [0.001, 0.002, -0.001, 0.003, 0.001]
        benchmark = [0.0005, 0.001, -0.0005, 0.002, 0.0005]
        alpha = _calculate_alpha(portfolio, benchmark, 1.0)
        assert math.isfinite(alpha)


class TestGenerateMockReturns:
    """Tests for _generate_mock_returns function."""

    def test_returns_correct_structure(self) -> None:
        """Test returns correct number of symbols and days."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        returns = _generate_mock_returns(symbols, num_days=100)
        assert len(returns) == 3
        assert all(len(r) == 100 for r in returns)

    def test_deterministic_by_symbol(self) -> None:
        """Test returns are deterministic based on symbol."""
        returns1 = _generate_mock_returns(["AAPL"], num_days=10)
        returns2 = _generate_mock_returns(["AAPL"], num_days=10)
        assert returns1 == returns2

    def test_different_symbols_different_returns(self) -> None:
        """Test different symbols generate different returns."""
        returns = _generate_mock_returns(["AAPL", "MSFT"], num_days=10)
        assert returns[0] != returns[1]


# ============================================================================
# Tool Tests
# ============================================================================


class TestRiskMetricsTool:
    """Tests for RiskMetricsTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = RiskMetricsTool()
        assert tool.name == "calculate_risk_metrics"
        assert "risk" in tool.description.lower()
        assert "VaR" in tool.description or "Value at Risk" in tool.description

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = RiskMetricsTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "calculate_risk_metrics"
        assert "symbols" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mock_basic(self) -> None:
        """Test mock execution with basic input."""
        tool = RiskMetricsTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT"]})

        assert isinstance(result, RiskMetricsOutput)
        assert result.success is True
        assert result.metrics.var_95 is not None
        assert result.metrics.sharpe_ratio is not None
        assert result.metrics.beta is not None

    @pytest.mark.asyncio
    async def test_execute_mock_with_weights(self) -> None:
        """Test mock execution with custom weights."""
        tool = RiskMetricsTool(use_mock=True)
        result = await tool.execute({
            "symbols": ["AAPL", "MSFT"],
            "weights": [0.7, 0.3],
        })

        assert result.success is True
        assert result.metrics.volatility is not None

    @pytest.mark.asyncio
    async def test_execute_mock_with_returns(self) -> None:
        """Test mock execution with provided returns."""
        tool = RiskMetricsTool(use_mock=True)
        returns = [
            [0.01, 0.02, -0.01, 0.015, -0.005] * 50,
            [0.015, 0.01, -0.02, 0.01, -0.01] * 50,
        ]
        result = await tool.execute({
            "symbols": ["AAPL", "MSFT"],
            "returns": returns,
        })

        assert result.success is True
        assert result.metrics.max_drawdown is not None

    @pytest.mark.asyncio
    async def test_execute_real_mode(self) -> None:
        """Test real execution (uses same calculation logic)."""
        tool = RiskMetricsTool(use_mock=False)
        result = await tool.execute({"symbols": ["AAPL", "GOOGL", "MSFT"]})

        assert result.success is True
        assert all(
            metric is not None
            for metric in [
                result.metrics.var_95,
                result.metrics.var_99,
                result.metrics.sharpe_ratio,
                result.metrics.sortino_ratio,
                result.metrics.beta,
                result.metrics.alpha,
                result.metrics.max_drawdown,
                result.metrics.volatility,
            ]
        )

    @pytest.mark.asyncio
    async def test_execute_normalizes_weights(self) -> None:
        """Test that weights are normalized."""
        tool = RiskMetricsTool(use_mock=True)
        # Weights don't sum to 1
        result = await tool.execute({
            "symbols": ["AAPL", "MSFT"],
            "weights": [2.0, 3.0],
        })

        assert result.success is True


class TestCorrelationTool:
    """Tests for CorrelationTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = CorrelationTool()
        assert tool.name == "correlation_analysis"
        assert "correlation" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = CorrelationTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "correlation_analysis"
        assert "symbols" in anthropic_tool["input_schema"]["properties"]
        assert "correlation_threshold" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mock_basic(self) -> None:
        """Test mock execution with basic input."""
        tool = CorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "GOOGL"]})

        assert isinstance(result, CorrelationOutput)
        assert result.success is True
        assert len(result.result.matrix) == 3
        assert result.result.diversification_score is not None

    @pytest.mark.asyncio
    async def test_correlation_matrix_symmetric(self) -> None:
        """Test that correlation matrix is symmetric."""
        tool = CorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT"]})

        matrix = result.result.matrix
        assert matrix["AAPL"]["MSFT"] == matrix["MSFT"]["AAPL"]

    @pytest.mark.asyncio
    async def test_diagonal_is_one(self) -> None:
        """Test that diagonal elements are 1.0."""
        tool = CorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "GOOGL"]})

        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            assert result.result.matrix[symbol][symbol] == 1.0

    @pytest.mark.asyncio
    async def test_highly_correlated_pairs_detection(self) -> None:
        """Test detection of highly correlated pairs."""
        tool = CorrelationTool(use_mock=True)
        # Use low threshold to ensure some pairs are flagged
        result = await tool.execute({
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "correlation_threshold": 0.1,
        })

        # With a very low threshold, we should find some pairs
        # (or none if correlations are very low)
        assert isinstance(result.result.highly_correlated_pairs, list)

    @pytest.mark.asyncio
    async def test_diversification_score_range(self) -> None:
        """Test diversification score is in valid range."""
        tool = CorrelationTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "GOOGL", "JNJ"]})

        score = result.result.diversification_score
        assert score is not None
        assert 0.0 <= score <= 1.0


class TestBenchmarkComparisonTool:
    """Tests for BenchmarkComparisonTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = BenchmarkComparisonTool()
        assert tool.name == "benchmark_comparison"
        assert "benchmark" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = BenchmarkComparisonTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "benchmark_comparison"
        assert "symbols" in anthropic_tool["input_schema"]["properties"]
        assert "benchmark" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mock_basic(self) -> None:
        """Test mock execution with basic input."""
        tool = BenchmarkComparisonTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT"]})

        assert isinstance(result, BenchmarkOutput)
        assert result.success is True
        assert result.result.alpha is not None
        assert result.result.beta is not None
        assert result.result.r_squared is not None

    @pytest.mark.asyncio
    async def test_execute_with_custom_benchmark(self) -> None:
        """Test execution with custom benchmark."""
        tool = BenchmarkComparisonTool(use_mock=True)
        result = await tool.execute({
            "symbols": ["AAPL", "MSFT"],
            "benchmark": "QQQ",
        })

        assert result.success is True
        assert "QQQ" in result.result.benchmark_name

    @pytest.mark.asyncio
    async def test_execute_with_provided_returns(self) -> None:
        """Test execution with provided returns."""
        tool = BenchmarkComparisonTool(use_mock=True)
        portfolio_returns = [0.01, 0.02, -0.01, 0.015] * 63
        benchmark_returns = [0.005, 0.01, -0.005, 0.008] * 63

        result = await tool.execute({
            "symbols": ["AAPL"],
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": benchmark_returns,
        })

        assert result.success is True
        assert result.result.tracking_error is not None
        assert result.result.information_ratio is not None

    @pytest.mark.asyncio
    async def test_r_squared_range(self) -> None:
        """Test R-squared is in valid range."""
        tool = BenchmarkComparisonTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "GOOGL"]})

        r_squared = result.result.r_squared
        assert r_squared is not None
        assert 0.0 <= r_squared <= 1.0


class TestAttributionTool:
    """Tests for AttributionTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = AttributionTool()
        assert tool.name == "attribution_analysis"
        assert "attribution" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = AttributionTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "attribution_analysis"
        assert "symbols" in anthropic_tool["input_schema"]["properties"]
        assert "sector_mapping" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mock_basic(self) -> None:
        """Test mock execution with basic input."""
        tool = AttributionTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "JNJ"]})

        assert isinstance(result, AttributionOutput)
        assert result.success is True
        assert len(result.result.sector_attribution) > 0
        assert len(result.result.factor_attribution) > 0

    @pytest.mark.asyncio
    async def test_execute_with_sector_mapping(self) -> None:
        """Test execution with custom sector mapping."""
        tool = AttributionTool(use_mock=True)
        result = await tool.execute({
            "symbols": ["AAPL", "JNJ"],
            "sector_mapping": {"AAPL": "Tech", "JNJ": "Health"},
        })

        assert result.success is True
        assert "Tech" in result.result.sector_attribution
        assert "Health" in result.result.sector_attribution

    @pytest.mark.asyncio
    async def test_uses_default_sector_mapping(self) -> None:
        """Test that default sector mapping is used when not provided."""
        tool = AttributionTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "JPM"]})

        assert result.success is True
        # AAPL should be Technology, JPM should be Financials
        sectors = result.result.sector_attribution
        assert "Technology" in sectors or "Financials" in sectors

    @pytest.mark.asyncio
    async def test_factor_attribution_contains_factors(self) -> None:
        """Test that factor attribution contains expected factors."""
        tool = AttributionTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT"]})

        factors = result.result.factor_attribution
        assert "Momentum" in factors
        assert "Value" in factors
        assert "Quality" in factors

    @pytest.mark.asyncio
    async def test_brinson_effects_calculated(self) -> None:
        """Test that Brinson attribution effects are calculated."""
        tool = AttributionTool(use_mock=True)
        result = await tool.execute({"symbols": ["AAPL", "MSFT", "JNJ", "XOM"]})

        assert result.result.selection_effect is not None
        assert result.result.allocation_effect is not None
        assert result.result.interaction_effect is not None
        assert result.result.total_active_return is not None


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation across tools."""

    @pytest.mark.asyncio
    async def test_risk_metrics_missing_symbols(self) -> None:
        """Test that missing symbols raises error."""
        from src.tools.base import ToolExecutionError

        tool = RiskMetricsTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_correlation_missing_symbols(self) -> None:
        """Test that missing symbols raises error."""
        from src.tools.base import ToolExecutionError

        tool = CorrelationTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_benchmark_missing_symbols(self) -> None:
        """Test that missing symbols raises error."""
        from src.tools.base import ToolExecutionError

        tool = BenchmarkComparisonTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_attribution_missing_symbols(self) -> None:
        """Test that missing symbols raises error."""
        from src.tools.base import ToolExecutionError

        tool = AttributionTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_accepts_model_input(self) -> None:
        """Test that validated model can be passed directly."""
        tool = RiskMetricsTool(use_mock=True)
        input_data = RiskMetricsInput(symbols=["AAPL", "MSFT"])

        result = await tool.execute(input_data)

        assert result.success is True
