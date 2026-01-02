"""Risk Metrics Tools for portfolio analysis.

This module provides tools for calculating portfolio risk metrics,
correlations, benchmark comparisons, and performance attribution.

Note: For learning purposes, these use simplified calculations.
Production implementations would use more sophisticated methods.
"""

import hashlib
import math
import random
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolInput, ToolOutput

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
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    alpha: float | None = None
    beta: float | None = None
    r_squared: float | None = None
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
    total_active_return: float | None = None


# ============================================================================
# Tool Input/Output Schemas
# ============================================================================


class RiskMetricsInput(ToolInput):
    """Input for risk metrics calculation."""

    symbols: list[str] = Field(..., description="List of symbols in the portfolio")
    weights: list[float] | None = Field(default=None, description="Portfolio weights (must sum to 1)")
    returns: list[list[float]] | None = Field(
        default=None, description="Historical returns per symbol (optional, will generate if not provided)"
    )
    period_days: int = Field(default=252, description="Historical period in trading days")
    risk_free_rate: float = Field(default=0.05, description="Annual risk-free rate")


class RiskMetricsOutput(ToolOutput):
    """Output from risk metrics calculation."""

    metrics: RiskMetrics = Field(default_factory=lambda: RiskMetrics())


class CorrelationInput(ToolInput):
    """Input for correlation analysis."""

    symbols: list[str] = Field(..., description="Symbols to analyze")
    returns: list[list[float]] | None = Field(
        default=None, description="Historical returns per symbol (optional)"
    )
    period_days: int = Field(default=252, description="Historical period")
    correlation_threshold: float = Field(
        default=0.7, description="Threshold for flagging high correlation"
    )


class CorrelationOutput(ToolOutput):
    """Output from correlation analysis."""

    result: CorrelationResult = Field(default_factory=lambda: CorrelationResult())


class BenchmarkInput(ToolInput):
    """Input for benchmark comparison."""

    symbols: list[str] = Field(..., description="Portfolio symbols")
    weights: list[float] | None = Field(default=None, description="Portfolio weights")
    benchmark: str = Field(default="SPY", description="Benchmark symbol")
    portfolio_returns: list[float] | None = Field(
        default=None, description="Portfolio returns (optional)"
    )
    benchmark_returns: list[float] | None = Field(
        default=None, description="Benchmark returns (optional)"
    )
    period_days: int = Field(default=252, description="Historical period")


class BenchmarkOutput(ToolOutput):
    """Output from benchmark comparison."""

    result: BenchmarkComparison = Field(default_factory=lambda: BenchmarkComparison())


class AttributionInput(ToolInput):
    """Input for performance attribution."""

    symbols: list[str] = Field(..., description="Portfolio symbols")
    weights: list[float] | None = Field(default=None, description="Portfolio weights")
    sector_mapping: dict[str, str] | None = Field(
        default=None, description="Symbol to sector mapping"
    )
    returns: list[list[float]] | None = Field(
        default=None, description="Historical returns per symbol (optional)"
    )


class AttributionOutput(ToolOutput):
    """Output from performance attribution."""

    result: AttributionResult = Field(default_factory=lambda: AttributionResult())


# ============================================================================
# Calculation Utilities
# ============================================================================


def _generate_mock_returns(
    symbols: list[str],
    num_days: int = 252,
    base_volatility: float = 0.02,
) -> list[list[float]]:
    """Generate deterministic mock returns for symbols.

    Uses symbol hash for reproducibility.

    Args:
        symbols: List of stock symbols.
        num_days: Number of daily returns to generate.
        base_volatility: Base daily volatility.

    Returns:
        List of return series (one per symbol).
    """
    all_returns: list[list[float]] = []

    for symbol in symbols:
        # Deterministic seed based on symbol
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        random.seed(seed)

        # Generate returns with symbol-specific characteristics
        symbol_volatility = base_volatility * (0.8 + random.random() * 0.8)
        symbol_drift = (random.random() - 0.4) * 0.0005  # Slight positive drift

        returns = []
        for _ in range(num_days):
            daily_return = random.gauss(symbol_drift, symbol_volatility)
            returns.append(daily_return)

        all_returns.append(returns)

    return all_returns


def _mean(values: list[float]) -> float:
    """Calculate mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], ddof: int = 1) -> float:
    """Calculate standard deviation of a list.

    Args:
        values: List of values.
        ddof: Delta degrees of freedom (1 for sample, 0 for population).

    Returns:
        Standard deviation.
    """
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def _covariance(x: list[float], y: list[float]) -> float:
    """Calculate covariance between two lists."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mean_x = _mean(x)
    mean_y = _mean(y)
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False)) / (len(x) - 1)


def _correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation between two lists."""
    std_x = _std(x)
    std_y = _std(y)
    if std_x == 0 or std_y == 0:
        return 0.0
    return _covariance(x, y) / (std_x * std_y)


def _calculate_var(returns: list[float], confidence: float = 0.95) -> float:
    """Calculate Value at Risk using historical simulation.

    Args:
        returns: List of historical returns.
        confidence: Confidence level (e.g., 0.95 for 95%).

    Returns:
        VaR as a positive number representing potential loss.
    """
    if not returns:
        return 0.0
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    return -sorted_returns[index]  # Return as positive number


def _calculate_max_drawdown(returns: list[float]) -> float:
    """Calculate maximum drawdown from returns.

    Args:
        returns: List of daily returns.

    Returns:
        Maximum drawdown as a positive decimal.
    """
    if not returns:
        return 0.0

    # Calculate cumulative returns
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))

    # Find max drawdown
    peak = cumulative[0]
    max_dd = 0.0

    for value in cumulative:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def _calculate_sharpe(
    returns: list[float],
    risk_free_rate: float = 0.05,
    annualization_factor: float = 252,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: List of daily returns.
        risk_free_rate: Annual risk-free rate.
        annualization_factor: Trading days per year.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = _mean(returns) * annualization_factor
    std_return = _std(returns) * math.sqrt(annualization_factor)

    if std_return == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_return


def _calculate_sortino(
    returns: list[float],
    risk_free_rate: float = 0.05,
    annualization_factor: float = 252,
) -> float:
    """Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: List of daily returns.
        risk_free_rate: Annual risk-free rate.
        annualization_factor: Trading days per year.

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0

    # Calculate downside returns
    daily_rf = risk_free_rate / annualization_factor
    downside_returns = [r for r in returns if r < daily_rf]

    if len(downside_returns) < 2:
        # No significant downside, return high value
        return 3.0

    mean_return = _mean(returns) * annualization_factor
    downside_std = _std(downside_returns) * math.sqrt(annualization_factor)

    if downside_std == 0:
        return 0.0

    return (mean_return - risk_free_rate) / downside_std


def _calculate_beta(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
) -> float:
    """Calculate portfolio beta vs benchmark.

    Args:
        portfolio_returns: Portfolio daily returns.
        benchmark_returns: Benchmark daily returns.

    Returns:
        Beta coefficient.
    """
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
        return 1.0

    cov = _covariance(portfolio_returns, benchmark_returns)
    var_benchmark = _std(benchmark_returns) ** 2

    if var_benchmark == 0:
        return 1.0

    return cov / var_benchmark


def _calculate_alpha(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    beta: float,
    risk_free_rate: float = 0.05,
    annualization_factor: float = 252,
) -> float:
    """Calculate Jensen's alpha.

    Args:
        portfolio_returns: Portfolio daily returns.
        benchmark_returns: Benchmark daily returns.
        beta: Portfolio beta.
        risk_free_rate: Annual risk-free rate.
        annualization_factor: Trading days per year.

    Returns:
        Annualized alpha.
    """
    if not portfolio_returns or not benchmark_returns:
        return 0.0

    daily_rf = risk_free_rate / annualization_factor
    mean_portfolio = _mean(portfolio_returns)
    mean_benchmark = _mean(benchmark_returns)

    daily_alpha = mean_portfolio - (daily_rf + beta * (mean_benchmark - daily_rf))
    return daily_alpha * annualization_factor


# ============================================================================
# Risk Metrics Tool
# ============================================================================


class RiskMetricsTool(BaseTool[RiskMetricsInput, RiskMetricsOutput]):
    """Tool for calculating portfolio risk metrics.

    Calculates VaR, Sharpe ratio, Sortino ratio, beta, alpha,
    maximum drawdown, and volatility.
    """

    name = "calculate_risk_metrics"
    description = (
        "Calculates comprehensive portfolio risk metrics including Value at Risk (VaR), "
        "Sharpe ratio, Sortino ratio, beta, alpha, maximum drawdown, and volatility."
    )

    @property
    def input_schema(self) -> type[RiskMetricsInput]:
        return RiskMetricsInput

    @property
    def output_schema(self) -> type[RiskMetricsOutput]:
        return RiskMetricsOutput

    async def _execute_real(self, input_data: RiskMetricsInput) -> RiskMetricsOutput:
        """Calculate risk metrics from real or provided data."""
        return await self._calculate_metrics(input_data)

    async def _execute_mock(self, input_data: RiskMetricsInput) -> RiskMetricsOutput:
        """Calculate risk metrics with mock data."""
        return await self._calculate_metrics(input_data)

    async def _calculate_metrics(self, input_data: RiskMetricsInput) -> RiskMetricsOutput:
        """Core calculation logic.

        Args:
            input_data: Input with symbols, weights, and optional returns.

        Returns:
            Calculated risk metrics.
        """
        symbols = input_data.symbols
        weights = input_data.weights or [1.0 / len(symbols)] * len(symbols)

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Get or generate returns
        all_returns = input_data.returns or _generate_mock_returns(symbols, input_data.period_days)

        # Calculate portfolio returns
        portfolio_returns = []
        for i in range(len(all_returns[0])):
            daily_return = sum(
                weights[j] * all_returns[j][i] for j in range(len(symbols))
            )
            portfolio_returns.append(daily_return)

        # Generate benchmark returns (for beta/alpha calculation)
        benchmark_returns = _generate_mock_returns(["SPY"], input_data.period_days)[0]

        # Calculate metrics
        var_95 = _calculate_var(portfolio_returns, 0.95)
        var_99 = _calculate_var(portfolio_returns, 0.99)
        sharpe = _calculate_sharpe(portfolio_returns, input_data.risk_free_rate)
        sortino = _calculate_sortino(portfolio_returns, input_data.risk_free_rate)
        max_dd = _calculate_max_drawdown(portfolio_returns)
        volatility = _std(portfolio_returns) * math.sqrt(252)
        beta = _calculate_beta(portfolio_returns, benchmark_returns)
        alpha = _calculate_alpha(
            portfolio_returns, benchmark_returns, beta, input_data.risk_free_rate
        )

        return RiskMetricsOutput(
            metrics=RiskMetrics(
                var_95=round(var_95, 4),
                var_99=round(var_99, 4),
                sharpe_ratio=round(sharpe, 4),
                sortino_ratio=round(sortino, 4),
                beta=round(beta, 4),
                alpha=round(alpha, 4),
                max_drawdown=round(max_dd, 4),
                volatility=round(volatility, 4),
                calculated_at=datetime.now(UTC),
            )
        )


# ============================================================================
# Correlation Analysis Tool
# ============================================================================


class CorrelationTool(BaseTool[CorrelationInput, CorrelationOutput]):
    """Tool for analyzing cross-asset correlations.

    Calculates correlation matrix and identifies highly correlated pairs.
    """

    name = "correlation_analysis"
    description = (
        "Analyzes cross-asset correlations in the portfolio. Returns correlation matrix "
        "and identifies pairs with high correlation that may reduce diversification."
    )

    @property
    def input_schema(self) -> type[CorrelationInput]:
        return CorrelationInput

    @property
    def output_schema(self) -> type[CorrelationOutput]:
        return CorrelationOutput

    async def _execute_real(self, input_data: CorrelationInput) -> CorrelationOutput:
        """Calculate correlations from real or provided data."""
        return await self._calculate_correlations(input_data)

    async def _execute_mock(self, input_data: CorrelationInput) -> CorrelationOutput:
        """Calculate correlations with mock data."""
        return await self._calculate_correlations(input_data)

    async def _calculate_correlations(self, input_data: CorrelationInput) -> CorrelationOutput:
        """Core correlation calculation logic.

        Args:
            input_data: Input with symbols and optional returns.

        Returns:
            Correlation analysis results.
        """
        symbols = input_data.symbols

        # Get or generate returns
        all_returns = input_data.returns or _generate_mock_returns(symbols, input_data.period_days)

        # Calculate correlation matrix
        matrix: dict[str, dict[str, float]] = {}
        highly_correlated: list[tuple[str, str, float]] = []

        for i, sym_i in enumerate(symbols):
            matrix[sym_i] = {}
            for j, sym_j in enumerate(symbols):
                if i == j:
                    corr = 1.0
                elif j < i:
                    # Use already calculated value
                    corr = matrix[sym_j][sym_i]
                else:
                    corr = _correlation(all_returns[i], all_returns[j])

                matrix[sym_i][sym_j] = round(corr, 4)

                # Track highly correlated pairs (only count once)
                if i < j and abs(corr) >= input_data.correlation_threshold:
                    highly_correlated.append((sym_i, sym_j, round(corr, 4)))

        # Calculate diversification score
        # Based on average off-diagonal correlation
        off_diag_corrs = []
        for i, sym_i in enumerate(symbols):
            for j, sym_j in enumerate(symbols):
                if i < j:
                    off_diag_corrs.append(abs(matrix[sym_i][sym_j]))

        avg_correlation = _mean(off_diag_corrs) if off_diag_corrs else 0.0
        # Score is inverse of average correlation (lower correlation = better diversification)
        diversification_score = 1.0 - avg_correlation

        return CorrelationOutput(
            result=CorrelationResult(
                matrix=matrix,
                highly_correlated_pairs=highly_correlated,
                diversification_score=round(diversification_score, 4),
            )
        )


# ============================================================================
# Benchmark Comparison Tool
# ============================================================================


class BenchmarkComparisonTool(BaseTool[BenchmarkInput, BenchmarkOutput]):
    """Tool for comparing portfolio performance against a benchmark.

    Calculates alpha, beta, R-squared, tracking error, and information ratio.
    """

    name = "benchmark_comparison"
    description = (
        "Compares portfolio performance against a benchmark index. Calculates "
        "alpha, beta, R-squared, tracking error, and information ratio."
    )

    @property
    def input_schema(self) -> type[BenchmarkInput]:
        return BenchmarkInput

    @property
    def output_schema(self) -> type[BenchmarkOutput]:
        return BenchmarkOutput

    async def _execute_real(self, input_data: BenchmarkInput) -> BenchmarkOutput:
        """Calculate benchmark comparison from real or provided data."""
        return await self._calculate_comparison(input_data)

    async def _execute_mock(self, input_data: BenchmarkInput) -> BenchmarkOutput:
        """Calculate benchmark comparison with mock data."""
        return await self._calculate_comparison(input_data)

    async def _calculate_comparison(self, input_data: BenchmarkInput) -> BenchmarkOutput:
        """Core benchmark comparison logic.

        Args:
            input_data: Input with portfolio info and benchmark.

        Returns:
            Benchmark comparison results.
        """
        symbols = input_data.symbols
        weights = input_data.weights or [1.0 / len(symbols)] * len(symbols)

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Get portfolio returns
        if input_data.portfolio_returns:
            portfolio_returns = input_data.portfolio_returns
        else:
            all_returns = _generate_mock_returns(symbols, input_data.period_days)
            portfolio_returns = []
            for i in range(len(all_returns[0])):
                daily_return = sum(
                    weights[j] * all_returns[j][i] for j in range(len(symbols))
                )
                portfolio_returns.append(daily_return)

        # Get benchmark returns
        if input_data.benchmark_returns:
            benchmark_returns = input_data.benchmark_returns
        else:
            benchmark_returns = _generate_mock_returns(
                [input_data.benchmark], input_data.period_days
            )[0]

        # Calculate metrics
        beta = _calculate_beta(portfolio_returns, benchmark_returns)
        alpha = _calculate_alpha(portfolio_returns, benchmark_returns, beta)

        # R-squared
        corr = _correlation(portfolio_returns, benchmark_returns)
        r_squared = corr ** 2

        # Tracking error (std of return differences)
        tracking_diff = [p - b for p, b in zip(portfolio_returns, benchmark_returns, strict=False)]
        tracking_error = _std(tracking_diff) * math.sqrt(252)

        # Information ratio
        excess_return = _mean(tracking_diff) * 252
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0.0

        # Total returns
        portfolio_total = sum(portfolio_returns) * 252 / len(portfolio_returns) if portfolio_returns else 0.0
        benchmark_total = sum(benchmark_returns) * 252 / len(benchmark_returns) if benchmark_returns else 0.0
        relative_perf = portfolio_total - benchmark_total

        return BenchmarkOutput(
            result=BenchmarkComparison(
                benchmark_name=f"{input_data.benchmark} (S&P 500 proxy)",
                portfolio_return=round(portfolio_total, 4),
                benchmark_return=round(benchmark_total, 4),
                alpha=round(alpha, 4),
                beta=round(beta, 4),
                r_squared=round(r_squared, 4),
                tracking_error=round(tracking_error, 4),
                information_ratio=round(information_ratio, 4),
                relative_performance=round(relative_perf, 4),
            )
        )


# ============================================================================
# Attribution Analysis Tool
# ============================================================================


# Default sector mapping for common stocks
DEFAULT_SECTOR_MAPPING = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "WMT": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "SPY": "Index",
}


class AttributionTool(BaseTool[AttributionInput, AttributionOutput]):
    """Tool for performance attribution analysis.

    Breaks down portfolio performance by sector and factor exposures.
    """

    name = "attribution_analysis"
    description = (
        "Performs performance attribution analysis. Breaks down returns by sector "
        "and calculates selection, allocation, and interaction effects."
    )

    @property
    def input_schema(self) -> type[AttributionInput]:
        return AttributionInput

    @property
    def output_schema(self) -> type[AttributionOutput]:
        return AttributionOutput

    async def _execute_real(self, input_data: AttributionInput) -> AttributionOutput:
        """Calculate attribution from real or provided data."""
        return await self._calculate_attribution(input_data)

    async def _execute_mock(self, input_data: AttributionInput) -> AttributionOutput:
        """Calculate attribution with mock data."""
        return await self._calculate_attribution(input_data)

    async def _calculate_attribution(self, input_data: AttributionInput) -> AttributionOutput:
        """Core attribution calculation logic.

        Args:
            input_data: Input with portfolio info and sector mapping.

        Returns:
            Attribution analysis results.
        """
        symbols = input_data.symbols
        weights = input_data.weights or [1.0 / len(symbols)] * len(symbols)
        sector_mapping = input_data.sector_mapping or DEFAULT_SECTOR_MAPPING

        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        # Get returns
        all_returns = input_data.returns or _generate_mock_returns(symbols)

        # Calculate per-symbol returns
        symbol_returns = {}
        for i, symbol in enumerate(symbols):
            avg_return = _mean(all_returns[i]) * 252  # Annualized
            symbol_returns[symbol] = avg_return

        # Calculate sector attribution
        sector_weights: dict[str, float] = {}
        sector_returns: dict[str, list[float]] = {}

        for i, symbol in enumerate(symbols):
            sector = sector_mapping.get(symbol, "Other")
            if sector not in sector_weights:
                sector_weights[sector] = 0.0
                sector_returns[sector] = []
            sector_weights[sector] += weights[i]
            sector_returns[sector].append(symbol_returns[symbol] * weights[i])

        sector_attribution = {}
        for sector in sector_weights:
            if sector_weights[sector] > 0:
                # Contribution = weighted return from sector
                sector_attribution[sector] = round(
                    sum(sector_returns[sector]), 4
                )

        # Calculate factor attribution (simplified momentum/value proxy)
        factor_attribution = {}

        # Use hash-based deterministic factor scores
        momentum_contrib = 0.0
        value_contrib = 0.0
        quality_contrib = 0.0

        for i, symbol in enumerate(symbols):
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            random.seed(seed)

            # Simplified factor exposures
            momentum_exposure = random.uniform(-0.3, 0.5)
            value_exposure = random.uniform(-0.2, 0.4)
            quality_exposure = random.uniform(0.1, 0.6)

            momentum_contrib += weights[i] * momentum_exposure * symbol_returns[symbol]
            value_contrib += weights[i] * value_exposure * symbol_returns[symbol]
            quality_contrib += weights[i] * quality_exposure * symbol_returns[symbol]

        factor_attribution["Momentum"] = round(momentum_contrib, 4)
        factor_attribution["Value"] = round(value_contrib, 4)
        factor_attribution["Quality"] = round(quality_contrib, 4)

        # Calculate Brinson attribution effects (simplified)
        # Using mock benchmark weights
        total_portfolio_return = sum(
            weights[i] * symbol_returns[symbol]
            for i, symbol in enumerate(symbols)
        )

        # Simplified benchmark return
        benchmark_return = 0.08  # Assume 8% benchmark return

        # Selection effect: stock picking within sectors
        selection_effect = total_portfolio_return * 0.4  # Simplified

        # Allocation effect: sector weighting
        allocation_effect = (total_portfolio_return - benchmark_return) * 0.3

        # Interaction effect: remainder
        total_active = total_portfolio_return - benchmark_return
        interaction_effect = total_active - selection_effect - allocation_effect

        return AttributionOutput(
            result=AttributionResult(
                sector_attribution=sector_attribution,
                factor_attribution=factor_attribution,
                selection_effect=round(selection_effect, 4),
                allocation_effect=round(allocation_effect, 4),
                interaction_effect=round(interaction_effect, 4),
                total_active_return=round(total_active, 4),
            )
        )
