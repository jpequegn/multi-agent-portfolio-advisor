"""Tools available to agents.

This module contains tools for:
- Market data fetching
- News search
- Risk calculations
- Trade generation
"""

from src.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
    default_registry,
)
from src.tools.market_data import (
    MarketDataCache,
    MarketDataInput,
    MarketDataOutput,
    MarketDataTool,
)
from src.tools.news_search import (
    NewsItem,
    NewsSearchInput,
    NewsSearchOutput,
    NewsSearchTool,
)
from src.tools.risk_metrics import (
    AttributionResult,
    AttributionTool,
    BenchmarkComparison,
    BenchmarkComparisonTool,
    CorrelationResult,
    CorrelationTool,
    RiskMetrics,
    RiskMetricsTool,
)
from src.tools.trade_generator import (
    ComplianceResult,
    ComplianceTool,
    ExecutionCosts,
    ExecutionCostTool,
    RebalancingTool,
    TaxImpact,
    TaxImpactTool,
    Trade,
    TradeList,
)

__all__ = [
    "AttributionResult",
    "AttributionTool",
    "BaseTool",
    "BenchmarkComparison",
    "BenchmarkComparisonTool",
    "ComplianceResult",
    "ComplianceTool",
    "CorrelationResult",
    "CorrelationTool",
    "ExecutionCosts",
    "ExecutionCostTool",
    "MarketDataCache",
    "MarketDataInput",
    "MarketDataOutput",
    "MarketDataTool",
    "NewsItem",
    "NewsSearchInput",
    "NewsSearchOutput",
    "NewsSearchTool",
    "RebalancingTool",
    "RiskMetrics",
    "RiskMetricsTool",
    "TaxImpact",
    "TaxImpactTool",
    "ToolExecutionError",
    "ToolInput",
    "ToolOutput",
    "ToolRegistry",
    "Trade",
    "TradeList",
    "default_registry",
]
