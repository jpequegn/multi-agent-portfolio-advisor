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

__all__ = [
    "AttributionResult",
    "AttributionTool",
    "BaseTool",
    "BenchmarkComparison",
    "BenchmarkComparisonTool",
    "CorrelationResult",
    "CorrelationTool",
    "MarketDataCache",
    "MarketDataInput",
    "MarketDataOutput",
    "MarketDataTool",
    "NewsItem",
    "NewsSearchInput",
    "NewsSearchOutput",
    "NewsSearchTool",
    "RiskMetrics",
    "RiskMetricsTool",
    "ToolExecutionError",
    "ToolInput",
    "ToolOutput",
    "ToolRegistry",
    "default_registry",
]
