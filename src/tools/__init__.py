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

__all__ = [
    "BaseTool",
    "MarketDataCache",
    "MarketDataInput",
    "MarketDataOutput",
    "MarketDataTool",
    "NewsItem",
    "NewsSearchInput",
    "NewsSearchOutput",
    "NewsSearchTool",
    "ToolExecutionError",
    "ToolInput",
    "ToolOutput",
    "ToolRegistry",
    "default_registry",
]
