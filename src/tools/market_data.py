"""Market Data Tool for fetching real-time stock data.

This module implements the MarketDataTool that fetches market data
using the DataSourceRouter (Polygon → Yahoo → Cache → Mock fallback chain).
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import Field

from src.cache.manager import CacheManager
from src.data.models import DataSource
from src.data.polygon import PolygonClient
from src.data.router import DataSourceRouter
from src.tools.base import BaseTool, ToolInput, ToolOutput

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


# ============================================================================
# Input/Output Schemas
# ============================================================================


class MarketDataInput(ToolInput):
    """Input for market data tool."""

    symbol: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    data_type: Literal["quote", "history", "fundamentals"] = Field(
        default="quote",
        description="Type of data to fetch",
    )


class MarketDataOutput(ToolOutput):
    """Output from market data tool."""

    symbol: str
    price: float
    change_percent: float
    volume: int
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    timestamp: str
    source: Literal["polygon", "yahoo", "cache", "mock"]


# ============================================================================
# Mock Data (used when all fallbacks fail)
# ============================================================================


MOCK_STOCKS: dict[str, dict[str, Any]] = {
    "AAPL": {
        "price": 185.50,
        "change_percent": 1.2,
        "volume": 50_000_000,
        "market_cap": 2.9e12,
        "pe_ratio": 28.5,
        "dividend_yield": 0.5,
        "fifty_two_week_high": 199.62,
        "fifty_two_week_low": 164.08,
    },
    "GOOGL": {
        "price": 140.25,
        "change_percent": -0.5,
        "volume": 20_000_000,
        "market_cap": 1.8e12,
        "pe_ratio": 25.2,
        "dividend_yield": None,
        "fifty_two_week_high": 153.78,
        "fifty_two_week_low": 121.46,
    },
    "MSFT": {
        "price": 378.90,
        "change_percent": 0.82,
        "volume": 18_000_000,
        "market_cap": 2.8e12,
        "pe_ratio": 35.1,
        "dividend_yield": 0.8,
        "fifty_two_week_high": 420.82,
        "fifty_two_week_low": 362.90,
    },
    "AMZN": {
        "price": 178.25,
        "change_percent": 1.5,
        "volume": 35_000_000,
        "market_cap": 1.9e12,
        "pe_ratio": 60.5,
        "dividend_yield": None,
        "fifty_two_week_high": 201.20,
        "fifty_two_week_low": 151.61,
    },
    "NVDA": {
        "price": 495.50,
        "change_percent": 2.8,
        "volume": 45_000_000,
        "market_cap": 1.2e12,
        "pe_ratio": 65.2,
        "dividend_yield": 0.03,
        "fifty_two_week_high": 505.48,
        "fifty_two_week_low": 222.97,
    },
    "TSLA": {
        "price": 248.50,
        "change_percent": -1.2,
        "volume": 100_000_000,
        "market_cap": 790e9,
        "pe_ratio": 72.5,
        "dividend_yield": None,
        "fifty_two_week_high": 299.29,
        "fifty_two_week_low": 152.37,
    },
}

DEFAULT_MOCK: dict[str, Any] = {
    "price": 100.0,
    "change_percent": 0.0,
    "volume": 1_000_000,
    "market_cap": 10e9,
    "pe_ratio": 20.0,
    "dividend_yield": None,
    "fifty_two_week_high": 120.0,
    "fifty_two_week_low": 80.0,
}


# ============================================================================
# Market Data Tool
# ============================================================================


class MarketDataTool(BaseTool[MarketDataInput, MarketDataOutput]):
    """Tool for fetching market data with intelligent fallback.

    Uses the DataSourceRouter to fetch data through the fallback chain:
    Polygon.io → Yahoo Finance → Cache → Mock

    Features:
    - Real-time data via Polygon.io (primary) or Yahoo Finance (fallback)
    - Rate-limited queue for Polygon's 5 req/min free tier
    - Redis caching with configurable TTL
    - Automatic fallback through the chain
    - Execution tracing
    """

    name = "get_market_data"
    description = (
        "Fetches current market data for a stock symbol including price, "
        "volume, and key metrics like P/E ratio and market cap"
    )

    def __init__(
        self,
        use_mock: bool | None = None,
        fallback_to_mock: bool | None = None,
        router: DataSourceRouter | None = None,
        polygon_client: PolygonClient | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        """Initialize the MarketDataTool.

        Args:
            use_mock: Whether to use mock data instead of real API.
            fallback_to_mock: Whether to fall back to mock on API failure.
            router: Optional DataSourceRouter instance. If not provided,
                creates one with the polygon_client and cache_manager.
            polygon_client: Optional PolygonClient for Polygon.io API.
            cache_manager: Optional CacheManager for Redis caching.
        """
        super().__init__(use_mock=use_mock, fallback_to_mock=fallback_to_mock)

        # Create or use provided router
        if router:
            self._router = router
        else:
            self._router = DataSourceRouter(
                polygon=polygon_client or PolygonClient(),
                cache=cache_manager,
            )

    @property
    def input_schema(self) -> type[MarketDataInput]:
        return MarketDataInput

    @property
    def output_schema(self) -> type[MarketDataOutput]:
        return MarketDataOutput

    @property
    def router(self) -> DataSourceRouter:
        """Get the underlying data source router."""
        return self._router

    async def _execute_real(self, input_data: MarketDataInput) -> MarketDataOutput:
        """Fetch market data through the DataSourceRouter.

        The router handles the fallback chain:
        Polygon → Yahoo → Cache → Mock

        Args:
            input_data: Validated input with symbol and data_type.

        Returns:
            Market data output.
        """
        symbol = input_data.symbol.upper()
        self._logger.info("fetching_market_data", symbol=symbol)

        # Use router to get quote (handles all fallbacks)
        result = await self._router.get_quote(symbol)

        if result.quote is None:
            # This shouldn't happen as mock is the last fallback,
            # but handle it gracefully
            raise ValueError(f"Could not fetch data for symbol: {symbol}")

        # Map DataSource enum to string for backward compatibility
        source_map = {
            DataSource.POLYGON: "polygon",
            DataSource.YAHOO: "yahoo",
            DataSource.CACHE: "cache",
            DataSource.MOCK: "mock",
        }

        return MarketDataOutput(
            symbol=symbol,
            price=result.quote.price,
            change_percent=result.quote.change_percent,
            volume=result.quote.volume,
            market_cap=None,  # Quote doesn't include market_cap
            pe_ratio=None,  # Quote doesn't include pe_ratio
            dividend_yield=None,  # Quote doesn't include dividend_yield
            fifty_two_week_high=result.quote.high,
            fifty_two_week_low=result.quote.low,
            timestamp=result.quote.timestamp.isoformat(),
            source=source_map.get(result.source, "mock"),
        )

    async def _execute_mock(self, input_data: MarketDataInput) -> MarketDataOutput:
        """Return mock market data.

        Args:
            input_data: Validated input with symbol.

        Returns:
            Mock market data output.
        """
        symbol = input_data.symbol.upper()
        data = MOCK_STOCKS.get(symbol, DEFAULT_MOCK)

        return MarketDataOutput(
            symbol=symbol,
            price=data["price"],
            change_percent=data["change_percent"],
            volume=data["volume"],
            market_cap=data.get("market_cap"),
            pe_ratio=data.get("pe_ratio"),
            dividend_yield=data.get("dividend_yield"),
            fifty_two_week_high=data.get("fifty_two_week_high"),
            fifty_two_week_low=data.get("fifty_two_week_low"),
            timestamp=datetime.now().isoformat(),
            source="mock",
        )

    async def close(self) -> None:
        """Close underlying connections."""
        if self._router:
            await self._router.close()
