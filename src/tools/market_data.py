"""Market Data Tool for fetching real-time stock data.

This module implements the MarketDataTool that fetches market data from
Yahoo Finance with Redis caching and mock fallback support.
"""

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import redis.asyncio as redis
import structlog
import yfinance as yf
from pydantic import Field

from src.tools.base import BaseTool, ToolInput, ToolOutput

if TYPE_CHECKING:
    from redis.asyncio.client import Redis

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
    source: Literal["real", "mock", "cache"]


# ============================================================================
# Mock Data
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
# Redis Cache
# ============================================================================


class MarketDataCache:
    """Redis cache for market data with configurable TTL."""

    DEFAULT_TTL = 300  # 5 minutes

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = DEFAULT_TTL,
    ) -> None:
        """Initialize the cache.

        Args:
            redis_url: Redis connection URL.
            ttl: Cache TTL in seconds.
        """
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: "Redis[str] | None" = None  # noqa: UP037
        self._logger = logger.bind(component="market_data_cache")

    async def _get_client(self) -> "Redis[str]":
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _cache_key(self, symbol: str, data_type: str) -> str:
        """Generate cache key for a symbol."""
        return f"market_data:{symbol.upper()}:{data_type}"

    async def get(self, symbol: str, data_type: str = "quote") -> MarketDataOutput | None:
        """Get cached market data.

        Args:
            symbol: Stock symbol.
            data_type: Type of data.

        Returns:
            Cached data or None if not found/expired.
        """
        try:
            client = await self._get_client()
            key = self._cache_key(symbol, data_type)
            data = await client.get(key)

            if data:
                self._logger.debug("cache_hit", symbol=symbol, data_type=data_type)
                parsed = json.loads(data)
                return MarketDataOutput(**parsed)

            self._logger.debug("cache_miss", symbol=symbol, data_type=data_type)
            return None

        except redis.RedisError as e:
            self._logger.warning("cache_get_error", error=str(e))
            return None

    async def set(
        self,
        symbol: str,
        data_type: str,
        data: MarketDataOutput,
    ) -> bool:
        """Cache market data.

        Args:
            symbol: Stock symbol.
            data_type: Type of data.
            data: Market data to cache.

        Returns:
            True if cached successfully.
        """
        try:
            client = await self._get_client()
            key = self._cache_key(symbol, data_type)
            await client.setex(key, self._ttl, data.model_dump_json())
            self._logger.debug("cache_set", symbol=symbol, data_type=data_type, ttl=self._ttl)
            return True

        except redis.RedisError as e:
            self._logger.warning("cache_set_error", error=str(e))
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# ============================================================================
# Market Data Tool
# ============================================================================


class MarketDataTool(BaseTool[MarketDataInput, MarketDataOutput]):
    """Tool for fetching market data from Yahoo Finance.

    Features:
    - Real-time data via yfinance
    - Redis caching with configurable TTL
    - Mock data fallback for testing/resilience
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
        cache: MarketDataCache | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the MarketDataTool.

        Args:
            use_mock: Whether to use mock data instead of real API.
            fallback_to_mock: Whether to fall back to mock on API failure.
            cache: Optional custom cache instance.
            use_cache: Whether to use caching (default True).
        """
        super().__init__(use_mock=use_mock, fallback_to_mock=fallback_to_mock)
        self._cache = cache
        self._use_cache = use_cache

    @property
    def input_schema(self) -> type[MarketDataInput]:
        return MarketDataInput

    @property
    def output_schema(self) -> type[MarketDataOutput]:
        return MarketDataOutput

    async def _execute_real(self, input_data: MarketDataInput) -> MarketDataOutput:
        """Fetch real market data from Yahoo Finance.

        Args:
            input_data: Validated input with symbol and data_type.

        Returns:
            Market data output.

        Raises:
            ValueError: If the symbol is invalid or data unavailable.
        """
        symbol = input_data.symbol.upper()

        # Check cache first
        if self._use_cache and self._cache:
            cached = await self._cache.get(symbol, input_data.data_type)
            if cached:
                return cached

        # Fetch from yfinance (runs sync in thread pool)
        self._logger.info("fetching_market_data", symbol=symbol)
        ticker_data = await asyncio.to_thread(self._fetch_yfinance_data, symbol)

        if ticker_data is None:
            raise ValueError(f"Could not fetch data for symbol: {symbol}")

        output = MarketDataOutput(
            symbol=symbol,
            price=ticker_data["price"],
            change_percent=ticker_data["change_percent"],
            volume=ticker_data["volume"],
            market_cap=ticker_data.get("market_cap"),
            pe_ratio=ticker_data.get("pe_ratio"),
            dividend_yield=ticker_data.get("dividend_yield"),
            fifty_two_week_high=ticker_data.get("fifty_two_week_high"),
            fifty_two_week_low=ticker_data.get("fifty_two_week_low"),
            timestamp=datetime.now().isoformat(),
            source="real",
        )

        # Cache the result
        if self._use_cache and self._cache:
            await self._cache.set(symbol, input_data.data_type, output)

        return output

    def _fetch_yfinance_data(self, symbol: str) -> dict[str, Any] | None:
        """Fetch data from yfinance (sync method).

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dict with market data or None if unavailable.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got valid data
            if not info or info.get("regularMarketPrice") is None:
                self._logger.warning("invalid_ticker_data", symbol=symbol)
                return None

            # Extract current price
            price = info.get("regularMarketPrice") or info.get("currentPrice", 0.0)

            # Calculate change percent
            previous_close = info.get("previousClose", price)
            if previous_close and previous_close > 0:
                change_percent = ((price - previous_close) / previous_close) * 100
            else:
                change_percent = 0.0

            return {
                "price": float(price),
                "change_percent": round(change_percent, 2),
                "volume": int(info.get("regularMarketVolume", 0)),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": self._to_percent(info.get("dividendYield")),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            }

        except Exception as e:
            self._logger.error("yfinance_fetch_error", symbol=symbol, error=str(e))
            return None

    @staticmethod
    def _to_percent(value: float | None) -> float | None:
        """Convert decimal to percentage (e.g., 0.005 -> 0.5)."""
        if value is None:
            return None
        return round(value * 100, 2)

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
