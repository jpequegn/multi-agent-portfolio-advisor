"""Data source routing with fallback chain and rate limiting.

This module provides:
- RateLimitedQueue: Request queue that respects rate limits
- DataSourceRouter: Orchestrates the fallback chain (Polygon → Yahoo → Cache → Mock)
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, TypeVar

import structlog

from src.cache.manager import CacheKeyBuilder, CacheManager, CacheType
from src.data.models import (
    Bar,
    CompanyInfo,
    DataSource,
    MarketDataResult,
    NewsArticle,
    Quote,
)
from src.data.polygon import (
    PolygonAPIError,
    PolygonAuthError,
    PolygonClient,
    PolygonRateLimitError,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RateLimitedQueue:
    """Queue that delays requests to respect rate limits.

    Uses a semaphore-based approach where tokens are released
    after a fixed interval to maintain the request rate.

    Example:
        queue = RateLimitedQueue(rate_limit=5)  # 5 req/min

        # This will wait if rate limit is reached
        result = await queue.submit(fetch_data())
    """

    def __init__(self, rate_limit: int = 5, window_seconds: int = 60) -> None:
        """Initialize the rate-limited queue.

        Args:
            rate_limit: Maximum requests per window.
            window_seconds: Time window in seconds.
        """
        self._rate_limit = rate_limit
        self._window_seconds = window_seconds
        self._refill_interval = window_seconds / rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit)
        self._logger = logger.bind(component="rate_limited_queue")
        self._pending_count = 0

    @property
    def refill_interval(self) -> float:
        """Seconds between token refills."""
        return self._refill_interval

    @property
    def pending_requests(self) -> int:
        """Number of requests waiting in queue."""
        return self._pending_count

    async def submit(self, coro: Awaitable[T]) -> T:
        """Submit a request to the rate-limited queue.

        Waits for an available slot before executing the coroutine.
        Releases the slot after the refill interval.

        Args:
            coro: Coroutine to execute.

        Returns:
            Result of the coroutine.
        """
        self._pending_count += 1
        self._logger.debug(
            "queue_submit",
            pending=self._pending_count,
            refill_interval=self._refill_interval,
        )

        await self._semaphore.acquire()
        self._pending_count -= 1

        try:
            result = await coro
            return result
        finally:
            # Schedule token release after refill interval
            asyncio.get_event_loop().call_later(
                self._refill_interval,
                self._semaphore.release,
            )

    def reset(self) -> None:
        """Reset the queue (for testing)."""
        self._semaphore = asyncio.Semaphore(self._rate_limit)
        self._pending_count = 0


class DataSourceRouter:
    """Routes data requests through the fallback chain.

    Fallback order: Polygon → Yahoo → Cache → Mock

    Handles:
    - Rate limiting for Polygon API
    - Automatic fallback on errors
    - Caching of successful responses
    - Metrics and logging

    Example:
        router = DataSourceRouter(polygon_client, cache_manager)
        result = await router.get_quote("AAPL")
        print(f"Price: {result.quote.price} (source: {result.source})")
    """

    # Cache TTLs matching the design
    QUOTE_TTL = 60  # 1 minute
    BARS_TTL = 3600  # 1 hour
    COMPANY_INFO_TTL = 86400  # 24 hours
    NEWS_TTL = 900  # 15 minutes

    def __init__(
        self,
        polygon: PolygonClient | None = None,
        cache: CacheManager | None = None,
        rate_limit: int = 5,
    ) -> None:
        """Initialize the data source router.

        Args:
            polygon: Polygon.io client instance.
            cache: Cache manager instance.
            rate_limit: Polygon API rate limit (requests per minute).
        """
        self._polygon = polygon or PolygonClient()
        self._cache = cache
        self._queue = RateLimitedQueue(rate_limit=rate_limit)
        self._logger = logger.bind(component="data_source_router")

        # Track fallback stats
        self._stats = {
            "polygon_success": 0,
            "polygon_rate_limited": 0,
            "yahoo_fallback": 0,
            "cache_fallback": 0,
            "mock_fallback": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Get fallback statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset fallback statistics."""
        for key in self._stats:
            self._stats[key] = 0

    async def get_quote(self, symbol: str) -> MarketDataResult:
        """Get quote with full fallback chain.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            MarketDataResult with quote data.
        """
        symbol = symbol.upper()
        result = MarketDataResult(symbol=symbol)

        # Try Polygon first (through rate-limited queue)
        try:
            if self._polygon.is_configured:
                quote = await self._queue.submit(self._polygon.get_quote(symbol))
                result.quote = quote
                result.source = DataSource.POLYGON
                self._stats["polygon_success"] += 1

                # Cache the result
                if self._cache:
                    cache_key = CacheKeyBuilder.market_data(symbol, "quote")
                    await self._cache.set(
                        cache_key,
                        quote.model_dump(mode="json"),
                        ttl=self.QUOTE_TTL,
                    )

                self._logger.info("quote_fetched", symbol=symbol, source="polygon")
                return result

        except PolygonRateLimitError:
            self._stats["polygon_rate_limited"] += 1
            self._logger.warning("polygon_rate_limited", symbol=symbol)
            result.errors.append("Polygon rate limited")

        except (PolygonAuthError, PolygonAPIError) as e:
            self._logger.warning("polygon_error", symbol=symbol, error=str(e))
            result.errors.append(f"Polygon error: {e}")

        except Exception as e:
            self._logger.error("polygon_unexpected_error", symbol=symbol, error=str(e))
            result.errors.append(f"Unexpected error: {e}")

        # Try Yahoo fallback
        try:
            quote = await self._fetch_yahoo_quote(symbol)
            if quote:
                result.quote = quote
                result.source = DataSource.YAHOO
                result.fallback_used = True
                self._stats["yahoo_fallback"] += 1

                # Cache the result
                if self._cache:
                    cache_key = CacheKeyBuilder.market_data(symbol, "quote")
                    await self._cache.set(
                        cache_key,
                        quote.model_dump(mode="json"),
                        ttl=self.QUOTE_TTL,
                    )

                self._logger.info("quote_fetched", symbol=symbol, source="yahoo")
                return result

        except Exception as e:
            self._logger.warning("yahoo_error", symbol=symbol, error=str(e))
            result.errors.append(f"Yahoo error: {e}")

        # Try cache fallback
        if self._cache:
            try:
                cache_key = CacheKeyBuilder.market_data(symbol, "quote")
                cached = await self._cache.get(cache_key)
                if cached:
                    result.quote = Quote(**cached)
                    result.source = DataSource.CACHE
                    result.cached = True
                    result.fallback_used = True
                    self._stats["cache_fallback"] += 1
                    self._logger.info("quote_fetched", symbol=symbol, source="cache")
                    return result

            except Exception as e:
                self._logger.warning("cache_error", symbol=symbol, error=str(e))
                result.errors.append(f"Cache error: {e}")

        # Mock fallback (last resort)
        result.quote = self._get_mock_quote(symbol)
        result.source = DataSource.MOCK
        result.fallback_used = True
        self._stats["mock_fallback"] += 1
        self._logger.info("quote_fetched", symbol=symbol, source="mock")
        return result

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "day",
        limit: int = 30,
    ) -> MarketDataResult:
        """Get historical bars with fallback chain.

        Args:
            symbol: Stock ticker symbol.
            timeframe: Bar timeframe ("day", "week", "month").
            limit: Number of bars to fetch.

        Returns:
            MarketDataResult with bars data.
        """
        symbol = symbol.upper()
        result = MarketDataResult(symbol=symbol)
        cache_key = CacheKeyBuilder.market_data(symbol, f"bars_{timeframe}")

        # Try Polygon
        try:
            if self._polygon.is_configured:
                bars = await self._queue.submit(
                    self._polygon.get_bars(symbol, timeframe, limit)
                )
                result.bars = bars
                result.source = DataSource.POLYGON
                self._stats["polygon_success"] += 1

                if self._cache:
                    await self._cache.set(
                        cache_key,
                        [b.model_dump(mode="json") for b in bars],
                        ttl=self.BARS_TTL,
                    )

                return result

        except (PolygonRateLimitError, PolygonAuthError, PolygonAPIError) as e:
            self._logger.warning("polygon_bars_error", symbol=symbol, error=str(e))
            result.errors.append(str(e))

        # Try cache
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                result.bars = [Bar(**b) for b in cached]
                result.source = DataSource.CACHE
                result.cached = True
                result.fallback_used = True
                self._stats["cache_fallback"] += 1
                return result

        # Mock fallback
        result.bars = self._get_mock_bars(symbol, limit)
        result.source = DataSource.MOCK
        result.fallback_used = True
        self._stats["mock_fallback"] += 1
        return result

    async def get_company_info(self, symbol: str) -> MarketDataResult:
        """Get company info with fallback chain.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            MarketDataResult with company info.
        """
        symbol = symbol.upper()
        result = MarketDataResult(symbol=symbol)
        cache_key = CacheKeyBuilder.custom("company", symbol)

        # Try Polygon
        try:
            if self._polygon.is_configured:
                info = await self._queue.submit(self._polygon.get_company_info(symbol))
                result.company_info = info
                result.source = DataSource.POLYGON
                self._stats["polygon_success"] += 1

                if self._cache:
                    await self._cache.set(
                        cache_key,
                        info.model_dump(mode="json"),
                        ttl=self.COMPANY_INFO_TTL,
                    )

                return result

        except (PolygonRateLimitError, PolygonAuthError, PolygonAPIError) as e:
            self._logger.warning("polygon_company_error", symbol=symbol, error=str(e))
            result.errors.append(str(e))

        # Try cache
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                result.company_info = CompanyInfo(**cached)
                result.source = DataSource.CACHE
                result.cached = True
                result.fallback_used = True
                self._stats["cache_fallback"] += 1
                return result

        # Mock fallback
        result.company_info = self._get_mock_company_info(symbol)
        result.source = DataSource.MOCK
        result.fallback_used = True
        self._stats["mock_fallback"] += 1
        return result

    async def search_news(
        self,
        symbol: str | None = None,
        limit: int = 10,
    ) -> MarketDataResult:
        """Search news with fallback chain.

        Args:
            symbol: Optional stock symbol to filter news.
            limit: Maximum number of articles.

        Returns:
            MarketDataResult with news articles.
        """
        result = MarketDataResult(symbol=symbol or "MARKET")
        cache_key = CacheKeyBuilder.custom("news", symbol or "market")

        # Try Polygon
        try:
            if self._polygon.is_configured:
                articles = await self._queue.submit(
                    self._polygon.search_news(symbol, limit)
                )
                result.news = articles
                result.source = DataSource.POLYGON
                self._stats["polygon_success"] += 1

                if self._cache:
                    await self._cache.set(
                        cache_key,
                        [a.model_dump(mode="json") for a in articles],
                        ttl=self.NEWS_TTL,
                    )

                return result

        except (PolygonRateLimitError, PolygonAuthError, PolygonAPIError) as e:
            self._logger.warning("polygon_news_error", error=str(e))
            result.errors.append(str(e))

        # Try cache
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                result.news = [NewsArticle(**a) for a in cached]
                result.source = DataSource.CACHE
                result.cached = True
                result.fallback_used = True
                self._stats["cache_fallback"] += 1
                return result

        # Mock fallback
        result.news = self._get_mock_news(symbol, limit)
        result.source = DataSource.MOCK
        result.fallback_used = True
        self._stats["mock_fallback"] += 1
        return result

    # =========================================================================
    # Yahoo Finance Fallback
    # =========================================================================

    async def _fetch_yahoo_quote(self, symbol: str) -> Quote | None:
        """Fetch quote from Yahoo Finance as fallback.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote or None if unavailable.
        """
        try:
            import yfinance as yf

            ticker = await asyncio.to_thread(lambda: yf.Ticker(symbol))
            info = await asyncio.to_thread(lambda: ticker.info)

            if not info or info.get("regularMarketPrice") is None:
                return None

            price = info.get("regularMarketPrice") or info.get("currentPrice", 0.0)
            previous_close = info.get("previousClose", price)

            if previous_close and previous_close > 0:
                change_percent = ((price - previous_close) / previous_close) * 100
            else:
                change_percent = 0.0

            return Quote(
                symbol=symbol,
                price=float(price),
                change_percent=round(change_percent, 2),
                volume=int(info.get("regularMarketVolume", 0)),
                open=info.get("regularMarketOpen"),
                high=info.get("regularMarketDayHigh"),
                low=info.get("regularMarketDayLow"),
                previous_close=previous_close,
                source=DataSource.YAHOO,
            )

        except Exception as e:
            self._logger.error("yahoo_fetch_error", symbol=symbol, error=str(e))
            return None

    # =========================================================================
    # Mock Data Fallback
    # =========================================================================

    def _get_mock_quote(self, symbol: str) -> Quote:
        """Generate mock quote data."""
        mock_prices = {
            "AAPL": 185.50,
            "GOOGL": 140.25,
            "MSFT": 378.90,
            "AMZN": 178.25,
            "NVDA": 495.50,
            "TSLA": 248.50,
        }
        price = mock_prices.get(symbol, 100.0)

        return Quote(
            symbol=symbol,
            price=price,
            change_percent=0.0,
            volume=1_000_000,
            open=price * 0.99,
            high=price * 1.01,
            low=price * 0.98,
            previous_close=price,
            source=DataSource.MOCK,
        )

    def _get_mock_bars(self, symbol: str, limit: int) -> list[Bar]:
        """Generate mock historical bars."""
        bars = []
        base_price = 100.0
        now = datetime.now()

        for i in range(limit):
            bars.append(
                Bar(
                    symbol=symbol,
                    open=base_price * (1 + i * 0.001),
                    high=base_price * (1 + i * 0.002),
                    low=base_price * (1 - i * 0.001),
                    close=base_price * (1 + i * 0.0015),
                    volume=1_000_000 + i * 10000,
                    timestamp=now.replace(day=max(1, now.day - i)),
                )
            )

        return bars

    def _get_mock_company_info(self, symbol: str) -> CompanyInfo:
        """Generate mock company info."""
        return CompanyInfo(
            symbol=symbol,
            name=f"{symbol} Inc.",
            description=f"Mock company data for {symbol}",
            sector="Technology",
            industry="Software",
            market_cap=100_000_000_000,
            exchange="NASDAQ",
            source=DataSource.MOCK,
        )

    def _get_mock_news(self, symbol: str | None, limit: int) -> list[NewsArticle]:
        """Generate mock news articles."""
        articles = []
        now = datetime.now()

        for i in range(min(limit, 5)):
            articles.append(
                NewsArticle(
                    id=f"mock-{i}",
                    title=f"Market Update: {symbol or 'Markets'} Analysis",
                    author="Mock Author",
                    publisher="Mock Financial News",
                    url=f"https://example.com/news/{i}",
                    summary=f"Mock news summary for {symbol or 'the market'}.",
                    symbols=[symbol] if symbol else [],
                    sentiment="neutral",
                    published_at=now,
                    source=DataSource.MOCK,
                )
            )

        return articles

    async def close(self) -> None:
        """Close underlying clients."""
        if self._polygon:
            await self._polygon.close()
