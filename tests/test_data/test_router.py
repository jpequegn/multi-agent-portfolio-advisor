"""Tests for the DataSourceRouter and RateLimitedQueue."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import Bar, CompanyInfo, DataSource, NewsArticle, Quote
from src.data.polygon import PolygonAPIError, PolygonAuthError, PolygonRateLimitError
from src.data.router import DataSourceRouter, RateLimitedQueue


class TestRateLimitedQueue:
    """Tests for RateLimitedQueue."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        queue = RateLimitedQueue()
        assert queue.refill_interval == 12.0  # 60/5 = 12 seconds
        assert queue.pending_requests == 0

    def test_init_custom_rate(self) -> None:
        """Test initialization with custom rate."""
        queue = RateLimitedQueue(rate_limit=10, window_seconds=60)
        assert queue.refill_interval == 6.0  # 60/10 = 6 seconds

    @pytest.mark.asyncio
    async def test_submit_single_request(self) -> None:
        """Test submitting a single request."""
        queue = RateLimitedQueue(rate_limit=5)

        async def mock_request() -> str:
            return "result"

        result = await queue.submit(mock_request())
        assert result == "result"

    @pytest.mark.asyncio
    async def test_submit_tracks_pending(self) -> None:
        """Test that pending count is tracked correctly."""
        queue = RateLimitedQueue(rate_limit=2)

        async def slow_request() -> str:
            await asyncio.sleep(0.1)
            return "done"

        # Both should complete without waiting
        results = await asyncio.gather(
            queue.submit(slow_request()),
            queue.submit(slow_request()),
        )
        assert results == ["done", "done"]

    def test_reset(self) -> None:
        """Test queue reset."""
        queue = RateLimitedQueue(rate_limit=5)
        queue._pending_count = 3
        queue.reset()
        assert queue.pending_requests == 0


class TestDataSourceRouter:
    """Tests for DataSourceRouter."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        router = DataSourceRouter()
        assert router.stats["polygon_success"] == 0
        assert router.stats["mock_fallback"] == 0

    def test_init_with_polygon(self) -> None:
        """Test initialization with Polygon client."""
        mock_polygon = MagicMock()
        router = DataSourceRouter(polygon=mock_polygon)
        assert router._polygon == mock_polygon

    def test_stats_reset(self) -> None:
        """Test stats reset."""
        router = DataSourceRouter()
        router._stats["polygon_success"] = 5
        router.reset_stats()
        assert router.stats["polygon_success"] == 0

    @pytest.mark.asyncio
    async def test_get_quote_polygon_success(self) -> None:
        """Test successful quote fetch from Polygon."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        mock_quote = Quote(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            source=DataSource.POLYGON,
        )

        async def mock_get_quote(symbol: str) -> Quote:
            return mock_quote

        mock_polygon.get_quote = mock_get_quote

        router = DataSourceRouter(polygon=mock_polygon)
        result = await router.get_quote("AAPL")

        assert result.quote is not None
        assert result.quote.symbol == "AAPL"
        assert result.quote.price == 185.50
        assert result.source == DataSource.POLYGON
        assert not result.fallback_used
        assert router.stats["polygon_success"] == 1

    @pytest.mark.asyncio
    async def test_get_quote_yahoo_fallback(self) -> None:
        """Test quote fetch falls back to Yahoo when Polygon not configured."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = False

        mock_yahoo_quote = Quote(
            symbol="AAPL",
            price=186.00,
            change_percent=0.5,
            volume=45_000_000,
            source=DataSource.YAHOO,
        )

        router = DataSourceRouter(polygon=mock_polygon)

        # Mock Yahoo to return a quote
        async def mock_yahoo(symbol: str) -> Quote:
            return mock_yahoo_quote

        with patch.object(router, "_fetch_yahoo_quote", side_effect=mock_yahoo):
            result = await router.get_quote("AAPL")

            assert result.quote is not None
            assert result.source == DataSource.YAHOO
            assert result.quote.price == 186.00
            assert result.fallback_used
            assert router.stats["yahoo_fallback"] == 1

    @pytest.mark.asyncio
    async def test_get_quote_polygon_not_configured(self) -> None:
        """Test quote fetch when Polygon is not configured and Yahoo fails."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = False

        router = DataSourceRouter(polygon=mock_polygon)

        # Mock Yahoo to fail so we hit mock fallback
        with patch.object(router, "_fetch_yahoo_quote", return_value=None):
            result = await router.get_quote("AAPL")

            # Should fallback to mock
            assert result.quote is not None
            assert result.source == DataSource.MOCK
            assert result.fallback_used
            assert router.stats["mock_fallback"] == 1

    @pytest.mark.asyncio
    async def test_get_quote_polygon_rate_limited(self) -> None:
        """Test quote fetch when Polygon is rate limited and Yahoo fails."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        async def mock_get_quote_rate_limited(symbol: str) -> Quote:
            raise PolygonRateLimitError(retry_after=60.0)

        mock_polygon.get_quote = mock_get_quote_rate_limited

        router = DataSourceRouter(polygon=mock_polygon)

        # Mock Yahoo to fail so we hit mock fallback
        with patch.object(router, "_fetch_yahoo_quote", return_value=None):
            result = await router.get_quote("AAPL")

            # Should fallback to mock (Yahoo and cache not available)
            assert result.quote is not None
            assert result.source == DataSource.MOCK
            assert result.fallback_used
            assert router.stats["polygon_rate_limited"] == 1

    @pytest.mark.asyncio
    async def test_get_quote_with_cache_fallback(self) -> None:
        """Test quote fetch falls back to cache when Polygon and Yahoo fail."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        async def mock_get_quote_error(symbol: str) -> Quote:
            raise PolygonAPIError(500, "Server error")

        mock_polygon.get_quote = mock_get_quote_error

        # Mock cache with stored quote
        mock_cache = AsyncMock()
        cached_quote = {
            "symbol": "AAPL",
            "price": 180.00,
            "change_percent": 0.5,
            "volume": 40_000_000,
            "source": "cache",
            "timestamp": datetime.now().isoformat(),
        }
        mock_cache.get = AsyncMock(return_value=cached_quote)

        router = DataSourceRouter(polygon=mock_polygon, cache=mock_cache)

        # Mock Yahoo to fail so we hit cache fallback
        with patch.object(router, "_fetch_yahoo_quote", return_value=None):
            result = await router.get_quote("AAPL")

            assert result.quote is not None
            assert result.source == DataSource.CACHE
            assert result.cached
            assert result.fallback_used
            assert router.stats["cache_fallback"] == 1

    @pytest.mark.asyncio
    async def test_get_quote_caches_result(self) -> None:
        """Test that successful quote is cached."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        mock_quote = Quote(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            source=DataSource.POLYGON,
        )

        async def mock_get_quote(symbol: str) -> Quote:
            return mock_quote

        mock_polygon.get_quote = mock_get_quote

        mock_cache = AsyncMock()
        mock_cache.set = AsyncMock()

        router = DataSourceRouter(polygon=mock_polygon, cache=mock_cache)
        await router.get_quote("AAPL")

        # Verify cache was called
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        # CacheKeyBuilder uses "portfolio_advisor:market:AAPL:quote" format
        assert "AAPL" in call_args[0][0]
        assert "quote" in call_args[0][0]
        assert call_args[1]["ttl"] == DataSourceRouter.QUOTE_TTL

    @pytest.mark.asyncio
    async def test_get_bars_polygon_success(self) -> None:
        """Test successful bars fetch from Polygon."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        mock_bars = [
            Bar(
                symbol="AAPL",
                open=184.0,
                high=186.0,
                low=183.5,
                close=185.5,
                volume=50_000_000,
                timestamp=datetime.now(),
            ),
            Bar(
                symbol="AAPL",
                open=185.5,
                high=187.0,
                low=185.0,
                close=186.5,
                volume=45_000_000,
                timestamp=datetime.now(),
            ),
        ]

        async def mock_get_bars(symbol: str, timeframe: str, limit: int) -> list[Bar]:
            return mock_bars

        mock_polygon.get_bars = mock_get_bars

        router = DataSourceRouter(polygon=mock_polygon)
        result = await router.get_bars("AAPL", "day", 5)

        assert result.bars is not None
        assert len(result.bars) == 2
        assert result.source == DataSource.POLYGON
        assert router.stats["polygon_success"] == 1

    @pytest.mark.asyncio
    async def test_get_bars_mock_fallback(self) -> None:
        """Test bars fetch falls back to mock."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = False

        router = DataSourceRouter(polygon=mock_polygon)
        result = await router.get_bars("AAPL", "day", 5)

        assert result.bars is not None
        assert len(result.bars) == 5
        assert result.source == DataSource.MOCK
        assert result.fallback_used
        assert router.stats["mock_fallback"] == 1

    @pytest.mark.asyncio
    async def test_get_company_info_polygon_success(self) -> None:
        """Test successful company info fetch from Polygon."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        mock_info = CompanyInfo(
            symbol="AAPL",
            name="Apple Inc.",
            description="Technology company",
            sector="Technology",
            market_cap=2_900_000_000_000,
            source=DataSource.POLYGON,
        )

        async def mock_get_company_info(symbol: str) -> CompanyInfo:
            return mock_info

        mock_polygon.get_company_info = mock_get_company_info

        router = DataSourceRouter(polygon=mock_polygon)
        result = await router.get_company_info("AAPL")

        assert result.company_info is not None
        assert result.company_info.name == "Apple Inc."
        assert result.source == DataSource.POLYGON
        assert router.stats["polygon_success"] == 1

    @pytest.mark.asyncio
    async def test_search_news_polygon_success(self) -> None:
        """Test successful news search from Polygon."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = True

        mock_articles = [
            NewsArticle(
                id="article-1",
                title="Apple Reports Strong Earnings",
                author="John Doe",
                publisher="Financial Times",
                url="https://ft.com/article-1",
                summary="Apple beat expectations...",
                symbols=["AAPL"],
                published_at=datetime.now(),
                source=DataSource.POLYGON,
            )
        ]

        async def mock_search_news(symbol: str | None, limit: int) -> list[NewsArticle]:
            return mock_articles

        mock_polygon.search_news = mock_search_news

        router = DataSourceRouter(polygon=mock_polygon)
        result = await router.search_news("AAPL", limit=5)

        assert result.news is not None
        assert len(result.news) == 1
        assert result.news[0].title == "Apple Reports Strong Earnings"
        assert result.source == DataSource.POLYGON
        assert router.stats["polygon_success"] == 1

    @pytest.mark.asyncio
    async def test_search_news_mock_fallback(self) -> None:
        """Test news search falls back to mock."""
        mock_polygon = MagicMock()
        mock_polygon.is_configured = False

        router = DataSourceRouter(polygon=mock_polygon)
        result = await router.search_news("AAPL", limit=3)

        assert result.news is not None
        assert len(result.news) == 3
        assert result.source == DataSource.MOCK
        assert result.fallback_used

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing the router."""
        mock_polygon = AsyncMock()
        router = DataSourceRouter(polygon=mock_polygon)
        await router.close()
        mock_polygon.close.assert_called_once()


class TestMockData:
    """Tests for mock data generation."""

    @pytest.mark.asyncio
    async def test_mock_quote_known_symbol(self) -> None:
        """Test mock quote for known symbol when all sources fail."""
        router = DataSourceRouter(polygon=MagicMock(is_configured=False))

        # Mock Yahoo to fail so we hit mock fallback
        with patch.object(router, "_fetch_yahoo_quote", return_value=None):
            result = await router.get_quote("AAPL")

            assert result.quote is not None
            assert result.quote.price == 185.50
            assert result.source == DataSource.MOCK

    @pytest.mark.asyncio
    async def test_mock_quote_unknown_symbol(self) -> None:
        """Test mock quote for unknown symbol when all sources fail."""
        router = DataSourceRouter(polygon=MagicMock(is_configured=False))

        # Mock Yahoo to fail so we hit mock fallback
        with patch.object(router, "_fetch_yahoo_quote", return_value=None):
            result = await router.get_quote("UNKNOWN")

            assert result.quote is not None
            assert result.quote.price == 100.0  # Default price
            assert result.source == DataSource.MOCK

    @pytest.mark.asyncio
    async def test_mock_bars_generates_correct_count(self) -> None:
        """Test mock bars generates requested count when Polygon not configured."""
        router = DataSourceRouter(polygon=MagicMock(is_configured=False))
        # Bars only use Polygon and cache, no Yahoo fallback
        result = await router.get_bars("AAPL", "day", 10)

        assert result.bars is not None
        assert len(result.bars) == 10

    @pytest.mark.asyncio
    async def test_mock_news_limits_correctly(self) -> None:
        """Test mock news respects limit when Polygon not configured."""
        router = DataSourceRouter(polygon=MagicMock(is_configured=False))
        # News only uses Polygon and cache, no Yahoo fallback

        # Request more than max mock articles
        result = await router.search_news("AAPL", limit=10)
        assert result.news is not None
        assert len(result.news) == 5  # Max mock articles

        # Request fewer
        result = await router.search_news("AAPL", limit=2)
        assert len(result.news) == 2
