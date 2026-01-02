"""Tests for the Market Data Tool."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.base import ToolExecutionError
from src.tools.market_data import (
    DEFAULT_MOCK,
    MOCK_STOCKS,
    MarketDataCache,
    MarketDataInput,
    MarketDataOutput,
    MarketDataTool,
)


class TestMarketDataInput:
    """Tests for MarketDataInput model."""

    def test_required_fields(self) -> None:
        """Test that symbol is required."""
        input_data = MarketDataInput(symbol="AAPL")
        assert input_data.symbol == "AAPL"
        assert input_data.data_type == "quote"

    def test_data_type_options(self) -> None:
        """Test valid data_type values."""
        for data_type in ["quote", "history", "fundamentals"]:
            input_data = MarketDataInput(symbol="AAPL", data_type=data_type)  # type: ignore[arg-type]
            assert input_data.data_type == data_type

    def test_invalid_data_type(self) -> None:
        """Test invalid data_type raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MarketDataInput(symbol="AAPL", data_type="invalid")  # type: ignore[arg-type]


class TestMarketDataOutput:
    """Tests for MarketDataOutput model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        output = MarketDataOutput(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            timestamp="2024-01-15T10:00:00",
            source="real",
        )
        assert output.symbol == "AAPL"
        assert output.price == 185.50
        assert output.source == "real"
        assert output.success is True

    def test_optional_fields(self) -> None:
        """Test optional fields default to None."""
        output = MarketDataOutput(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            timestamp="2024-01-15T10:00:00",
            source="mock",
        )
        assert output.market_cap is None
        assert output.pe_ratio is None
        assert output.dividend_yield is None

    def test_all_fields(self) -> None:
        """Test with all fields populated."""
        output = MarketDataOutput(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            market_cap=2.9e12,
            pe_ratio=28.5,
            dividend_yield=0.5,
            fifty_two_week_high=199.62,
            fifty_two_week_low=164.08,
            timestamp="2024-01-15T10:00:00",
            source="cache",
        )
        assert output.market_cap == 2.9e12
        assert output.fifty_two_week_high == 199.62


class TestMarketDataCache:
    """Tests for MarketDataCache."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create mock Redis client."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_cache_key_format(self) -> None:
        """Test cache key generation."""
        cache = MarketDataCache()
        key = cache._cache_key("aapl", "quote")
        assert key == "market_data:AAPL:quote"

    @pytest.mark.asyncio
    async def test_cache_get_miss(self) -> None:
        """Test cache miss returns None."""
        cache = MarketDataCache()
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        cache._client = mock_client

        result = await cache.get("AAPL", "quote")

        assert result is None
        mock_client.get.assert_called_once_with("market_data:AAPL:quote")

    @pytest.mark.asyncio
    async def test_cache_get_hit(self) -> None:
        """Test cache hit returns data."""
        cache = MarketDataCache()
        mock_client = AsyncMock()
        cached_data = MarketDataOutput(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            timestamp="2024-01-15T10:00:00",
            source="cache",
        )
        mock_client.get.return_value = cached_data.model_dump_json()
        cache._client = mock_client

        result = await cache.get("AAPL", "quote")

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.price == 185.50

    @pytest.mark.asyncio
    async def test_cache_set(self) -> None:
        """Test setting cache."""
        cache = MarketDataCache(ttl=300)
        mock_client = AsyncMock()
        mock_client.setex.return_value = True
        cache._client = mock_client

        data = MarketDataOutput(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            timestamp="2024-01-15T10:00:00",
            source="real",
        )

        result = await cache.set("AAPL", "quote", data)

        assert result is True
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == "market_data:AAPL:quote"
        assert call_args[0][1] == 300

    @pytest.mark.asyncio
    async def test_cache_error_handling(self) -> None:
        """Test cache handles Redis errors gracefully."""
        import redis.asyncio as redis_async

        cache = MarketDataCache()
        mock_client = AsyncMock()
        mock_client.get.side_effect = redis_async.RedisError("Connection failed")
        cache._client = mock_client

        result = await cache.get("AAPL", "quote")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_close(self) -> None:
        """Test closing cache connection."""
        cache = MarketDataCache()
        mock_client = AsyncMock()
        cache._client = mock_client

        await cache.close()

        mock_client.close.assert_called_once()
        assert cache._client is None


class TestMarketDataTool:
    """Tests for MarketDataTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = MarketDataTool()
        assert tool.name == "get_market_data"
        assert "market data" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = MarketDataTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "get_market_data"
        assert "symbol" in anthropic_tool["input_schema"]["properties"]
        assert "data_type" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mock_known_symbol(self) -> None:
        """Test mock execution for known symbols."""
        tool = MarketDataTool(use_mock=True)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.price == MOCK_STOCKS["AAPL"]["price"]
        assert result.source == "mock"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_mock_unknown_symbol(self) -> None:
        """Test mock execution for unknown symbols uses defaults."""
        tool = MarketDataTool(use_mock=True)
        result = await tool.execute({"symbol": "UNKNOWN"})

        assert result.symbol == "UNKNOWN"
        assert result.price == DEFAULT_MOCK["price"]
        assert result.source == "mock"

    @pytest.mark.asyncio
    async def test_execute_mock_case_insensitive(self) -> None:
        """Test mock handles case-insensitive symbols."""
        tool = MarketDataTool(use_mock=True)
        result = await tool.execute({"symbol": "aapl"})

        assert result.symbol == "AAPL"
        assert result.price == MOCK_STOCKS["AAPL"]["price"]

    @pytest.mark.asyncio
    async def test_execute_mock_all_fields(self) -> None:
        """Test mock returns all expected fields."""
        tool = MarketDataTool(use_mock=True)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.market_cap == MOCK_STOCKS["AAPL"]["market_cap"]
        assert result.pe_ratio == MOCK_STOCKS["AAPL"]["pe_ratio"]
        assert result.fifty_two_week_high == MOCK_STOCKS["AAPL"]["fifty_two_week_high"]
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_execute_real_with_cache_hit(self) -> None:
        """Test real execution uses cache when available."""
        mock_cache = AsyncMock(spec=MarketDataCache)
        cached_data = MarketDataOutput(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            timestamp="2024-01-15T10:00:00",
            source="cache",
        )
        mock_cache.get.return_value = cached_data

        tool = MarketDataTool(use_mock=False, cache=mock_cache)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.source == "cache"
        assert result.price == 185.50
        mock_cache.get.assert_called_once_with("AAPL", "quote")

    @pytest.mark.asyncio
    async def test_execute_real_fetches_data(self) -> None:
        """Test real execution fetches from yfinance."""
        mock_cache = AsyncMock(spec=MarketDataCache)
        mock_cache.get.return_value = None  # Cache miss

        tool = MarketDataTool(use_mock=False, cache=mock_cache)

        mock_ticker_data: dict[str, Any] = {
            "price": 190.50,
            "change_percent": 1.5,
            "volume": 55_000_000,
            "market_cap": 3.0e12,
            "pe_ratio": 29.0,
            "dividend_yield": 0.6,
            "fifty_two_week_high": 200.0,
            "fifty_two_week_low": 165.0,
        }

        with patch.object(tool, "_fetch_yfinance_data", return_value=mock_ticker_data):
            result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.price == 190.50
        assert result.source == "real"
        mock_cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_real_invalid_symbol(self) -> None:
        """Test real execution handles invalid symbols."""
        mock_cache = AsyncMock(spec=MarketDataCache)
        mock_cache.get.return_value = None

        tool = MarketDataTool(use_mock=False, fallback_to_mock=False, cache=mock_cache)

        with (
            patch.object(tool, "_fetch_yfinance_data", return_value=None),
            pytest.raises(ToolExecutionError),
        ):
            await tool.execute({"symbol": "INVALID"})

    @pytest.mark.asyncio
    async def test_execute_real_fallback_to_mock(self) -> None:
        """Test fallback to mock when real API fails."""
        mock_cache = AsyncMock(spec=MarketDataCache)
        mock_cache.get.return_value = None

        tool = MarketDataTool(use_mock=False, fallback_to_mock=True, cache=mock_cache)

        with patch.object(tool, "_fetch_yfinance_data", return_value=None):
            result = await tool.execute({"symbol": "AAPL"})

        assert result.source == "mock"
        assert result.success is True
        assert "Fallback" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_without_cache(self) -> None:
        """Test execution without cache."""
        tool = MarketDataTool(use_mock=False, use_cache=False)

        mock_ticker_data: dict[str, Any] = {
            "price": 190.50,
            "change_percent": 1.5,
            "volume": 55_000_000,
        }

        with patch.object(tool, "_fetch_yfinance_data", return_value=mock_ticker_data):
            result = await tool.execute({"symbol": "AAPL"})

        assert result.price == 190.50
        assert result.source == "real"


class TestYFinanceIntegration:
    """Tests for yfinance data fetching."""

    def test_fetch_yfinance_data_success(self) -> None:
        """Test successful yfinance data fetch."""
        tool = MarketDataTool()

        mock_ticker = MagicMock()
        mock_ticker.info = {
            "regularMarketPrice": 185.50,
            "previousClose": 183.00,
            "regularMarketVolume": 50_000_000,
            "marketCap": 2_900_000_000_000,
            "trailingPE": 28.5,
            "dividendYield": 0.005,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 164.08,
        }

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = tool._fetch_yfinance_data("AAPL")

        assert result is not None
        assert result["price"] == 185.50
        assert abs(result["change_percent"] - 1.37) < 0.01  # (185.5-183)/183 * 100
        assert result["volume"] == 50_000_000
        assert result["dividend_yield"] == 0.5  # Converted from 0.005 to 0.5%

    def test_fetch_yfinance_data_no_price(self) -> None:
        """Test handling of ticker with no price data."""
        tool = MarketDataTool()

        mock_ticker = MagicMock()
        mock_ticker.info = {"symbol": "INVALID", "regularMarketPrice": None}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = tool._fetch_yfinance_data("INVALID")

        assert result is None

    def test_fetch_yfinance_data_empty_info(self) -> None:
        """Test handling of empty ticker info."""
        tool = MarketDataTool()

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = tool._fetch_yfinance_data("INVALID")

        assert result is None

    def test_fetch_yfinance_data_exception(self) -> None:
        """Test handling of yfinance exception."""
        tool = MarketDataTool()

        with patch("yfinance.Ticker", side_effect=Exception("API Error")):
            result = tool._fetch_yfinance_data("AAPL")

        assert result is None

    def test_to_percent_conversion(self) -> None:
        """Test percentage conversion utility."""
        assert MarketDataTool._to_percent(0.005) == 0.5
        assert MarketDataTool._to_percent(0.0125) == 1.25
        assert MarketDataTool._to_percent(None) is None
        assert MarketDataTool._to_percent(0) == 0.0


class TestMarketDataToolInputValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_missing_symbol(self) -> None:
        """Test that missing symbol raises error."""
        tool = MarketDataTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_accepts_model_input(self) -> None:
        """Test that validated model can be passed directly."""
        tool = MarketDataTool(use_mock=True)
        input_data = MarketDataInput(symbol="AAPL")

        result = await tool.execute(input_data)

        assert result.symbol == "AAPL"
