"""Tests for the Market Data Tool."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import DataSource, MarketDataResult, Quote
from src.tools.base import ToolExecutionError
from src.tools.market_data import (
    DEFAULT_MOCK,
    MOCK_STOCKS,
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
            source="polygon",
        )
        assert output.symbol == "AAPL"
        assert output.price == 185.50
        assert output.source == "polygon"
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


class TestMarketDataTool:
    """Tests for MarketDataTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = MarketDataTool(use_mock=True)
        assert tool.name == "get_market_data"
        assert "market data" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = MarketDataTool(use_mock=True)
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


class TestMarketDataToolWithRouter:
    """Tests for MarketDataTool with DataSourceRouter."""

    @pytest.mark.asyncio
    async def test_execute_real_uses_router(self) -> None:
        """Test real execution uses DataSourceRouter."""
        mock_router = MagicMock()
        mock_quote = Quote(
            symbol="AAPL",
            price=185.50,
            change_percent=1.2,
            volume=50_000_000,
            high=186.00,
            low=184.00,
            timestamp=datetime.now(),
            source=DataSource.POLYGON,
        )
        mock_result = MarketDataResult(
            symbol="AAPL",
            quote=mock_quote,
            source=DataSource.POLYGON,
        )

        async def mock_get_quote(symbol: str) -> MarketDataResult:
            return mock_result

        mock_router.get_quote = mock_get_quote

        tool = MarketDataTool(use_mock=False, router=mock_router)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.price == 185.50
        assert result.source == "polygon"

    @pytest.mark.asyncio
    async def test_execute_real_yahoo_fallback(self) -> None:
        """Test real execution with Yahoo fallback."""
        mock_router = MagicMock()
        mock_quote = Quote(
            symbol="AAPL",
            price=186.00,
            change_percent=0.8,
            volume=48_000_000,
            timestamp=datetime.now(),
            source=DataSource.YAHOO,
        )
        mock_result = MarketDataResult(
            symbol="AAPL",
            quote=mock_quote,
            source=DataSource.YAHOO,
            fallback_used=True,
        )

        async def mock_get_quote(symbol: str) -> MarketDataResult:
            return mock_result

        mock_router.get_quote = mock_get_quote

        tool = MarketDataTool(use_mock=False, router=mock_router)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.source == "yahoo"

    @pytest.mark.asyncio
    async def test_execute_real_cache_fallback(self) -> None:
        """Test real execution with cache fallback."""
        mock_router = MagicMock()
        mock_quote = Quote(
            symbol="AAPL",
            price=184.00,
            change_percent=0.5,
            volume=45_000_000,
            timestamp=datetime.now(),
            source=DataSource.CACHE,
        )
        mock_result = MarketDataResult(
            symbol="AAPL",
            quote=mock_quote,
            source=DataSource.CACHE,
            cached=True,
            fallback_used=True,
        )

        async def mock_get_quote(symbol: str) -> MarketDataResult:
            return mock_result

        mock_router.get_quote = mock_get_quote

        tool = MarketDataTool(use_mock=False, router=mock_router)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.source == "cache"

    @pytest.mark.asyncio
    async def test_execute_real_mock_fallback(self) -> None:
        """Test real execution with mock fallback."""
        mock_router = MagicMock()
        mock_quote = Quote(
            symbol="AAPL",
            price=185.50,
            change_percent=0.0,
            volume=1_000_000,
            timestamp=datetime.now(),
            source=DataSource.MOCK,
        )
        mock_result = MarketDataResult(
            symbol="AAPL",
            quote=mock_quote,
            source=DataSource.MOCK,
            fallback_used=True,
        )

        async def mock_get_quote(symbol: str) -> MarketDataResult:
            return mock_result

        mock_router.get_quote = mock_get_quote

        tool = MarketDataTool(use_mock=False, router=mock_router)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.source == "mock"

    @pytest.mark.asyncio
    async def test_execute_real_no_quote_falls_back_to_mock(self) -> None:
        """Test real execution falls back to mock when no quote available."""
        mock_router = MagicMock()
        mock_result = MarketDataResult(
            symbol="INVALID",
            quote=None,
            source=DataSource.MOCK,
            fallback_used=True,
        )

        async def mock_get_quote(symbol: str) -> MarketDataResult:
            return mock_result

        mock_router.get_quote = mock_get_quote

        tool = MarketDataTool(use_mock=False, router=mock_router)

        # Tool falls back to mock mode when real data fails
        result = await tool.execute({"symbol": "INVALID"})
        assert result.symbol == "INVALID"
        assert result.source == "mock"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_close_closes_router(self) -> None:
        """Test closing tool closes router."""
        mock_router = AsyncMock()
        tool = MarketDataTool(use_mock=False, router=mock_router)

        await tool.close()

        mock_router.close.assert_called_once()


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
