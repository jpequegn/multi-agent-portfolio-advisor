"""Tests for the Polygon.io client."""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import DataSource
from src.data.polygon import (
    PolygonAPIError,
    PolygonAuthError,
    PolygonClient,
    PolygonRateLimitError,
)


class TestPolygonClient:
    """Tests for PolygonClient."""

    def test_init_with_api_key(self) -> None:
        """Test initialization with explicit API key."""
        client = PolygonClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.is_configured is True

    def test_init_from_env(self) -> None:
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"POLYGON_API_KEY": "env_key"}):
            client = PolygonClient()
            assert client.api_key == "env_key"
            assert client.is_configured is True

    def test_init_no_key(self) -> None:
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove POLYGON_API_KEY if it exists
            os.environ.pop("POLYGON_API_KEY", None)
            client = PolygonClient(api_key=None)
            assert client.is_configured is False

    @pytest.mark.asyncio
    async def test_get_quote_success(self) -> None:
        """Test successful quote fetch."""
        client = PolygonClient(api_key="test_key")

        mock_response = {
            "resultsCount": 1,
            "results": [
                {
                    "c": 185.50,
                    "o": 184.00,
                    "h": 186.00,
                    "l": 183.50,
                    "v": 50000000,
                    "t": 1704067200000,  # Unix timestamp in ms
                }
            ],
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            quote = await client.get_quote("AAPL")

            assert quote.symbol == "AAPL"
            assert quote.price == 185.50
            assert quote.volume == 50000000
            assert quote.source == DataSource.POLYGON

    @pytest.mark.asyncio
    async def test_get_quote_no_data(self) -> None:
        """Test quote fetch when no data available."""
        client = PolygonClient(api_key="test_key")

        mock_response = {"resultsCount": 0, "results": []}

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            with pytest.raises(ValueError, match="No data available"):
                await client.get_quote("INVALID")

    @pytest.mark.asyncio
    async def test_get_bars_success(self) -> None:
        """Test successful bars fetch."""
        client = PolygonClient(api_key="test_key")

        mock_response = {
            "results": [
                {
                    "o": 184.00,
                    "h": 186.00,
                    "l": 183.50,
                    "c": 185.50,
                    "v": 50000000,
                    "vw": 185.00,
                    "t": 1704067200000,
                    "n": 100000,
                },
                {
                    "o": 185.50,
                    "h": 187.00,
                    "l": 185.00,
                    "c": 186.50,
                    "v": 45000000,
                    "vw": 186.00,
                    "t": 1704153600000,
                    "n": 95000,
                },
            ]
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            bars = await client.get_bars("AAPL", "day", 5)

            assert len(bars) == 2
            assert bars[0].symbol == "AAPL"
            assert bars[0].close == 185.50
            assert bars[0].volume == 50000000

    @pytest.mark.asyncio
    async def test_get_company_info_success(self) -> None:
        """Test successful company info fetch."""
        client = PolygonClient(api_key="test_key")

        mock_response = {
            "status": "OK",
            "results": {
                "name": "Apple Inc.",
                "description": "Apple designs and manufactures consumer electronics.",
                "sic_description": "Technology",
                "market_cap": 2900000000000,
                "total_employees": 150000,
                "homepage_url": "https://apple.com",
                "primary_exchange": "NASDAQ",
                "currency_name": "USD",
            },
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            info = await client.get_company_info("AAPL")

            assert info.symbol == "AAPL"
            assert info.name == "Apple Inc."
            assert info.market_cap == 2900000000000
            assert info.source == DataSource.POLYGON

    @pytest.mark.asyncio
    async def test_search_news_success(self) -> None:
        """Test successful news search."""
        client = PolygonClient(api_key="test_key")

        mock_response = {
            "results": [
                {
                    "id": "article-1",
                    "title": "Apple Reports Strong Earnings",
                    "author": "John Doe",
                    "publisher": {"name": "Financial Times"},
                    "article_url": "https://ft.com/article-1",
                    "description": "Apple beat expectations...",
                    "tickers": ["AAPL"],
                    "published_utc": "2024-01-15T10:00:00Z",
                }
            ]
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            articles = await client.search_news("AAPL", limit=5)

            assert len(articles) == 1
            assert articles[0].title == "Apple Reports Strong Earnings"
            assert articles[0].publisher == "Financial Times"
            assert articles[0].source == DataSource.POLYGON

    @pytest.mark.asyncio
    async def test_request_no_api_key(self) -> None:
        """Test request without API key raises error."""
        client = PolygonClient(api_key=None)

        with pytest.raises(PolygonAuthError, match="not configured"):
            await client._request("/v2/aggs/ticker/AAPL/prev")

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with PolygonClient(api_key="test_key") as client:
            assert client.api_key == "test_key"


class TestPolygonErrors:
    """Tests for Polygon error classes."""

    def test_polygon_auth_error(self) -> None:
        """Test PolygonAuthError."""
        error = PolygonAuthError("Invalid API key")
        assert "Invalid API key" in str(error)

    def test_polygon_rate_limit_error(self) -> None:
        """Test PolygonRateLimitError."""
        error = PolygonRateLimitError(retry_after=12.0)
        assert error.retry_after == 12.0
        assert "Rate limit exceeded" in str(error)

    def test_polygon_api_error(self) -> None:
        """Test PolygonAPIError."""
        error = PolygonAPIError(500, "Internal server error")
        assert error.status == 500
        assert error.message == "Internal server error"
        assert "500" in str(error)
