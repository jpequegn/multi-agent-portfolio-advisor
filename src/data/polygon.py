"""Polygon.io API client for market data.

This module provides an async client for the Polygon.io REST API
with support for quotes, historical bars, company info, and news.
"""

import os
from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog

from src.data.models import Bar, CompanyInfo, DataSource, NewsArticle, Quote

logger = structlog.get_logger(__name__)


class PolygonError(Exception):
    """Base exception for Polygon API errors."""

    pass


class PolygonAuthError(PolygonError):
    """Raised when API key is invalid or missing."""

    pass


class PolygonRateLimitError(PolygonError):
    """Raised when rate limit is exceeded (HTTP 429)."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")


class PolygonAPIError(PolygonError):
    """Raised for general API errors."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"Polygon API error {status}: {message}")


class PolygonClient:
    """Async client for Polygon.io REST API.

    Provides methods to fetch:
    - Real-time quotes
    - Historical OHLCV bars
    - Company reference data
    - Market news

    Example:
        client = PolygonClient()
        quote = await client.get_quote("AAPL")
        print(f"AAPL: ${quote.price}")
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Polygon client.

        Args:
            api_key: Polygon.io API key. If not provided, reads from
                POLYGON_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        self._client: httpx.AsyncClient | None = None
        self._logger = logger.bind(component="polygon_client")

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an authenticated request to Polygon API.

        Args:
            endpoint: API endpoint (e.g., "/v2/aggs/ticker/AAPL/prev").
            params: Optional query parameters.

        Returns:
            JSON response data.

        Raises:
            PolygonAuthError: If API key is missing or invalid.
            PolygonRateLimitError: If rate limit is exceeded.
            PolygonAPIError: For other API errors.
        """
        if not self.api_key:
            raise PolygonAuthError("POLYGON_API_KEY not configured")

        client = await self._get_client()
        url = f"{self.BASE_URL}{endpoint}"

        self._logger.debug("polygon_request", endpoint=endpoint, params=params)

        try:
            response = await client.get(url, params=params)

            if response.status_code == 401:
                raise PolygonAuthError("Invalid API key")

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise PolygonRateLimitError(
                    retry_after=float(retry_after) if retry_after else 12.0
                )

            if response.status_code != 200:
                raise PolygonAPIError(response.status_code, response.text)

            data = response.json()
            self._logger.debug(
                "polygon_response",
                endpoint=endpoint,
                status=response.status_code,
            )
            return data

        except httpx.RequestError as e:
            self._logger.error("polygon_client_error", error=str(e))
            raise PolygonAPIError(0, str(e)) from e

    async def get_quote(self, symbol: str) -> Quote:
        """Get the previous day's OHLCV for a symbol.

        Uses the /v2/aggs/ticker/{symbol}/prev endpoint.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").

        Returns:
            Quote with price and volume data.

        Raises:
            PolygonError: If the request fails.
            ValueError: If symbol data is not available.
        """
        symbol = symbol.upper()
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"

        data = await self._request(endpoint)

        if data.get("resultsCount", 0) == 0:
            raise ValueError(f"No data available for symbol: {symbol}")

        result = data["results"][0]

        # Calculate change percent from open to close
        open_price = result.get("o", 0)
        close_price = result.get("c", 0)
        if open_price > 0:
            change_percent = ((close_price - open_price) / open_price) * 100
        else:
            change_percent = 0.0

        return Quote(
            symbol=symbol,
            price=close_price,
            change_percent=round(change_percent, 2),
            volume=int(result.get("v", 0)),
            open=result.get("o"),
            high=result.get("h"),
            low=result.get("l"),
            previous_close=result.get("c"),
            timestamp=datetime.fromtimestamp(result.get("t", 0) / 1000),
            source=DataSource.POLYGON,
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "day",
        limit: int = 30,
    ) -> list[Bar]:
        """Get historical OHLCV bars for a symbol.

        Uses the /v2/aggs/ticker/{symbol}/range endpoint.

        Args:
            symbol: Stock ticker symbol.
            timeframe: Bar timeframe ("day", "week", "month").
            limit: Number of bars to fetch (max 50000).

        Returns:
            List of Bar objects.
        """
        symbol = symbol.upper()

        # Map timeframe to Polygon format
        timeframe_map = {
            "day": ("1", "day"),
            "week": ("1", "week"),
            "month": ("1", "month"),
        }
        multiplier, timespan = timeframe_map.get(timeframe, ("1", "day"))

        # Calculate date range
        end_date = datetime.now()
        if timeframe == "day":
            start_date = end_date - timedelta(days=limit)
        elif timeframe == "week":
            start_date = end_date - timedelta(weeks=limit)
        else:
            start_date = end_date - timedelta(days=limit * 30)

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

        data = await self._request(endpoint, params={"limit": limit, "sort": "desc"})

        bars = []
        for result in data.get("results", []):
            bars.append(
                Bar(
                    symbol=symbol,
                    open=result["o"],
                    high=result["h"],
                    low=result["l"],
                    close=result["c"],
                    volume=int(result["v"]),
                    vwap=result.get("vw"),
                    timestamp=datetime.fromtimestamp(result["t"] / 1000),
                    transactions=result.get("n"),
                )
            )

        return bars

    async def get_company_info(self, symbol: str) -> CompanyInfo:
        """Get company reference data.

        Uses the /v3/reference/tickers/{symbol} endpoint.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            CompanyInfo with company details.
        """
        symbol = symbol.upper()
        endpoint = f"/v3/reference/tickers/{symbol}"

        data = await self._request(endpoint)

        if data.get("status") != "OK" or "results" not in data:
            raise ValueError(f"Company info not available for: {symbol}")

        result = data["results"]

        return CompanyInfo(
            symbol=symbol,
            name=result.get("name", symbol),
            description=result.get("description"),
            sector=result.get("sic_description"),
            industry=result.get("sic_description"),
            market_cap=result.get("market_cap"),
            employees=result.get("total_employees"),
            website=result.get("homepage_url"),
            exchange=result.get("primary_exchange"),
            currency=result.get("currency_name", "USD"),
            source=DataSource.POLYGON,
        )

    async def search_news(
        self,
        symbol: str | None = None,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Search for market news.

        Uses the /v2/reference/news endpoint.

        Args:
            symbol: Optional stock symbol to filter news.
            limit: Maximum number of articles (default 10).

        Returns:
            List of NewsArticle objects.
        """
        endpoint = "/v2/reference/news"
        params: dict[str, Any] = {"limit": limit, "order": "desc"}

        if symbol:
            params["ticker"] = symbol.upper()

        data = await self._request(endpoint, params=params)

        articles = []
        for result in data.get("results", []):
            # Parse published timestamp
            published_str = result.get("published_utc", "")
            try:
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                published_at = datetime.now()

            articles.append(
                NewsArticle(
                    id=result.get("id", ""),
                    title=result.get("title", ""),
                    author=result.get("author"),
                    publisher=result.get("publisher", {}).get("name", "Unknown"),
                    url=result.get("article_url", ""),
                    summary=result.get("description"),
                    symbols=result.get("tickers", []),
                    published_at=published_at,
                    source=DataSource.POLYGON,
                )
            )

        return articles

    async def __aenter__(self) -> "PolygonClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
