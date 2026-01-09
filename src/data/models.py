"""Data models for market data integration.

This module defines the Pydantic models used across data sources
(Polygon.io, Yahoo Finance) and the caching layer.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Source of market data."""

    POLYGON = "polygon"
    YAHOO = "yahoo"
    CACHE = "cache"
    MOCK = "mock"


class Quote(BaseModel):
    """Real-time quote data for a symbol.

    Attributes:
        symbol: Stock ticker symbol.
        price: Current/last price.
        change_percent: Percentage change from previous close.
        volume: Trading volume.
        open: Opening price.
        high: Day high.
        low: Day low.
        previous_close: Previous day's close.
        timestamp: When the quote was fetched.
        source: Data source (polygon, yahoo, cache, mock).
    """

    symbol: str
    price: float
    change_percent: float
    volume: int
    open: float | None = None
    high: float | None = None
    low: float | None = None
    previous_close: float | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    source: DataSource = DataSource.POLYGON


class Bar(BaseModel):
    """OHLCV bar data for a time period.

    Attributes:
        symbol: Stock ticker symbol.
        open: Opening price.
        high: High price.
        low: Low price.
        close: Closing price.
        volume: Trading volume.
        vwap: Volume-weighted average price.
        timestamp: Bar timestamp.
        transactions: Number of transactions (if available).
    """

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float | None = None
    timestamp: datetime
    transactions: int | None = None


class CompanyInfo(BaseModel):
    """Company reference data.

    Attributes:
        symbol: Stock ticker symbol.
        name: Company name.
        description: Company description.
        sector: Industry sector.
        industry: Specific industry.
        market_cap: Market capitalization.
        employees: Number of employees.
        website: Company website URL.
        exchange: Stock exchange (e.g., NASDAQ, NYSE).
        currency: Trading currency.
        timestamp: When the data was fetched.
        source: Data source.
    """

    symbol: str
    name: str
    description: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    employees: int | None = None
    website: str | None = None
    exchange: str | None = None
    currency: str = "USD"
    timestamp: datetime = Field(default_factory=datetime.now)
    source: DataSource = DataSource.POLYGON


class NewsArticle(BaseModel):
    """News article related to a symbol or market.

    Attributes:
        id: Unique article identifier.
        title: Article title.
        author: Article author.
        publisher: Publishing source.
        url: Article URL.
        summary: Article summary/description.
        symbols: Related stock symbols.
        sentiment: Sentiment analysis (positive, negative, neutral).
        published_at: Publication timestamp.
        source: Data source.
    """

    id: str
    title: str
    author: str | None = None
    publisher: str
    url: str
    summary: str | None = None
    symbols: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "negative", "neutral"] | None = None
    published_at: datetime
    source: DataSource = DataSource.POLYGON


class MarketDataResult(BaseModel):
    """Unified result from any data source.

    Used by DataSourceRouter to return consistent data regardless of source.

    Attributes:
        symbol: Stock ticker symbol.
        quote: Quote data (if requested).
        bars: Historical bar data (if requested).
        company_info: Company reference data (if requested).
        news: Related news articles (if requested).
        source: Which source provided the data.
        cached: Whether this came from cache.
        fallback_used: Whether a fallback source was used.
        errors: Any errors encountered during fetch.
    """

    symbol: str
    quote: Quote | None = None
    bars: list[Bar] | None = None
    company_info: CompanyInfo | None = None
    news: list[NewsArticle] | None = None
    source: DataSource = DataSource.POLYGON
    cached: bool = False
    fallback_used: bool = False
    errors: list[str] = Field(default_factory=list)
