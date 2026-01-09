"""News Search Tool for fetching financial news.

This module implements the NewsSearchTool that fetches news using the
DataSourceRouter (Polygon → Google News → Cache → Mock fallback chain).
"""

import hashlib
import random
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

import httpx
import structlog
from pydantic import Field

from src.tools.base import BaseTool, ToolInput, ToolOutput

if TYPE_CHECKING:
    from src.data.router import DataSourceRouter

logger = structlog.get_logger(__name__)


# ============================================================================
# Input/Output Schemas
# ============================================================================


class NewsSearchInput(ToolInput):
    """Input for news search tool."""

    query: str = Field(..., description="Search query for news articles")
    symbols: list[str] = Field(
        default_factory=list,
        description="Stock symbols to include in search",
    )
    days_back: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of days to search back",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return",
    )


class NewsItem(ToolOutput):
    """A single news item."""

    title: str
    summary: str
    source: str
    url: str
    published_at: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    sentiment: Literal["positive", "negative", "neutral"] | None = None


class NewsSearchOutput(ToolOutput):
    """Output from news search tool."""

    query: str
    items: list[NewsItem] = Field(default_factory=list)
    total_found: int = 0
    source: Literal["real", "mock"]


# ============================================================================
# Mock Data Generator
# ============================================================================


# Templates for generating realistic financial news headlines
HEADLINE_TEMPLATES = [
    "{symbol} stock {movement} after {event}",
    "{symbol} reports {result} earnings, {reaction}",
    "Analysts {action} {symbol} price target to ${price}",
    "{symbol} announces {announcement}",
    "Breaking: {symbol} {breaking_news}",
    "{symbol} shares {direction} on {catalyst}",
    "Wall Street {sentiment} on {symbol} ahead of {upcoming}",
    "{symbol} CEO discusses {topic} in interview",
    "{symbol} beats estimates, stock {post_earnings}",
    "Insider {insider_action} at {symbol} raises questions",
]

MOVEMENTS = ["surges", "drops", "rallies", "tumbles", "rises", "falls", "jumps", "slides"]
EVENTS = [
    "earnings beat",
    "product launch",
    "acquisition news",
    "analyst upgrade",
    "market rally",
    "sector rotation",
    "FDA approval",
    "partnership announcement",
]
RESULTS = ["strong", "weak", "mixed", "record", "disappointing", "surprising"]
REACTIONS = ["stock jumps", "shares fall", "investors cheer", "market reacts"]
ACTIONS = ["raise", "lower", "maintain", "initiate coverage on"]
ANNOUNCEMENTS = [
    "new product line",
    "strategic partnership",
    "share buyback",
    "dividend increase",
    "restructuring plan",
    "expansion into new markets",
]
DIRECTIONS = ["surge", "plunge", "climb", "sink", "edge higher", "edge lower"]
CATALYSTS = [
    "strong guidance",
    "analyst upgrade",
    "sector momentum",
    "economic data",
    "Fed comments",
    "trade news",
]
SENTIMENTS = ["bullish", "bearish", "cautious", "optimistic", "uncertain"]
UPCOMINGS = ["earnings", "product launch", "investor day", "Fed meeting", "economic report"]
TOPICS = ["growth strategy", "market conditions", "competition", "innovation", "sustainability"]
INSIDER_ACTIONS = ["buying", "selling", "trading activity"]

SOURCES = [
    "Reuters",
    "Bloomberg",
    "CNBC",
    "Wall Street Journal",
    "Financial Times",
    "MarketWatch",
    "Yahoo Finance",
    "Barron's",
    "Seeking Alpha",
    "The Motley Fool",
]


def generate_mock_headline(symbol: str) -> tuple[str, str]:
    """Generate a realistic mock headline for a stock symbol.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        Tuple of (headline, summary).
    """
    template = random.choice(HEADLINE_TEMPLATES)
    price = random.randint(50, 500)

    headline = template.format(
        symbol=symbol,
        movement=random.choice(MOVEMENTS),
        event=random.choice(EVENTS),
        result=random.choice(RESULTS),
        reaction=random.choice(REACTIONS),
        action=random.choice(ACTIONS),
        price=price,
        announcement=random.choice(ANNOUNCEMENTS),
        breaking_news=random.choice(EVENTS),
        direction=random.choice(DIRECTIONS),
        catalyst=random.choice(CATALYSTS),
        sentiment=random.choice(SENTIMENTS),
        upcoming=random.choice(UPCOMINGS),
        topic=random.choice(TOPICS),
        insider_action=random.choice(INSIDER_ACTIONS),
        post_earnings=random.choice(MOVEMENTS),
    )

    # Generate a summary
    summary = f"Financial news regarding {symbol}. {headline}. " f"Market analysts are monitoring developments closely."

    return headline, summary


def generate_mock_news(
    query: str,
    symbols: list[str],
    days_back: int,
    max_results: int,
) -> list[NewsItem]:
    """Generate mock news items.

    Args:
        query: Search query.
        symbols: Stock symbols.
        days_back: Number of days back to generate news for.
        max_results: Maximum number of items to generate.

    Returns:
        List of mock news items.
    """
    items: list[NewsItem] = []

    # Use symbols if provided, otherwise extract from query
    target_symbols = symbols if symbols else [query.upper()[:5]]

    now = datetime.now(UTC)

    for i in range(max_results):
        symbol = target_symbols[i % len(target_symbols)]
        headline, summary = generate_mock_headline(symbol)

        # Generate a random timestamp within the date range
        hours_back = random.randint(1, days_back * 24)
        published = now - timedelta(hours=hours_back)

        # Generate deterministic but varied relevance score based on content
        content_hash = hashlib.md5(headline.encode()).hexdigest()
        base_score = int(content_hash[:2], 16) / 255  # 0-1 range
        relevance = round(0.5 + (base_score * 0.5), 2)  # 0.5-1.0 range

        # Determine sentiment from headline
        positive_words = ["surge", "rally", "jump", "rise", "beat", "bullish", "optimistic"]
        negative_words = ["drop", "tumble", "fall", "slide", "sink", "bearish", "disappointing"]

        headline_lower = headline.lower()
        if any(word in headline_lower for word in positive_words):
            sentiment: Literal["positive", "negative", "neutral"] = "positive"
        elif any(word in headline_lower for word in negative_words):
            sentiment = "negative"
        else:
            sentiment = "neutral"

        items.append(
            NewsItem(
                title=headline,
                summary=summary,
                source=random.choice(SOURCES),
                url=f"https://example.com/news/{symbol.lower()}-{i}",
                published_at=published.isoformat(),
                relevance_score=relevance,
                sentiment=sentiment,
            )
        )

    # Sort by relevance score descending
    items.sort(key=lambda x: x.relevance_score, reverse=True)

    return items


# ============================================================================
# Relevance Scoring
# ============================================================================


def calculate_relevance_score(
    title: str,
    summary: str,
    query: str,
    symbols: list[str],
) -> float:
    """Calculate relevance score for a news item.

    Args:
        title: News title.
        summary: News summary.
        query: Original search query.
        symbols: Stock symbols to match.

    Returns:
        Relevance score between 0.0 and 1.0.
    """
    score = 0.0
    text = f"{title} {summary}".lower()
    query_lower = query.lower()

    # Query word matches (up to 0.4)
    query_words = query_lower.split()
    matches = sum(1 for word in query_words if word in text)
    query_score = min(matches / max(len(query_words), 1), 1.0) * 0.4
    score += query_score

    # Symbol matches (up to 0.4)
    symbol_matches = sum(1 for symbol in symbols if symbol.lower() in text or symbol.upper() in text)
    symbol_score = min(symbol_matches / max(len(symbols), 1), 1.0) * 0.4
    score += symbol_score

    # Financial keyword boost (up to 0.2)
    financial_keywords = [
        "stock",
        "shares",
        "market",
        "earnings",
        "revenue",
        "profit",
        "investor",
        "trading",
        "analyst",
        "dividend",
    ]
    keyword_matches = sum(1 for kw in financial_keywords if kw in text)
    keyword_score = min(keyword_matches / 5, 1.0) * 0.2
    score += keyword_score

    return round(min(score, 1.0), 2)


# ============================================================================
# Google News RSS Parser
# ============================================================================


def parse_google_news_rss(xml_content: str, query: str, symbols: list[str]) -> list[NewsItem]:
    """Parse Google News RSS feed XML.

    Args:
        xml_content: Raw XML content from Google News RSS.
        query: Original search query.
        symbols: Stock symbols for relevance scoring.

    Returns:
        List of parsed news items.
    """
    items: list[NewsItem] = []

    try:
        root = ET.fromstring(xml_content)
        channel = root.find("channel")
        if channel is None:
            return items

        for item in channel.findall("item"):
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_date_elem = item.find("pubDate")
            source_elem = item.find("source")
            description_elem = item.find("description")

            if title_elem is None or link_elem is None:
                continue

            title = title_elem.text or ""
            url = link_elem.text or ""
            source = source_elem.text or "Unknown" if source_elem is not None else "Unknown"

            # Parse publication date
            pub_date_str = pub_date_elem.text if pub_date_elem is not None else None
            if pub_date_str:
                try:
                    # Parse RFC 2822 date format
                    published_at = _parse_rfc2822_date(pub_date_str)
                except ValueError:
                    published_at = datetime.now(UTC)
            else:
                published_at = datetime.now(UTC)

            # Extract summary from description (strip HTML)
            description = description_elem.text if description_elem is not None else ""
            summary = _strip_html(description)[:500] if description else title

            # Calculate relevance
            relevance = calculate_relevance_score(title, summary, query, symbols)

            # Determine sentiment
            sentiment = _analyze_sentiment(title)

            items.append(
                NewsItem(
                    title=title,
                    summary=summary,
                    source=source,
                    url=url,
                    published_at=published_at.isoformat(),
                    relevance_score=relevance,
                    sentiment=sentiment,
                )
            )

    except ET.ParseError as e:
        logger.warning("xml_parse_error", error=str(e))

    return items


def _parse_rfc2822_date(date_str: str) -> datetime:
    """Parse RFC 2822 date format.

    Args:
        date_str: Date string in RFC 2822 format.

    Returns:
        Parsed datetime.
    """
    # Try common formats
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue

    # Default to now if parsing fails
    return datetime.now(UTC)


def _strip_html(html: str) -> str:
    """Strip HTML tags from a string.

    Args:
        html: HTML string.

    Returns:
        Plain text string.
    """
    return re.sub(r"<[^>]+>", "", html).strip()


def _analyze_sentiment(text: str) -> Literal["positive", "negative", "neutral"]:
    """Simple sentiment analysis based on keywords.

    Args:
        text: Text to analyze.

    Returns:
        Sentiment classification.
    """
    text_lower = text.lower()

    positive_words = [
        "surge",
        "rally",
        "jump",
        "rise",
        "gain",
        "beat",
        "bullish",
        "optimistic",
        "growth",
        "profit",
        "success",
        "record",
        "upgrade",
    ]
    negative_words = [
        "drop",
        "tumble",
        "fall",
        "slide",
        "sink",
        "bearish",
        "disappointing",
        "loss",
        "decline",
        "downgrade",
        "crash",
        "plunge",
        "miss",
    ]

    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    return "neutral"


# ============================================================================
# News Search Tool
# ============================================================================


class NewsSearchTool(BaseTool[NewsSearchInput, NewsSearchOutput]):
    """Tool for searching financial news.

    Features:
    - Real-time news via Polygon.io (primary) or Google News RSS (fallback)
    - Rate-limited queue for Polygon's 5 req/min free tier
    - Mock news generator for testing
    - Relevance scoring based on query and symbols
    - Date filtering
    - Sentiment analysis
    """

    name = "search_news"
    description = (
        "Searches for recent financial news related to a query or stock symbols. "
        "Returns relevant news items with titles, summaries, and relevance scores."
    )

    GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

    def __init__(
        self,
        use_mock: bool | None = None,
        fallback_to_mock: bool | None = None,
        http_client: httpx.AsyncClient | None = None,
        router: "DataSourceRouter | None" = None,
    ) -> None:
        """Initialize the NewsSearchTool.

        Args:
            use_mock: Whether to use mock data instead of real API.
            fallback_to_mock: Whether to fall back to mock on API failure.
            http_client: Optional custom HTTP client for Google News fallback.
            router: Optional DataSourceRouter for Polygon.io integration.
        """
        super().__init__(use_mock=use_mock, fallback_to_mock=fallback_to_mock)
        self._http_client = http_client
        self._router = router

    @property
    def input_schema(self) -> type[NewsSearchInput]:
        return NewsSearchInput

    @property
    def output_schema(self) -> type[NewsSearchOutput]:
        return NewsSearchOutput

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _execute_real(self, input_data: NewsSearchInput) -> NewsSearchOutput:
        """Fetch real news with Polygon → Google News fallback chain.

        Args:
            input_data: Validated input with query and filters.

        Returns:
            News search output.

        Raises:
            ValueError: If all sources fail.
        """
        # Build search query
        search_terms = [input_data.query]
        if input_data.symbols:
            search_terms.extend(input_data.symbols)
        full_query = " ".join(search_terms)

        self._logger.info("searching_news", query=full_query)

        # Try Polygon first if router is available
        if self._router:
            try:
                # Use first symbol if available, otherwise None for market news
                symbol = input_data.symbols[0] if input_data.symbols else None
                result = await self._router.search_news(symbol, input_data.max_results)

                if result.news:
                    # Convert Polygon news to our format
                    items = []
                    for article in result.news:
                        # Calculate relevance
                        relevance = calculate_relevance_score(
                            article.title,
                            article.summary or "",
                            input_data.query,
                            input_data.symbols,
                        )

                        items.append(
                            NewsItem(
                                title=article.title,
                                summary=article.summary or article.title,
                                source=article.publisher,
                                url=article.url,
                                published_at=article.published_at.isoformat(),
                                relevance_score=relevance,
                                sentiment=article.sentiment,
                            )
                        )

                    # Sort by relevance
                    items.sort(key=lambda x: x.relevance_score, reverse=True)

                    return NewsSearchOutput(
                        query=input_data.query,
                        items=items[: input_data.max_results],
                        total_found=len(items),
                        source="real",
                    )

            except Exception as e:
                self._logger.warning("polygon_news_failed", error=str(e))
                # Fall through to Google News

        # Fallback to Google News RSS
        return await self._fetch_google_news(input_data)

    async def _fetch_google_news(self, input_data: NewsSearchInput) -> NewsSearchOutput:
        """Fetch news from Google News RSS.

        Args:
            input_data: Validated input with query and filters.

        Returns:
            News search output.

        Raises:
            ValueError: If the search fails.
        """
        search_terms = [input_data.query]
        if input_data.symbols:
            search_terms.extend(input_data.symbols)
        full_query = " ".join(search_terms)

        client = await self._get_http_client()
        url = f"{self.GOOGLE_NEWS_RSS_URL}?q={quote(full_query)}&hl=en-US&gl=US&ceid=US:en"

        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise ValueError(f"Failed to fetch news: {e}") from e

        # Parse RSS feed
        items = parse_google_news_rss(
            response.text,
            input_data.query,
            input_data.symbols,
        )

        # Filter by date
        cutoff = datetime.now(UTC) - timedelta(days=input_data.days_back)
        filtered_items: list[NewsItem] = []
        for item in items:
            try:
                pub_date = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
                if pub_date >= cutoff:
                    filtered_items.append(item)
            except ValueError:
                # Include items with unparseable dates
                filtered_items.append(item)

        # Sort by relevance and limit
        filtered_items.sort(key=lambda x: x.relevance_score, reverse=True)
        result_items = filtered_items[: input_data.max_results]

        return NewsSearchOutput(
            query=input_data.query,
            items=result_items,
            total_found=len(items),
            source="real",
        )

    async def _execute_mock(self, input_data: NewsSearchInput) -> NewsSearchOutput:
        """Return mock news data.

        Args:
            input_data: Validated input with query and filters.

        Returns:
            Mock news search output.
        """
        items = generate_mock_news(
            query=input_data.query,
            symbols=input_data.symbols,
            days_back=input_data.days_back,
            max_results=input_data.max_results,
        )

        return NewsSearchOutput(
            query=input_data.query,
            items=items,
            total_found=len(items),
            source="mock",
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
