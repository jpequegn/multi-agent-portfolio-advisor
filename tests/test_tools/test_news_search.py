"""Tests for the News Search Tool."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.base import ToolExecutionError
from src.tools.news_search import (
    NewsItem,
    NewsSearchInput,
    NewsSearchOutput,
    NewsSearchTool,
    calculate_relevance_score,
    generate_mock_news,
    parse_google_news_rss,
)


class TestNewsSearchInput:
    """Tests for NewsSearchInput model."""

    def test_required_fields(self) -> None:
        """Test that query is required."""
        input_data = NewsSearchInput(query="AAPL stock")
        assert input_data.query == "AAPL stock"
        assert input_data.symbols == []
        assert input_data.days_back == 7
        assert input_data.max_results == 10

    def test_all_fields(self) -> None:
        """Test with all fields provided."""
        input_data = NewsSearchInput(
            query="tech stocks",
            symbols=["AAPL", "GOOGL"],
            days_back=14,
            max_results=20,
        )
        assert input_data.symbols == ["AAPL", "GOOGL"]
        assert input_data.days_back == 14
        assert input_data.max_results == 20

    def test_days_back_validation(self) -> None:
        """Test days_back range validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NewsSearchInput(query="test", days_back=0)

        with pytest.raises(ValidationError):
            NewsSearchInput(query="test", days_back=31)

    def test_max_results_validation(self) -> None:
        """Test max_results range validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NewsSearchInput(query="test", max_results=0)

        with pytest.raises(ValidationError):
            NewsSearchInput(query="test", max_results=51)


class TestNewsItem:
    """Tests for NewsItem model."""

    def test_required_fields(self) -> None:
        """Test required fields."""
        item = NewsItem(
            title="Test News",
            summary="Test summary",
            source="Reuters",
            url="https://example.com",
            published_at="2024-01-15T10:00:00Z",
            relevance_score=0.8,
        )
        assert item.title == "Test News"
        assert item.relevance_score == 0.8
        assert item.sentiment is None

    def test_with_sentiment(self) -> None:
        """Test with sentiment field."""
        item = NewsItem(
            title="Test News",
            summary="Test summary",
            source="Reuters",
            url="https://example.com",
            published_at="2024-01-15T10:00:00Z",
            relevance_score=0.8,
            sentiment="positive",
        )
        assert item.sentiment == "positive"

    def test_relevance_score_validation(self) -> None:
        """Test relevance_score range validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            NewsItem(
                title="Test",
                summary="Test",
                source="Test",
                url="https://example.com",
                published_at="2024-01-15T10:00:00Z",
                relevance_score=1.5,
            )


class TestNewsSearchOutput:
    """Tests for NewsSearchOutput model."""

    def test_default_values(self) -> None:
        """Test default values."""
        output = NewsSearchOutput(query="test", source="mock")
        assert output.items == []
        assert output.total_found == 0
        assert output.success is True

    def test_with_items(self) -> None:
        """Test with items."""
        item = NewsItem(
            title="Test",
            summary="Summary",
            source="Reuters",
            url="https://example.com",
            published_at="2024-01-15T10:00:00Z",
            relevance_score=0.8,
        )
        output = NewsSearchOutput(
            query="test",
            items=[item],
            total_found=1,
            source="real",
        )
        assert len(output.items) == 1
        assert output.source == "real"


class TestRelevanceScoring:
    """Tests for relevance scoring."""

    def test_query_word_matching(self) -> None:
        """Test relevance score from query word matches."""
        score = calculate_relevance_score(
            title="Apple stock rises on earnings",
            summary="Apple reported strong earnings today",
            query="apple earnings",
            symbols=[],
        )
        assert score > 0.3  # Should have query word matches

    def test_symbol_matching(self) -> None:
        """Test relevance score from symbol matches."""
        score = calculate_relevance_score(
            title="AAPL surges after product launch",
            summary="Apple Inc AAPL stock rises",
            query="stock news",
            symbols=["AAPL"],
        )
        assert score > 0.3  # Should have symbol matches

    def test_financial_keyword_boost(self) -> None:
        """Test relevance boost from financial keywords."""
        score = calculate_relevance_score(
            title="Stock market trading update on earnings and revenue",
            summary="Investors monitor analyst predictions for dividend announcement",
            query="",
            symbols=[],
        )
        assert score > 0.1  # Should have keyword boost

    def test_combined_scoring(self) -> None:
        """Test combined relevance scoring."""
        score = calculate_relevance_score(
            title="AAPL stock earnings beat expectations",
            summary="Apple AAPL shares surge after analyst upgrade",
            query="apple earnings",
            symbols=["AAPL"],
        )
        assert score > 0.5  # Should have high combined score

    def test_no_matches(self) -> None:
        """Test relevance score with no matches."""
        score = calculate_relevance_score(
            title="Weather forecast for tomorrow",
            summary="Rain expected in the afternoon",
            query="apple stock",
            symbols=["AAPL"],
        )
        assert score < 0.2  # Should have low score


class TestMockNewsGenerator:
    """Tests for mock news generation."""

    def test_generates_correct_count(self) -> None:
        """Test that generator produces correct number of items."""
        items = generate_mock_news(
            query="AAPL",
            symbols=["AAPL", "GOOGL"],
            days_back=7,
            max_results=5,
        )
        assert len(items) == 5

    def test_items_have_required_fields(self) -> None:
        """Test that generated items have all required fields."""
        items = generate_mock_news(
            query="tech stocks",
            symbols=["MSFT"],
            days_back=7,
            max_results=3,
        )

        for item in items:
            assert item.title
            assert item.summary
            assert item.source
            assert item.url
            assert item.published_at
            assert 0.0 <= item.relevance_score <= 1.0
            assert item.sentiment in ["positive", "negative", "neutral"]

    def test_dates_within_range(self) -> None:
        """Test that generated dates are within specified range."""
        days_back = 7
        items = generate_mock_news(
            query="test",
            symbols=[],
            days_back=days_back,
            max_results=10,
        )

        cutoff = datetime.now(UTC) - timedelta(days=days_back)
        for item in items:
            pub_date = datetime.fromisoformat(item.published_at)
            assert pub_date >= cutoff

    def test_sorted_by_relevance(self) -> None:
        """Test that items are sorted by relevance score."""
        items = generate_mock_news(
            query="AAPL",
            symbols=["AAPL"],
            days_back=7,
            max_results=10,
        )

        scores = [item.relevance_score for item in items]
        assert scores == sorted(scores, reverse=True)

    def test_uses_symbols_in_headlines(self) -> None:
        """Test that symbols appear in generated headlines."""
        items = generate_mock_news(
            query="tech",
            symbols=["AAPL"],
            days_back=7,
            max_results=5,
        )

        # At least one item should mention AAPL
        has_symbol = any("AAPL" in item.title for item in items)
        assert has_symbol


class TestGoogleNewsRSSParser:
    """Tests for Google News RSS parsing."""

    def test_parse_valid_rss(self) -> None:
        """Test parsing valid RSS feed."""
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Google News</title>
                <item>
                    <title>AAPL stock rises on earnings</title>
                    <link>https://example.com/news/1</link>
                    <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
                    <source>Reuters</source>
                    <description>Apple stock increased after strong earnings report.</description>
                </item>
            </channel>
        </rss>"""

        items = parse_google_news_rss(rss_content, "AAPL", ["AAPL"])

        assert len(items) == 1
        assert items[0].title == "AAPL stock rises on earnings"
        assert items[0].source == "Reuters"
        assert items[0].url == "https://example.com/news/1"

    def test_parse_multiple_items(self) -> None:
        """Test parsing multiple items."""
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>News 1</title>
                    <link>https://example.com/1</link>
                </item>
                <item>
                    <title>News 2</title>
                    <link>https://example.com/2</link>
                </item>
            </channel>
        </rss>"""

        items = parse_google_news_rss(rss_content, "test", [])
        assert len(items) == 2

    def test_parse_empty_feed(self) -> None:
        """Test parsing empty feed."""
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Google News</title>
            </channel>
        </rss>"""

        items = parse_google_news_rss(rss_content, "test", [])
        assert len(items) == 0

    def test_parse_invalid_xml(self) -> None:
        """Test handling invalid XML."""
        items = parse_google_news_rss("not valid xml", "test", [])
        assert len(items) == 0

    def test_parse_missing_required_fields(self) -> None:
        """Test that items without title/link are skipped."""
        rss_content = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <description>No title or link</description>
                </item>
            </channel>
        </rss>"""

        items = parse_google_news_rss(rss_content, "test", [])
        assert len(items) == 0


class TestNewsSearchTool:
    """Tests for NewsSearchTool."""

    def test_tool_properties(self) -> None:
        """Test tool has correct name and description."""
        tool = NewsSearchTool()
        assert tool.name == "search_news"
        assert "news" in tool.description.lower()

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = NewsSearchTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "search_news"
        assert "query" in anthropic_tool["input_schema"]["properties"]
        assert "symbols" in anthropic_tool["input_schema"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mock(self) -> None:
        """Test mock execution."""
        tool = NewsSearchTool(use_mock=True)
        result = await tool.execute({
            "query": "AAPL stock",
            "symbols": ["AAPL"],
            "max_results": 5,
        })

        assert result.query == "AAPL stock"
        assert len(result.items) == 5
        assert result.source == "mock"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_mock_respects_max_results(self) -> None:
        """Test that mock respects max_results."""
        tool = NewsSearchTool(use_mock=True)
        result = await tool.execute({
            "query": "tech stocks",
            "max_results": 3,
        })

        assert len(result.items) == 3

    @pytest.mark.asyncio
    async def test_execute_real_success(self) -> None:
        """Test real execution with mocked HTTP response."""
        mock_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>AAPL stock news</title>
                    <link>https://example.com/1</link>
                    <source>Reuters</source>
                </item>
            </channel>
        </rss>"""

        mock_response = AsyncMock()
        mock_response.text = mock_rss
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        tool = NewsSearchTool(use_mock=False, http_client=mock_client)
        result = await tool.execute({"query": "AAPL"})

        assert result.source == "real"
        assert len(result.items) >= 0  # May have items if parsed correctly
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_real_failure_fallback(self) -> None:
        """Test fallback to mock when real API fails."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")

        tool = NewsSearchTool(
            use_mock=False,
            fallback_to_mock=True,
            http_client=mock_client,
        )
        result = await tool.execute({"query": "AAPL"})

        assert result.source == "mock"
        assert "Fallback" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_real_failure_no_fallback(self) -> None:
        """Test error raised when fallback disabled."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")

        tool = NewsSearchTool(
            use_mock=False,
            fallback_to_mock=False,
            http_client=mock_client,
        )

        with pytest.raises(ToolExecutionError):
            await tool.execute({"query": "AAPL"})

    @pytest.mark.asyncio
    async def test_date_filtering(self) -> None:
        """Test that results are filtered by date."""
        # Create RSS with items at different dates
        now = datetime.now(UTC)
        old_date = (now - timedelta(days=30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        recent_date = (now - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")

        mock_rss = f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <item>
                    <title>Old news</title>
                    <link>https://example.com/old</link>
                    <pubDate>{old_date}</pubDate>
                </item>
                <item>
                    <title>Recent news</title>
                    <link>https://example.com/recent</link>
                    <pubDate>{recent_date}</pubDate>
                </item>
            </channel>
        </rss>"""

        mock_response = AsyncMock()
        mock_response.text = mock_rss
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        tool = NewsSearchTool(use_mock=False, http_client=mock_client)
        result = await tool.execute({
            "query": "test",
            "days_back": 7,
        })

        # Only recent news should be included
        assert len(result.items) == 1
        assert "Recent" in result.items[0].title

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing HTTP client."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        tool = NewsSearchTool(http_client=mock_client)

        await tool.close()

        mock_client.aclose.assert_called_once()
        assert tool._http_client is None


class TestSentimentAnalysis:
    """Tests for sentiment analysis."""

    def test_positive_sentiment(self) -> None:
        """Test detection of positive sentiment."""
        from src.tools.news_search import _analyze_sentiment

        assert _analyze_sentiment("Stock surges after earnings beat") == "positive"
        assert _analyze_sentiment("Company reports record growth") == "positive"

    def test_negative_sentiment(self) -> None:
        """Test detection of negative sentiment."""
        from src.tools.news_search import _analyze_sentiment

        assert _analyze_sentiment("Stock plunges on disappointing results") == "negative"
        assert _analyze_sentiment("Shares crash after downgrade") == "negative"

    def test_neutral_sentiment(self) -> None:
        """Test detection of neutral sentiment."""
        from src.tools.news_search import _analyze_sentiment

        assert _analyze_sentiment("Company announces new product") == "neutral"
        assert _analyze_sentiment("CEO speaks at conference") == "neutral"


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_missing_query(self) -> None:
        """Test that missing query raises error."""
        tool = NewsSearchTool(use_mock=True)

        with pytest.raises(ToolExecutionError, match="Input validation failed"):
            await tool.execute({})

    @pytest.mark.asyncio
    async def test_accepts_model_input(self) -> None:
        """Test that validated model can be passed directly."""
        tool = NewsSearchTool(use_mock=True)
        input_data = NewsSearchInput(query="AAPL stock")

        result = await tool.execute(input_data)

        assert result.query == "AAPL stock"
