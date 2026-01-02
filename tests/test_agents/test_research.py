"""Tests for the Research Agent."""

import pytest

from src.agents.base import AgentState
from src.agents.research import (
    MarketDataInput,
    MarketDataOutput,
    NewsItem,
    NewsSearchInput,
    NewsSearchOutput,
    PlaceholderMarketDataTool,
    PlaceholderNewsSearchTool,
    ResearchAgent,
    ResearchOutput,
    SymbolData,
)
from src.tools.base import ToolRegistry


class TestNewsItem:
    """Tests for NewsItem model."""

    def test_required_fields(self) -> None:
        """Test that required fields work."""
        item = NewsItem(title="Test News", source="Test Source")
        assert item.title == "Test News"
        assert item.source == "Test Source"
        assert item.url is None
        assert item.summary is None
        assert item.sentiment is None
        assert item.published_at is None

    def test_all_fields(self) -> None:
        """Test with all fields."""
        item = NewsItem(
            title="Market Update",
            source="Bloomberg",
            url="https://example.com",
            summary="Market is up",
            sentiment="positive",
            published_at="2024-01-15T10:00:00Z",
        )
        assert item.url == "https://example.com"
        assert item.sentiment == "positive"


class TestSymbolData:
    """Tests for SymbolData model."""

    def test_required_fields(self) -> None:
        """Test that only symbol is required."""
        data = SymbolData(symbol="AAPL")
        assert data.symbol == "AAPL"
        assert data.price is None
        assert data.change_percent is None
        assert data.volume is None

    def test_all_fields(self) -> None:
        """Test with all fields."""
        data = SymbolData(
            symbol="AAPL",
            price=178.50,
            change_percent=1.25,
            volume=52_000_000,
            market_cap=2.8e12,
            pe_ratio=28.5,
            dividend_yield=0.5,
        )
        assert data.price == 178.50
        assert data.market_cap == 2.8e12


class TestResearchOutput:
    """Tests for ResearchOutput model."""

    def test_default_values(self) -> None:
        """Test default values."""
        output = ResearchOutput()
        assert output.symbols_analyzed == []
        assert output.market_data == {}
        assert output.news_items == []
        assert output.summary == ""
        assert output.sources == []
        assert output.errors == []

    def test_with_data(self) -> None:
        """Test with populated data."""
        output = ResearchOutput(
            symbols_analyzed=["AAPL", "GOOGL"],
            market_data={
                "AAPL": SymbolData(symbol="AAPL", price=178.50),
            },
            news_items=[
                NewsItem(title="Test", source="Source"),
            ],
            summary="Test summary",
            sources=["Source 1"],
            errors=[],
        )
        assert len(output.symbols_analyzed) == 2
        assert "AAPL" in output.market_data
        assert len(output.news_items) == 1


class TestPlaceholderMarketDataTool:
    """Tests for PlaceholderMarketDataTool."""

    @pytest.mark.asyncio
    async def test_mock_known_symbol(self) -> None:
        """Test mock data for known symbols."""
        tool = PlaceholderMarketDataTool(use_mock=True)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.price == 178.50
        assert result.change_percent == 1.25
        assert result.volume == 52_000_000

    @pytest.mark.asyncio
    async def test_mock_unknown_symbol(self) -> None:
        """Test mock data for unknown symbols returns defaults."""
        tool = PlaceholderMarketDataTool(use_mock=True)
        result = await tool.execute({"symbol": "UNKNOWN"})

        assert result.symbol == "UNKNOWN"
        assert result.price == 100.0
        assert result.change_percent == 0.0

    @pytest.mark.asyncio
    async def test_real_raises_not_implemented(self) -> None:
        """Test real implementation raises NotImplementedError."""
        tool = PlaceholderMarketDataTool(use_mock=False, fallback_to_mock=False)

        from src.tools.base import ToolExecutionError

        with pytest.raises(ToolExecutionError):
            await tool.execute({"symbol": "AAPL"})

    @pytest.mark.asyncio
    async def test_fallback_to_mock(self) -> None:
        """Test fallback to mock when real fails."""
        tool = PlaceholderMarketDataTool(use_mock=False, fallback_to_mock=True)
        result = await tool.execute({"symbol": "AAPL"})

        assert result.symbol == "AAPL"
        assert result.success is True
        assert "Fallback to mock" in (result.error or "")

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = PlaceholderMarketDataTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "get_market_data"
        assert "market data" in anthropic_tool["description"].lower()
        assert "symbol" in anthropic_tool["input_schema"]["properties"]


class TestPlaceholderNewsSearchTool:
    """Tests for PlaceholderNewsSearchTool."""

    @pytest.mark.asyncio
    async def test_mock_returns_news(self) -> None:
        """Test mock returns news items."""
        tool = PlaceholderNewsSearchTool(use_mock=True)
        result = await tool.execute({"query": "AAPL", "max_results": 5})

        assert len(result.items) <= 5
        assert all(isinstance(item, NewsItem) for item in result.items)
        assert result.items[0].source == "Financial Times"

    @pytest.mark.asyncio
    async def test_mock_respects_max_results(self) -> None:
        """Test mock respects max_results parameter."""
        tool = PlaceholderNewsSearchTool(use_mock=True)
        result = await tool.execute({"query": "AAPL", "max_results": 1})

        assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_real_raises_not_implemented(self) -> None:
        """Test real implementation raises NotImplementedError."""
        tool = PlaceholderNewsSearchTool(use_mock=False, fallback_to_mock=False)

        from src.tools.base import ToolExecutionError

        with pytest.raises(ToolExecutionError):
            await tool.execute({"query": "AAPL"})

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = PlaceholderNewsSearchTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "search_news"
        assert "news" in anthropic_tool["description"].lower()


class TestResearchAgent:
    """Tests for ResearchAgent."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        agent = ResearchAgent()

        assert agent.name == "research"
        assert "market data" in agent.description.lower()
        # Default uses mock tools - verify tools are registered
        assert agent.get_tool("get_market_data") is not None
        assert agent.get_tool("search_news") is not None

    def test_initialization_custom_registry(self) -> None:
        """Test initialization with custom tool registry."""
        registry = ToolRegistry()
        registry.register(PlaceholderMarketDataTool(use_mock=True))
        registry.register(PlaceholderNewsSearchTool(use_mock=True))

        agent = ResearchAgent(tool_registry=registry)

        assert agent.get_tool("get_market_data") is not None
        assert agent.get_tool("search_news") is not None

    def test_system_prompt(self) -> None:
        """Test system prompt contains expected content."""
        agent = ResearchAgent()

        assert "research" in agent.system_prompt.lower()
        assert "market data" in agent.system_prompt.lower()
        assert "news" in agent.system_prompt.lower()

    def test_tools_property(self) -> None:
        """Test tools are in Anthropic format."""
        agent = ResearchAgent()
        tools = agent.tools

        assert len(tools) == 2
        tool_names = [t["name"] for t in tools]
        assert "get_market_data" in tool_names
        assert "search_news" in tool_names

    def test_get_tool(self) -> None:
        """Test getting a tool by name."""
        agent = ResearchAgent()

        market_tool = agent.get_tool("get_market_data")
        assert market_tool.name == "get_market_data"

        news_tool = agent.get_tool("search_news")
        assert news_tool.name == "search_news"

    @pytest.mark.asyncio
    async def test_invoke_with_symbols(self) -> None:
        """Test invoke collects data for symbols."""
        agent = ResearchAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL", "GOOGL"]})

        result = await agent.invoke(state)

        assert "research" in result.context
        research = result.context["research"]

        assert research["symbols_analyzed"] == ["AAPL", "GOOGL"]
        assert "AAPL" in research["market_data"]
        assert "GOOGL" in research["market_data"]
        assert len(research["news_items"]) > 0
        assert len(research["sources"]) > 0
        assert research["summary"] != ""

    @pytest.mark.asyncio
    async def test_invoke_no_symbols(self) -> None:
        """Test invoke handles missing symbols gracefully."""
        agent = ResearchAgent(use_mock_tools=True)
        state = AgentState(context={})

        result = await agent.invoke(state)

        assert len(result.errors) == 1
        assert "No symbols provided" in result.errors[0]
        assert "research" not in result.context

    @pytest.mark.asyncio
    async def test_invoke_empty_symbols(self) -> None:
        """Test invoke handles empty symbols list."""
        agent = ResearchAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": []})

        result = await agent.invoke(state)

        assert len(result.errors) == 1
        assert "No symbols provided" in result.errors[0]

    @pytest.mark.asyncio
    async def test_invoke_adds_message(self) -> None:
        """Test invoke adds completion message."""
        agent = ResearchAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent.invoke(state)

        assert len(result.messages) == 1
        assert result.messages[0]["role"] == "assistant"
        assert "Research completed" in result.messages[0]["content"]

    @pytest.mark.asyncio
    async def test_invoke_generates_summary(self) -> None:
        """Test invoke generates meaningful summary."""
        agent = ResearchAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL", "GOOGL", "MSFT"]})

        result = await agent.invoke(state)

        research = result.context["research"]
        summary = research["summary"]

        # Should mention top gainer (AAPL has positive change)
        assert "gainer" in summary.lower() or "loser" in summary.lower()
        # Should mention news sentiment
        assert "sentiment" in summary.lower() or "news" in summary.lower()

    @pytest.mark.asyncio
    async def test_invoke_callable(self) -> None:
        """Test agent can be called as a function via BaseAgent.__call__."""
        agent = ResearchAgent(use_mock_tools=True)
        state = AgentState(context={"symbols": ["AAPL"]})

        result = await agent(state)

        assert "research" in result.context


class TestResearchAgentSummaryGeneration:
    """Tests for the _generate_summary method."""

    def test_summary_with_gainers_and_losers(self) -> None:
        """Test summary includes top gainer and loser."""
        agent = ResearchAgent(use_mock_tools=True)
        output = ResearchOutput(
            market_data={
                "AAPL": SymbolData(symbol="AAPL", change_percent=2.5),
                "GOOGL": SymbolData(symbol="GOOGL", change_percent=-1.5),
                "MSFT": SymbolData(symbol="MSFT", change_percent=0.5),
            },
        )

        summary = agent._generate_summary(output)

        assert "AAPL" in summary
        assert "+2.50%" in summary
        assert "GOOGL" in summary
        assert "-1.50%" in summary

    def test_summary_with_news_sentiment(self) -> None:
        """Test summary includes news sentiment counts."""
        agent = ResearchAgent(use_mock_tools=True)
        output = ResearchOutput(
            news_items=[
                NewsItem(title="Good news", source="S1", sentiment="positive"),
                NewsItem(title="Bad news", source="S2", sentiment="negative"),
                NewsItem(title="Neutral news", source="S3", sentiment="neutral"),
            ],
        )

        summary = agent._generate_summary(output)

        assert "1 positive" in summary
        assert "1 negative" in summary

    def test_summary_with_errors(self) -> None:
        """Test summary mentions errors."""
        agent = ResearchAgent(use_mock_tools=True)
        output = ResearchOutput(
            errors=["Error 1", "Error 2"],
        )

        summary = agent._generate_summary(output)

        assert "2 error" in summary.lower()

    def test_summary_empty_output(self) -> None:
        """Test summary for empty output."""
        agent = ResearchAgent(use_mock_tools=True)
        output = ResearchOutput()

        summary = agent._generate_summary(output)

        assert summary == "Research completed successfully."


class TestToolInputOutputModels:
    """Tests for tool input/output models."""

    def test_market_data_input(self) -> None:
        """Test MarketDataInput model."""
        input_data = MarketDataInput(symbol="AAPL")
        assert input_data.symbol == "AAPL"

    def test_market_data_output(self) -> None:
        """Test MarketDataOutput model."""
        output = MarketDataOutput(
            symbol="AAPL",
            price=178.50,
            change_percent=1.25,
            volume=52_000_000,
        )
        assert output.symbol == "AAPL"
        assert output.success is True

    def test_news_search_input(self) -> None:
        """Test NewsSearchInput model."""
        input_data = NewsSearchInput(query="AAPL")
        assert input_data.query == "AAPL"
        assert input_data.max_results == 5  # default

    def test_news_search_output(self) -> None:
        """Test NewsSearchOutput model."""
        output = NewsSearchOutput(items=[])
        assert output.items == []
        assert output.success is True
