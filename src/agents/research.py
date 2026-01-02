"""Research Agent for gathering market data and information.

This module implements the ResearchAgent that collects market data,
news, and other financial information for portfolio analysis.
"""

from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import AgentState, BaseAgent
from src.observability.tracing import traced_agent
from src.tools.base import BaseTool, ToolInput, ToolOutput, ToolRegistry

logger = structlog.get_logger(__name__)


# ============================================================================
# Output Schema
# ============================================================================


class NewsItem(BaseModel):
    """A single news item."""

    title: str
    source: str
    url: str | None = None
    summary: str | None = None
    sentiment: str | None = None  # positive, negative, neutral
    published_at: str | None = None


class SymbolData(BaseModel):
    """Market data for a single symbol."""

    symbol: str
    price: float | None = None
    change_percent: float | None = None
    volume: int | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None


class ResearchOutput(BaseModel):
    """Structured output from the Research Agent.

    Contains all gathered market data, news, and analysis summary.
    """

    symbols_analyzed: list[str] = Field(default_factory=list)
    market_data: dict[str, SymbolData] = Field(default_factory=dict)
    news_items: list[NewsItem] = Field(default_factory=list)
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ============================================================================
# Placeholder Tool Inputs/Outputs (to be replaced by actual tool implementations)
# ============================================================================


class MarketDataInput(ToolInput):
    """Input for market data tool."""

    symbol: str


class MarketDataOutput(ToolOutput):
    """Output from market data tool."""

    symbol: str
    price: float
    change_percent: float
    volume: int
    market_cap: float | None = None
    pe_ratio: float | None = None
    dividend_yield: float | None = None


class NewsSearchInput(ToolInput):
    """Input for news search tool."""

    query: str
    max_results: int = 5


class NewsSearchOutput(ToolOutput):
    """Output from news search tool."""

    items: list[NewsItem] = Field(default_factory=list)


# ============================================================================
# Placeholder Tools (to be replaced by actual implementations in #10 and #11)
# ============================================================================


class PlaceholderMarketDataTool(BaseTool[MarketDataInput, MarketDataOutput]):
    """Placeholder market data tool for testing.

    Will be replaced by actual implementation in issue #10.
    """

    name = "get_market_data"
    description = "Fetches current market data for a stock symbol including price, volume, and key metrics"

    @property
    def input_schema(self) -> type[MarketDataInput]:
        return MarketDataInput

    @property
    def output_schema(self) -> type[MarketDataOutput]:
        return MarketDataOutput

    async def _execute_real(self, input_data: MarketDataInput) -> MarketDataOutput:
        """Real implementation - placeholder raises to trigger mock fallback."""
        raise NotImplementedError("Real market data API not yet implemented")

    async def _execute_mock(self, input_data: MarketDataInput) -> MarketDataOutput:
        """Mock implementation with sample data."""
        # Mock data based on symbol
        mock_data = {
            "AAPL": {"price": 178.50, "change": 1.25, "volume": 52_000_000, "cap": 2.8e12, "pe": 28.5},
            "GOOGL": {"price": 141.80, "change": -0.45, "volume": 21_000_000, "cap": 1.8e12, "pe": 25.2},
            "MSFT": {"price": 378.90, "change": 0.82, "volume": 18_000_000, "cap": 2.9e12, "pe": 35.1},
        }

        data = mock_data.get(
            input_data.symbol.upper(),
            {"price": 100.0, "change": 0.0, "volume": 1_000_000, "cap": 1e9, "pe": 20.0},
        )

        return MarketDataOutput(
            symbol=input_data.symbol.upper(),
            price=data["price"],
            change_percent=data["change"],
            volume=int(data["volume"]),
            market_cap=data["cap"],
            pe_ratio=data["pe"],
        )


class PlaceholderNewsSearchTool(BaseTool[NewsSearchInput, NewsSearchOutput]):
    """Placeholder news search tool for testing.

    Will be replaced by actual implementation in issue #11.
    """

    name = "search_news"
    description = "Searches for recent financial news related to a query or stock symbol"

    @property
    def input_schema(self) -> type[NewsSearchInput]:
        return NewsSearchInput

    @property
    def output_schema(self) -> type[NewsSearchOutput]:
        return NewsSearchOutput

    async def _execute_real(self, input_data: NewsSearchInput) -> NewsSearchOutput:
        """Real implementation - placeholder raises to trigger mock fallback."""
        raise NotImplementedError("Real news search API not yet implemented")

    async def _execute_mock(self, input_data: NewsSearchInput) -> NewsSearchOutput:
        """Mock implementation with sample news."""
        items = [
            NewsItem(
                title=f"Market Update: {input_data.query} shows strong momentum",
                source="Financial Times",
                url="https://ft.com/example",
                summary=f"Analysis of recent {input_data.query} performance and outlook.",
                sentiment="positive",
                published_at="2024-01-15T10:30:00Z",
            ),
            NewsItem(
                title=f"Analyst Report: {input_data.query} earnings preview",
                source="Bloomberg",
                url="https://bloomberg.com/example",
                summary="Upcoming earnings expectations and key metrics to watch.",
                sentiment="neutral",
                published_at="2024-01-15T09:15:00Z",
            ),
        ]
        return NewsSearchOutput(items=items[: input_data.max_results])


# ============================================================================
# Research Agent
# ============================================================================


class ResearchAgent(BaseAgent):
    """Agent that gathers market data and financial information.

    The Research Agent is the first step in the portfolio analysis workflow.
    It collects market data, news, and other relevant information for the
    symbols in the portfolio.

    Tools:
        - get_market_data: Fetches current market data for symbols
        - search_news: Searches for relevant financial news

    Output:
        ResearchOutput containing market data, news items, and summary
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        use_mock_tools: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Research Agent.

        Args:
            tool_registry: Optional custom tool registry. If not provided,
                creates a new one with placeholder tools.
            use_mock_tools: Whether to use mock mode for tools.
            **kwargs: Additional arguments passed to BaseAgent.
        """
        super().__init__(**kwargs)

        # Set up tool registry
        if tool_registry is not None:
            self._tool_registry = tool_registry
        else:
            self._tool_registry = ToolRegistry()
            self._tool_registry.register(
                PlaceholderMarketDataTool(use_mock=use_mock_tools)
            )
            self._tool_registry.register(
                PlaceholderNewsSearchTool(use_mock=use_mock_tools)
            )

    @property
    def name(self) -> str:
        return "research"

    @property
    def description(self) -> str:
        return "Gathers market data and financial information for portfolio analysis"

    @property
    def system_prompt(self) -> str:
        return """You are a financial research agent specialized in gathering market data and information.

Your role is to:
1. Collect current market data for requested stock symbols
2. Search for relevant financial news
3. Compile a structured research document

Available tools:
- get_market_data: Fetches current price, volume, and key metrics for a stock symbol
- search_news: Searches for recent financial news related to a query

Guidelines:
- Always gather market data for all requested symbols
- Search for news related to the portfolio or specific symbols
- Report any errors or missing data clearly
- Provide a brief summary of your findings

Output your findings in a structured format with:
- Market data for each symbol
- Relevant news items
- A summary of key observations
- List of data sources used"""

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return tools in Anthropic format."""
        return self._tool_registry.to_anthropic_tools()

    def get_tool(self, name: str) -> BaseTool[Any, Any]:
        """Get a tool by name from the registry.

        Args:
            name: Name of the tool.

        Returns:
            The tool instance.
        """
        return self._tool_registry.get(name)

    @traced_agent("research_agent")
    async def invoke(self, state: AgentState) -> AgentState:
        """Execute the research agent workflow.

        Args:
            state: Current workflow state containing:
                - context.symbols: List of symbols to analyze
                - context.portfolio: Optional portfolio details

        Returns:
            Updated state with research results in context.research
        """
        self._logger.info("research_invoke_start", context_keys=list(state.context.keys()))

        # Extract symbols from state
        symbols: list[str] = state.context.get("symbols", [])
        if not symbols:
            self._logger.warning("no_symbols_provided")
            state.errors.append("ResearchAgent: No symbols provided in context")
            return state

        # Initialize output
        output = ResearchOutput(symbols_analyzed=symbols)

        # Gather market data for each symbol
        market_data_tool = self.get_tool("get_market_data")
        for symbol in symbols:
            try:
                result = await market_data_tool.execute({"symbol": symbol})
                output.market_data[symbol] = SymbolData(
                    symbol=result.symbol,
                    price=result.price,
                    change_percent=result.change_percent,
                    volume=result.volume,
                    market_cap=result.market_cap,
                    pe_ratio=result.pe_ratio,
                    dividend_yield=result.dividend_yield,
                )
                output.sources.append(f"Market data for {symbol}")
                self._logger.debug("market_data_collected", symbol=symbol)
            except Exception as e:
                error_msg = f"Failed to get market data for {symbol}: {e}"
                output.errors.append(error_msg)
                self._logger.warning("market_data_failed", symbol=symbol, error=str(e))

        # Search for relevant news
        news_tool = self.get_tool("search_news")
        try:
            # Search for portfolio-related news
            query = " ".join(symbols[:3])  # Use first 3 symbols as query
            news_result = await news_tool.execute({"query": query, "max_results": 5})
            output.news_items = news_result.items
            output.sources.append(f"News search for: {query}")
            self._logger.debug("news_collected", item_count=len(news_result.items))
        except Exception as e:
            error_msg = f"Failed to search news: {e}"
            output.errors.append(error_msg)
            self._logger.warning("news_search_failed", error=str(e))

        # Generate summary
        output.summary = self._generate_summary(output)

        # Update state with research results
        state.context["research"] = output.model_dump()
        state.messages.append({
            "role": "assistant",
            "content": f"Research completed for {len(symbols)} symbols. "
                       f"Collected {len(output.market_data)} market data entries "
                       f"and {len(output.news_items)} news items.",
        })

        self._logger.info(
            "research_invoke_complete",
            symbols_count=len(symbols),
            market_data_count=len(output.market_data),
            news_count=len(output.news_items),
            error_count=len(output.errors),
        )

        return state

    def _generate_summary(self, output: ResearchOutput) -> str:
        """Generate a summary of research findings.

        Args:
            output: The research output to summarize.

        Returns:
            Summary string.
        """
        parts = []

        # Market data summary
        if output.market_data:
            gainers = []
            losers = []
            for symbol, data in output.market_data.items():
                if data.change_percent is not None:
                    if data.change_percent > 0:
                        gainers.append((symbol, data.change_percent))
                    elif data.change_percent < 0:
                        losers.append((symbol, data.change_percent))

            if gainers:
                top_gainer = max(gainers, key=lambda x: x[1])
                parts.append(f"Top gainer: {top_gainer[0]} (+{top_gainer[1]:.2f}%)")
            if losers:
                top_loser = min(losers, key=lambda x: x[1])
                parts.append(f"Top loser: {top_loser[0]} ({top_loser[1]:.2f}%)")

        # News summary
        if output.news_items:
            positive_count = sum(1 for n in output.news_items if n.sentiment == "positive")
            negative_count = sum(1 for n in output.news_items if n.sentiment == "negative")
            parts.append(
                f"News sentiment: {positive_count} positive, {negative_count} negative, "
                f"{len(output.news_items) - positive_count - negative_count} neutral"
            )

        # Error summary
        if output.errors:
            parts.append(f"Encountered {len(output.errors)} error(s) during research")

        return ". ".join(parts) if parts else "Research completed successfully."
