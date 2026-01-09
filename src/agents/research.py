"""Research Agent for gathering market data and information.

This module implements the ResearchAgent that collects market data,
news, and other financial information for portfolio analysis using
the Polygon.io integration with intelligent fallback chain.
"""

from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import AgentState, BaseAgent
from src.cache.manager import CacheManager
from src.data.polygon import PolygonClient
from src.data.router import DataSourceRouter
from src.observability.tracing import traced_agent
from src.tools.base import BaseTool, ToolRegistry
from src.tools.market_data import MarketDataTool
from src.tools.news_search import NewsSearchTool

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
    data_source: str | None = None  # polygon, yahoo, cache, mock


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
# Research Agent
# ============================================================================


class ResearchAgent(BaseAgent):
    """Agent that gathers market data and financial information.

    The Research Agent is the first step in the portfolio analysis workflow.
    It collects market data, news, and other relevant information for the
    symbols in the portfolio.

    Data sources (in fallback order):
    1. Polygon.io - Primary source for real-time market data
    2. Yahoo Finance - Fallback when Polygon is unavailable
    3. Redis Cache - Fallback to cached data when APIs fail
    4. Mock Data - Last resort for testing/resilience

    Tools:
        - get_market_data: Fetches current market data for symbols
        - search_news: Searches for relevant financial news

    Output:
        ResearchOutput containing market data, news items, and summary
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        use_mock_tools: bool = False,
        polygon_client: PolygonClient | None = None,
        cache_manager: CacheManager | None = None,
        data_router: DataSourceRouter | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Research Agent.

        Args:
            tool_registry: Optional custom tool registry. If not provided,
                creates a new one with real tools.
            use_mock_tools: Whether to use mock mode for tools (default False).
            polygon_client: Optional Polygon.io client.
            cache_manager: Optional Redis cache manager.
            data_router: Optional DataSourceRouter (shared across tools).
            **kwargs: Additional arguments passed to BaseAgent.
        """
        super().__init__(**kwargs)

        # Create shared data router if not provided
        if data_router is None:
            data_router = DataSourceRouter(
                polygon=polygon_client or PolygonClient(),
                cache=cache_manager,
            )

        self._data_router = data_router

        # Set up tool registry
        if tool_registry is not None:
            self._tool_registry = tool_registry
        else:
            self._tool_registry = ToolRegistry()

            # Register real tools with shared router
            self._tool_registry.register(
                MarketDataTool(
                    use_mock=use_mock_tools,
                    router=data_router,
                )
            )
            self._tool_registry.register(
                NewsSearchTool(
                    use_mock=use_mock_tools,
                    router=data_router,
                )
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
  (Uses Polygon.io as primary source with Yahoo Finance fallback)
- search_news: Searches for recent financial news related to a query
  (Uses Polygon.io as primary source with Google News fallback)

Guidelines:
- Always gather market data for all requested symbols
- Search for news related to the portfolio or specific symbols
- Report any errors or missing data clearly
- Note the data source for transparency (polygon, yahoo, cache, mock)
- Provide a brief summary of your findings

Output your findings in a structured format with:
- Market data for each symbol (with source attribution)
- Relevant news items
- A summary of key observations
- List of data sources used"""

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return tools in Anthropic format."""
        return self._tool_registry.to_anthropic_tools()

    @property
    def data_router(self) -> DataSourceRouter:
        """Get the data source router for direct access."""
        return self._data_router

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
                    data_source=result.source,
                )
                output.sources.append(f"Market data for {symbol} ({result.source})")
                self._logger.debug(
                    "market_data_collected",
                    symbol=symbol,
                    source=result.source,
                )
            except Exception as e:
                error_msg = f"Failed to get market data for {symbol}: {e}"
                output.errors.append(error_msg)
                self._logger.warning("market_data_failed", symbol=symbol, error=str(e))

        # Search for relevant news
        news_tool = self.get_tool("search_news")
        try:
            # Search for portfolio-related news
            query = " ".join(symbols[:3])  # Use first 3 symbols as query
            news_result = await news_tool.execute({
                "query": query,
                "symbols": symbols[:3],
                "max_results": 5,
            })

            # Convert news items to our format
            for item in news_result.items:
                output.news_items.append(
                    NewsItem(
                        title=item.title,
                        source=item.source,
                        url=item.url,
                        summary=item.summary,
                        sentiment=item.sentiment,
                        published_at=item.published_at,
                    )
                )

            output.sources.append(f"News search for: {query} ({news_result.source})")
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

        # Data source summary
        sources_used = set()
        for symbol_data in output.market_data.values():
            if symbol_data.data_source:
                sources_used.add(symbol_data.data_source)
        if sources_used:
            parts.append(f"Data sources: {', '.join(sorted(sources_used))}")

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

    async def close(self) -> None:
        """Close underlying connections."""
        if self._data_router:
            await self._data_router.close()
