"""Tests for custom attributes and metadata."""

from datetime import UTC, datetime

import pytest

from src.observability.metrics import (
    AgentMetadata,
    MetadataBuilder,
    RequestMetadata,
    Tag,
    TagManager,
    ToolMetadata,
    collect_tags,
    create_agent_metadata,
    create_request_metadata,
    create_tool_metadata,
)


# ============================================================================
# Tag Tests
# ============================================================================


class TestTag:
    """Tests for Tag enum."""

    def test_all_tags_have_string_values(self) -> None:
        """Test that all tags have hyphenated string values."""
        for tag in Tag:
            assert isinstance(tag.value, str)
            # Should be lowercase with hyphens
            assert tag.value == tag.value.lower()

    def test_tag_categories_exist(self) -> None:
        """Test that expected tag categories exist."""
        # Analysis types
        assert Tag.PORTFOLIO_ANALYSIS.value == "portfolio-analysis"
        assert Tag.RISK_ASSESSMENT.value == "risk-assessment"

        # Portfolio characteristics
        assert Tag.HIGH_VALUE_PORTFOLIO.value == "high-value-portfolio"
        assert Tag.LARGE_PORTFOLIO.value == "large-portfolio"

        # Execution characteristics
        assert Tag.MOCK_DATA_USED.value == "mock-data-used"
        assert Tag.CACHE_HIT.value == "cache-hit"

        # Error handling
        assert Tag.ERROR_RECOVERED.value == "error-recovered"
        assert Tag.RETRY_SUCCEEDED.value == "retry-succeeded"


# ============================================================================
# RequestMetadata Tests
# ============================================================================


class TestRequestMetadata:
    """Tests for RequestMetadata class."""

    def test_default_values(self) -> None:
        """Test RequestMetadata with default values."""
        meta = RequestMetadata()

        assert meta.portfolio_size == 0
        assert meta.portfolio_value == 0.0
        assert meta.symbols == []
        assert meta.user_id is None
        assert meta.user_tier == "standard"
        assert meta.request_type == "full_analysis"
        assert isinstance(meta.started_at, datetime)

    def test_all_fields_set(self) -> None:
        """Test RequestMetadata with all fields."""
        meta = RequestMetadata(
            portfolio_size=15,
            portfolio_value=150_000.00,
            symbols=["AAPL", "GOOGL", "MSFT"],
            user_id="user-123",
            user_tier="premium",
            request_type="risk_assessment",
            workflow_id="wf-456",
            custom={"source": "api"},
        )

        assert meta.portfolio_size == 15
        assert meta.portfolio_value == 150_000.00
        assert meta.symbols == ["AAPL", "GOOGL", "MSFT"]
        assert meta.user_id == "user-123"
        assert meta.user_tier == "premium"
        assert meta.request_type == "risk_assessment"
        assert meta.workflow_id == "wf-456"
        assert meta.custom == {"source": "api"}

    def test_to_dict(self) -> None:
        """Test RequestMetadata.to_dict method."""
        meta = RequestMetadata(
            portfolio_size=10,
            portfolio_value=50_000.00,
            symbols=["AAPL", "GOOGL"],
            user_tier="standard",
            custom={"extra": "value"},
        )

        result = meta.to_dict()

        assert result["portfolio_size"] == 10
        assert result["portfolio_value"] == 50_000.00
        assert result["symbols"] == ["AAPL", "GOOGL"]
        assert result["symbols_count"] == 2
        assert result["user_tier"] == "standard"
        assert result["extra"] == "value"
        assert "started_at" in result

    def test_get_auto_tags_high_value(self) -> None:
        """Test auto tags for high-value portfolio."""
        meta = RequestMetadata(
            portfolio_size=5,
            portfolio_value=150_000.00,
            user_tier="premium",
        )

        tags = meta.get_auto_tags()

        assert Tag.PORTFOLIO_ANALYSIS.value in tags
        assert Tag.HIGH_VALUE_PORTFOLIO.value in tags
        assert Tag.USER_PREMIUM.value in tags
        assert Tag.LARGE_PORTFOLIO.value not in tags

    def test_get_auto_tags_large_portfolio(self) -> None:
        """Test auto tags for large portfolio."""
        meta = RequestMetadata(
            portfolio_size=25,
            portfolio_value=50_000.00,
            user_tier="standard",
        )

        tags = meta.get_auto_tags()

        assert Tag.LARGE_PORTFOLIO.value in tags
        assert Tag.USER_STANDARD.value in tags
        assert Tag.HIGH_VALUE_PORTFOLIO.value not in tags

    def test_from_portfolio_state(self) -> None:
        """Test creating metadata from portfolio state."""
        state = {
            "portfolio": {
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "current_price": 150.0},
                    {"symbol": "GOOGL", "quantity": 50, "current_price": 100.0},
                ]
            },
            "symbols": ["AAPL", "GOOGL"],
            "user_id": "user-123",
            "user_tier": "premium",
            "workflow_id": "wf-789",
        }

        meta = RequestMetadata.from_portfolio_state(state)

        assert meta.portfolio_size == 2
        assert meta.portfolio_value == 20_000.0  # 100*150 + 50*100
        assert meta.symbols == ["AAPL", "GOOGL"]
        assert meta.user_id == "user-123"
        assert meta.user_tier == "premium"
        assert meta.workflow_id == "wf-789"


# ============================================================================
# AgentMetadata Tests
# ============================================================================


class TestAgentMetadata:
    """Tests for AgentMetadata class."""

    def test_default_values(self) -> None:
        """Test AgentMetadata with default values."""
        meta = AgentMetadata()

        assert meta.agent_name == ""
        assert meta.tools_used == []
        assert meta.tool_calls_count == 0
        assert meta.reasoning_steps == 0
        assert meta.input_tokens == 0
        assert meta.output_tokens == 0
        assert meta.duration_ms == 0.0
        assert meta.errors == []
        assert meta.retries == 0

    def test_all_fields_set(self) -> None:
        """Test AgentMetadata with all fields."""
        meta = AgentMetadata(
            agent_name="research",
            tools_used=["market_data", "news_search"],
            tool_calls_count=5,
            reasoning_steps=3,
            input_tokens=1000,
            output_tokens=500,
            duration_ms=2500.0,
            errors=["API timeout"],
            retries=2,
            custom={"model": "claude-sonnet"},
        )

        assert meta.agent_name == "research"
        assert meta.tools_used == ["market_data", "news_search"]
        assert meta.tool_calls_count == 5
        assert meta.reasoning_steps == 3
        assert meta.input_tokens == 1000
        assert meta.output_tokens == 500
        assert meta.duration_ms == 2500.0
        assert meta.errors == ["API timeout"]
        assert meta.retries == 2

    def test_to_dict(self) -> None:
        """Test AgentMetadata.to_dict method."""
        meta = AgentMetadata(
            agent_name="analysis",
            tools_used=["risk_calculator"],
            tool_calls_count=3,
            input_tokens=500,
            output_tokens=300,
        )

        result = meta.to_dict()

        assert result["agent"] == "analysis"
        assert result["tools_used"] == ["risk_calculator"]
        assert result["tools_used_count"] == 1
        assert result["tool_calls_count"] == 3
        assert result["total_tokens"] == 800
        assert result["has_errors"] is False

    def test_add_tool_call(self) -> None:
        """Test adding tool calls."""
        meta = AgentMetadata(agent_name="test")

        meta.add_tool_call("market_data")
        meta.add_tool_call("news_search")
        meta.add_tool_call("market_data")  # Duplicate

        assert meta.tools_used == ["market_data", "news_search"]
        assert meta.tool_calls_count == 3

    def test_add_reasoning_step(self) -> None:
        """Test adding reasoning steps."""
        meta = AgentMetadata(agent_name="test")

        meta.add_reasoning_step()
        meta.add_reasoning_step()
        meta.add_reasoning_step()

        assert meta.reasoning_steps == 3

    def test_add_error(self) -> None:
        """Test adding errors."""
        meta = AgentMetadata(agent_name="test")

        meta.add_error("Connection timeout")
        meta.add_error("Rate limited")

        assert meta.errors == ["Connection timeout", "Rate limited"]

    def test_get_auto_tags_error_recovered(self) -> None:
        """Test auto tags for error recovery."""
        meta = AgentMetadata(
            agent_name="test",
            errors=["Initial error"],
            retries=2,
        )

        tags = meta.get_auto_tags()

        assert Tag.ERROR_RECOVERED.value in tags
        assert Tag.RETRY_SUCCEEDED.value in tags

    def test_get_auto_tags_high_token_usage(self) -> None:
        """Test auto tags for high token usage."""
        meta = AgentMetadata(
            agent_name="test",
            input_tokens=8000,
            output_tokens=5000,
        )

        tags = meta.get_auto_tags()

        assert Tag.HIGH_TOKEN_USAGE.value in tags

    def test_get_auto_tags_slow_request(self) -> None:
        """Test auto tags for slow request."""
        meta = AgentMetadata(
            agent_name="test",
            duration_ms=45_000,  # 45 seconds
        )

        tags = meta.get_auto_tags()

        assert Tag.SLOW_REQUEST.value in tags


# ============================================================================
# ToolMetadata Tests
# ============================================================================


class TestToolMetadata:
    """Tests for ToolMetadata class."""

    def test_default_values(self) -> None:
        """Test ToolMetadata with default values."""
        meta = ToolMetadata()

        assert meta.tool_name == ""
        assert meta.api_source == "real"
        assert meta.symbols_requested == 0
        assert meta.cache_hit is False
        assert meta.cache_key is None
        assert meta.api_latency_ms == 0.0
        assert meta.response_size == 0
        assert meta.error is None
        assert meta.retry_count == 0

    def test_all_fields_set(self) -> None:
        """Test ToolMetadata with all fields."""
        meta = ToolMetadata(
            tool_name="market_data",
            api_source="real",
            symbols_requested=5,
            cache_hit=False,
            cache_key="md:AAPL:1d",
            api_latency_ms=150.5,
            response_size=2048,
            error=None,
            retry_count=0,
            custom={"provider": "yahoo"},
        )

        assert meta.tool_name == "market_data"
        assert meta.api_source == "real"
        assert meta.symbols_requested == 5
        assert meta.cache_hit is False
        assert meta.cache_key == "md:AAPL:1d"
        assert meta.api_latency_ms == 150.5
        assert meta.response_size == 2048
        assert meta.custom == {"provider": "yahoo"}

    def test_to_dict(self) -> None:
        """Test ToolMetadata.to_dict method."""
        meta = ToolMetadata(
            tool_name="news_search",
            api_source="mock",
            symbols_requested=3,
            cache_hit=True,
            api_latency_ms=50.0,
        )

        result = meta.to_dict()

        assert result["tool"] == "news_search"
        assert result["api_source"] == "mock"
        assert result["symbols_requested"] == 3
        assert result["cache_hit"] is True
        assert result["has_error"] is False

    def test_get_auto_tags_mock_data(self) -> None:
        """Test auto tags for mock data."""
        meta = ToolMetadata(
            tool_name="market_data",
            api_source="mock",
        )

        tags = meta.get_auto_tags()

        assert Tag.MOCK_DATA_USED.value in tags

    def test_get_auto_tags_cache_hit(self) -> None:
        """Test auto tags for cache hit."""
        meta = ToolMetadata(
            tool_name="market_data",
            cache_hit=True,
        )

        tags = meta.get_auto_tags()

        assert Tag.CACHE_HIT.value in tags

    def test_get_auto_tags_api_fallback(self) -> None:
        """Test auto tags for API fallback."""
        meta = ToolMetadata(
            tool_name="market_data",
            api_source="fallback",
        )

        tags = meta.get_auto_tags()

        assert Tag.API_FALLBACK.value in tags


# ============================================================================
# TagManager Tests
# ============================================================================


class TestTagManager:
    """Tests for TagManager class."""

    def test_empty_tags(self) -> None:
        """Test TagManager starts empty."""
        manager = TagManager()

        assert manager.tags == []

    def test_add_string_tag(self) -> None:
        """Test adding string tags."""
        manager = TagManager()

        manager.add("custom-tag")
        manager.add("another-tag")

        assert "custom-tag" in manager.tags
        assert "another-tag" in manager.tags

    def test_add_enum_tag(self) -> None:
        """Test adding Tag enum values."""
        manager = TagManager()

        manager.add(Tag.PORTFOLIO_ANALYSIS)
        manager.add(Tag.HIGH_VALUE_PORTFOLIO)

        assert Tag.PORTFOLIO_ANALYSIS.value in manager.tags
        assert Tag.HIGH_VALUE_PORTFOLIO.value in manager.tags

    def test_add_many(self) -> None:
        """Test adding multiple tags at once."""
        manager = TagManager()

        manager.add_many([
            Tag.PORTFOLIO_ANALYSIS,
            "custom-tag",
            Tag.CACHE_HIT,
        ])

        assert len(manager.tags) == 3

    def test_deduplication(self) -> None:
        """Test that duplicate tags are deduplicated."""
        manager = TagManager()

        manager.add(Tag.CACHE_HIT)
        manager.add(Tag.CACHE_HIT)
        manager.add("cache-hit")

        assert manager.tags.count(Tag.CACHE_HIT.value) == 1

    def test_add_from_metadata(self) -> None:
        """Test adding tags from metadata objects."""
        manager = TagManager()

        request = RequestMetadata(portfolio_value=200_000, user_tier="premium")
        tool = ToolMetadata(tool_name="test", cache_hit=True)

        manager.add_from_metadata(request, tool)

        assert Tag.HIGH_VALUE_PORTFOLIO.value in manager.tags
        assert Tag.USER_PREMIUM.value in manager.tags
        assert Tag.CACHE_HIT.value in manager.tags

    def test_remove_tag(self) -> None:
        """Test removing tags."""
        manager = TagManager()
        manager.add(Tag.CACHE_HIT)
        manager.add(Tag.MOCK_DATA_USED)

        manager.remove(Tag.CACHE_HIT)

        assert Tag.CACHE_HIT.value not in manager.tags
        assert Tag.MOCK_DATA_USED.value in manager.tags

    def test_has_tag(self) -> None:
        """Test checking tag existence."""
        manager = TagManager()
        manager.add(Tag.CACHE_HIT)

        assert manager.has(Tag.CACHE_HIT) is True
        assert manager.has("cache-hit") is True
        assert manager.has(Tag.MOCK_DATA_USED) is False

    def test_clear(self) -> None:
        """Test clearing all tags."""
        manager = TagManager()
        manager.add_many([Tag.CACHE_HIT, Tag.MOCK_DATA_USED])

        manager.clear()

        assert manager.tags == []

    def test_tags_sorted(self) -> None:
        """Test that tags are returned sorted."""
        manager = TagManager()
        manager.add("z-tag")
        manager.add("a-tag")
        manager.add("m-tag")

        assert manager.tags == ["a-tag", "m-tag", "z-tag"]


# ============================================================================
# MetadataBuilder Tests
# ============================================================================


class TestMetadataBuilder:
    """Tests for MetadataBuilder class."""

    def test_empty_build(self) -> None:
        """Test building with no metadata."""
        result = MetadataBuilder().build()

        assert result == {"tags": []}

    def test_with_request(self) -> None:
        """Test adding request metadata."""
        request = RequestMetadata(
            portfolio_size=10,
            portfolio_value=50_000,
        )

        result = MetadataBuilder().with_request(request).build()

        assert "request" in result
        assert result["request"]["portfolio_size"] == 10
        assert Tag.PORTFOLIO_ANALYSIS.value in result["tags"]

    def test_with_agent(self) -> None:
        """Test adding agent metadata."""
        agent = AgentMetadata(
            agent_name="research",
            tool_calls_count=5,
        )

        result = MetadataBuilder().with_agent(agent).build()

        assert "agent" in result
        assert result["agent"]["agent"] == "research"
        assert result["agent"]["tool_calls_count"] == 5

    def test_with_tool(self) -> None:
        """Test adding tool metadata."""
        tool = ToolMetadata(
            tool_name="market_data",
            cache_hit=True,
        )

        result = MetadataBuilder().with_tool(tool).build()

        assert "tool" in result
        assert result["tool"]["tool"] == "market_data"
        assert Tag.CACHE_HIT.value in result["tags"]

    def test_with_tags(self) -> None:
        """Test adding custom tags."""
        result = (
            MetadataBuilder()
            .with_tags([Tag.HIGH_VALUE_PORTFOLIO, "custom-tag"])
            .build()
        )

        assert Tag.HIGH_VALUE_PORTFOLIO.value in result["tags"]
        assert "custom-tag" in result["tags"]

    def test_with_custom(self) -> None:
        """Test adding custom metadata."""
        result = (
            MetadataBuilder()
            .with_custom("environment", "production")
            .with_custom("version", "1.0.0")
            .build()
        )

        assert result["environment"] == "production"
        assert result["version"] == "1.0.0"

    def test_chaining(self) -> None:
        """Test fluent chaining of methods."""
        request = RequestMetadata(portfolio_size=5)
        agent = AgentMetadata(agent_name="test")
        tool = ToolMetadata(tool_name="test_tool")

        result = (
            MetadataBuilder()
            .with_request(request)
            .with_agent(agent)
            .with_tool(tool)
            .with_tags(["extra-tag"])
            .with_custom("key", "value")
            .build()
        )

        assert "request" in result
        assert "agent" in result
        assert "tool" in result
        assert "extra-tag" in result["tags"]
        assert result["key"] == "value"

    def test_build_tags(self) -> None:
        """Test building only tags."""
        request = RequestMetadata(portfolio_value=200_000)
        tool = ToolMetadata(cache_hit=True)

        tags = (
            MetadataBuilder()
            .with_request(request)
            .with_tool(tool)
            .with_tags(["custom"])
            .build_tags()
        )

        assert Tag.HIGH_VALUE_PORTFOLIO.value in tags
        assert Tag.CACHE_HIT.value in tags
        assert "custom" in tags


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_request_metadata(self) -> None:
        """Test create_request_metadata function."""
        meta = create_request_metadata(
            portfolio_size=10,
            portfolio_value=100_000,
            symbols=["AAPL", "GOOGL"],
            user_tier="premium",
            extra_field="value",
        )

        assert meta.portfolio_size == 10
        assert meta.portfolio_value == 100_000
        assert meta.symbols == ["AAPL", "GOOGL"]
        assert meta.user_tier == "premium"
        assert meta.custom == {"extra_field": "value"}

    def test_create_agent_metadata(self) -> None:
        """Test create_agent_metadata function."""
        meta = create_agent_metadata(
            "research",
            tools_used=["market_data"],
            tool_calls_count=3,
            extra="value",
        )

        assert meta.agent_name == "research"
        assert meta.tools_used == ["market_data"]
        assert meta.tool_calls_count == 3
        assert meta.custom == {"extra": "value"}

    def test_create_tool_metadata(self) -> None:
        """Test create_tool_metadata function."""
        meta = create_tool_metadata(
            "news_search",
            api_source="mock",
            cache_hit=True,
            provider="test",
        )

        assert meta.tool_name == "news_search"
        assert meta.api_source == "mock"
        assert meta.cache_hit is True
        assert meta.custom == {"provider": "test"}

    def test_collect_tags(self) -> None:
        """Test collect_tags function."""
        request = RequestMetadata(portfolio_value=200_000)
        agent = AgentMetadata(retries=1)
        tool = ToolMetadata(cache_hit=True)

        tags = collect_tags(request, agent, tool)

        assert Tag.HIGH_VALUE_PORTFOLIO.value in tags
        assert Tag.RETRY_SUCCEEDED.value in tags
        assert Tag.CACHE_HIT.value in tags
        # Should be sorted and deduplicated
        assert tags == sorted(set(tags))
