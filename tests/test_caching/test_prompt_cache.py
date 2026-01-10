"""Tests for prompt caching utilities."""

from src.caching.prompt_cache import (
    CacheableContent,
    CacheControl,
    CacheControlType,
    CachedPromptBuilder,
    build_cached_system_prompt,
    build_cached_tool_definitions,
    estimate_cache_tokens,
    is_cacheable,
)


class TestCacheControl:
    """Tests for CacheControl class."""

    def test_default_type_is_ephemeral(self):
        """CacheControl should default to ephemeral type."""
        control = CacheControl()
        assert control.type == CacheControlType.EPHEMERAL

    def test_to_dict(self):
        """to_dict should return correct format for API."""
        control = CacheControl()
        result = control.to_dict()
        assert result == {"type": "ephemeral"}

    def test_explicit_ephemeral_type(self):
        """Explicit ephemeral type should work."""
        control = CacheControl(type=CacheControlType.EPHEMERAL)
        assert control.to_dict() == {"type": "ephemeral"}


class TestCacheableContent:
    """Tests for CacheableContent class."""

    def test_uncached_content(self):
        """Uncached content should not have cache_control."""
        content = CacheableContent.uncached("test text")
        result = content.to_dict()
        assert result == {"type": "text", "text": "test text"}
        assert "cache_control" not in result

    def test_cached_content(self):
        """Cached content should include cache_control."""
        content = CacheableContent.cached("test text")
        result = content.to_dict()
        assert result == {
            "type": "text",
            "text": "test text",
            "cache_control": {"type": "ephemeral"},
        }

    def test_manual_cache_control(self):
        """Manual cache control should be applied."""
        control = CacheControl()
        content = CacheableContent(text="test", cache_control=control)
        result = content.to_dict()
        assert "cache_control" in result
        assert result["cache_control"] == {"type": "ephemeral"}

    def test_to_dict_preserves_text(self):
        """to_dict should preserve the original text."""
        text = "This is a test prompt with special chars: <>&"
        content = CacheableContent.cached(text)
        assert content.to_dict()["text"] == text


class TestCachedPromptBuilder:
    """Tests for CachedPromptBuilder class."""

    def test_empty_builder(self):
        """Empty builder should return empty list."""
        builder = CachedPromptBuilder()
        assert builder.build_system() == []

    def test_add_system_prompt_cached(self):
        """Adding cached system prompt should work."""
        builder = CachedPromptBuilder()
        builder.add_system_prompt("You are a helpful assistant.", cached=True)
        result = builder.build_system()

        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are a helpful assistant."
        assert "cache_control" in result[0]

    def test_add_system_prompt_uncached(self):
        """Adding uncached system prompt should work."""
        builder = CachedPromptBuilder()
        builder.add_system_prompt("Dynamic content", cached=False)
        result = builder.build_system()

        assert len(result) == 1
        assert "cache_control" not in result[0]

    def test_method_chaining(self):
        """Builder methods should support chaining."""
        builder = CachedPromptBuilder()
        result = (
            builder.add_system_prompt("Base prompt")
            .add_context("Extra context")
            .build_system()
        )
        assert len(result) == 2

    def test_add_context_with_label(self):
        """Adding context with label should format correctly."""
        builder = CachedPromptBuilder()
        builder.add_context("Some context info", label="Important")
        result = builder.build_system()

        assert len(result) == 1
        assert "## Important" in result[0]["text"]
        assert "Some context info" in result[0]["text"]

    def test_add_tool_definitions(self):
        """Adding tool definitions should format correctly."""
        tools = [
            {"name": "get_data", "description": "Gets data from API"},
            {"name": "search", "description": "Searches for info"},
        ]
        builder = CachedPromptBuilder()
        builder.add_tool_definitions(tools, cached=True)
        result = builder.build_system()

        assert len(result) == 1
        assert "get_data" in result[0]["text"]
        assert "search" in result[0]["text"]
        assert "cache_control" in result[0]

    def test_empty_tools_not_added(self):
        """Empty tool list should not add a block."""
        builder = CachedPromptBuilder()
        builder.add_tool_definitions([], cached=True)
        result = builder.build_system()
        assert len(result) == 0

    def test_full_builder_flow(self):
        """Full builder flow with all components."""
        tools = [{"name": "tool1", "description": "A tool"}]
        builder = CachedPromptBuilder()
        result = (
            builder.add_system_prompt("Base prompt", cached=True)
            .add_tool_definitions(tools, cached=True)
            .add_context("Memory info", label="Memory", cached=True)
            .add_context("Dynamic query", label="Query", cached=False)
            .build_system()
        )

        assert len(result) == 4
        # Order: system_blocks first (prompt, memory, query), then tool_blocks
        # result[0] = Base prompt (cached)
        # result[1] = Memory info (cached)
        # result[2] = Dynamic query (uncached)
        # result[3] = tools (cached)
        assert "cache_control" in result[0]  # Base prompt
        assert "cache_control" in result[1]  # Memory
        assert "cache_control" not in result[2]  # Dynamic query
        assert "cache_control" in result[3]  # Tools


class TestBuildCachedSystemPrompt:
    """Tests for build_cached_system_prompt function."""

    def test_basic_prompt_only(self):
        """Basic prompt should be cached."""
        result = build_cached_system_prompt(agent_prompt="You are helpful.")

        assert len(result) == 1
        assert result[0]["text"] == "You are helpful."
        assert "cache_control" in result[0]

    def test_with_tools(self):
        """Prompt with tools should include both."""
        tools = [{"name": "search", "description": "Search tool"}]
        result = build_cached_system_prompt(
            agent_prompt="You are helpful.",
            tool_definitions=tools,
        )

        assert len(result) == 2
        assert "cache_control" in result[0]
        assert "cache_control" in result[1]

    def test_with_memory_context(self):
        """Memory context should be cached."""
        result = build_cached_system_prompt(
            agent_prompt="You are helpful.",
            memory_context="Previous interaction data",
        )

        assert len(result) == 2
        assert "Memory Context" in result[1]["text"]
        assert "cache_control" in result[1]

    def test_with_additional_context(self):
        """Additional context should not be cached."""
        result = build_cached_system_prompt(
            agent_prompt="You are helpful.",
            additional_context="User query: test",
        )

        assert len(result) == 2
        assert "cache_control" in result[0]
        assert "cache_control" not in result[1]

    def test_full_combination(self):
        """Full combination of all options."""
        tools = [{"name": "t1", "description": "d1"}]
        result = build_cached_system_prompt(
            agent_prompt="Base",
            tool_definitions=tools,
            memory_context="Memory",
            additional_context="Dynamic",
        )

        assert len(result) == 4
        # Order in build_cached_system_prompt:
        # 1. agent_prompt (cached)
        # 2. tool_definitions (cached) - added to tool_blocks
        # 3. memory_context (cached)
        # 4. additional_context (uncached)
        # But system_blocks come first, then tool_blocks
        # result[0] = Base (system prompt, cached)
        # result[1] = Memory Context (cached)
        # result[2] = Current Context (uncached)
        # result[3] = tools (cached)
        assert "cache_control" in result[0]  # Base
        assert "cache_control" in result[1]  # Memory
        assert "cache_control" not in result[2]  # Dynamic
        assert "cache_control" in result[3]  # Tools


class TestBuildCachedToolDefinitions:
    """Tests for build_cached_tool_definitions function."""

    def test_empty_tools(self):
        """Empty tools list should return empty list."""
        result = build_cached_tool_definitions([])
        assert result == []

    def test_single_tool(self):
        """Single tool should get cache_control."""
        tools = [{"name": "test", "description": "Test tool"}]
        result = build_cached_tool_definitions(tools)

        assert len(result) == 1
        assert result[0]["name"] == "test"
        assert "cache_control" in result[0]
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_multiple_tools_only_last_cached(self):
        """Only last tool in list should have cache_control."""
        tools = [
            {"name": "tool1", "description": "First"},
            {"name": "tool2", "description": "Second"},
            {"name": "tool3", "description": "Third"},
        ]
        result = build_cached_tool_definitions(tools)

        assert len(result) == 3
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]
        assert "cache_control" in result[2]

    def test_does_not_mutate_original(self):
        """Should not mutate original tools list."""
        original_tool = {"name": "test", "description": "Test"}
        tools = [original_tool]
        build_cached_tool_definitions(tools)

        assert "cache_control" not in original_tool


class TestEstimateCacheTokens:
    """Tests for estimate_cache_tokens function."""

    def test_empty_string(self):
        """Empty string should return 0."""
        assert estimate_cache_tokens("") == 0

    def test_short_text(self):
        """Short text should return proportional estimate."""
        # ~4 chars per token
        result = estimate_cache_tokens("test")
        assert result == 1

    def test_longer_text(self):
        """Longer text should scale proportionally."""
        text = "a" * 4000  # ~1000 tokens
        result = estimate_cache_tokens(text)
        assert 900 <= result <= 1100  # Allow some variance


class TestIsCacheable:
    """Tests for is_cacheable function."""

    def test_short_text_not_cacheable(self):
        """Short text should not be cacheable."""
        result = is_cacheable("Short text")
        assert result is False

    def test_long_text_cacheable(self):
        """Text with 1024+ tokens should be cacheable."""
        # Create text with ~1100 tokens (4400 chars)
        text = "a" * 4400
        result = is_cacheable(text)
        assert result is True

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        text = "a" * 400  # ~100 tokens
        assert is_cacheable(text, min_tokens=50) is True
        assert is_cacheable(text, min_tokens=200) is False

    def test_boundary_case(self):
        """Test near the boundary."""
        # 4096 chars = ~1024 tokens
        text = "a" * 4096
        result = is_cacheable(text)
        assert result is True
