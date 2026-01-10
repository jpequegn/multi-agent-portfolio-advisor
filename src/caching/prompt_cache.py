"""Prompt caching utilities for Claude API.

This module provides utilities for structuring prompts to take advantage
of Claude's prompt caching feature, which can reduce costs by up to 90%
and latency by up to 85% for repeated content.

Cache Control:
    - "ephemeral": 5-minute TTL (default), refreshes on hit
    - Content must be at least 1,024 tokens to be cached

Usage:
    # Build cached system prompt
    system = build_cached_system_prompt(
        agent_prompt="You are a research agent...",
        tool_definitions=tools,
    )

    # Use with Anthropic API
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=system,
        messages=messages,
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CacheControlType(str, Enum):
    """Supported cache control types."""

    EPHEMERAL = "ephemeral"


@dataclass
class CacheControl:
    """Cache control configuration for a content block.

    Attributes:
        type: The cache control type (currently only "ephemeral").
    """

    type: CacheControlType = CacheControlType.EPHEMERAL

    def to_dict(self) -> dict[str, str]:
        """Convert to API-compatible dictionary.

        Returns:
            Dictionary with cache_control configuration.
        """
        return {"type": self.type.value}


@dataclass
class CacheableContent:
    """A content block that can be cached.

    Attributes:
        text: The text content to cache.
        cache_control: Optional cache control settings.
    """

    text: str
    cache_control: CacheControl | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary.

        Returns:
            Dictionary representation for API calls.
        """
        result: dict[str, Any] = {
            "type": "text",
            "text": self.text,
        }
        if self.cache_control:
            result["cache_control"] = self.cache_control.to_dict()
        return result

    @classmethod
    def cached(cls, text: str) -> "CacheableContent":
        """Create a cached content block.

        Args:
            text: The text content to cache.

        Returns:
            CacheableContent with ephemeral caching enabled.
        """
        return cls(text=text, cache_control=CacheControl())

    @classmethod
    def uncached(cls, text: str) -> "CacheableContent":
        """Create an uncached content block.

        Args:
            text: The text content (not cached).

        Returns:
            CacheableContent without caching.
        """
        return cls(text=text, cache_control=None)


@dataclass
class CachedPromptBuilder:
    """Builder for constructing prompts with optimal cache placement.

    This builder helps structure prompts to maximize cache efficiency
    by placing static content at the beginning with cache breakpoints.

    Example:
        builder = CachedPromptBuilder()
        builder.add_system_prompt(AGENT_PROMPT, cached=True)
        builder.add_tool_definitions(tools, cached=True)
        builder.add_context(dynamic_context, cached=False)

        system = builder.build_system()
    """

    _system_blocks: list[CacheableContent] = field(default_factory=list)
    _tool_blocks: list[CacheableContent] = field(default_factory=list)

    def add_system_prompt(self, prompt: str, *, cached: bool = True) -> "CachedPromptBuilder":
        """Add a system prompt block.

        Args:
            prompt: The system prompt text.
            cached: Whether to enable caching for this block.

        Returns:
            Self for method chaining.
        """
        if cached:
            self._system_blocks.append(CacheableContent.cached(prompt))
        else:
            self._system_blocks.append(CacheableContent.uncached(prompt))

        logger.debug(
            "system_prompt_added",
            length=len(prompt),
            cached=cached,
        )
        return self

    def add_tool_definitions(
        self,
        tools: list[dict[str, Any]],
        *,
        cached: bool = True,
    ) -> "CachedPromptBuilder":
        """Add tool definitions as a cacheable block.

        Args:
            tools: List of tool definitions in Anthropic format.
            cached: Whether to enable caching for this block.

        Returns:
            Self for method chaining.
        """
        if not tools:
            return self

        # Format tools as text block for system prompt
        # Note: Tools are typically passed separately to the API,
        # but we can include descriptions in the system prompt for caching
        tool_text = self._format_tools_for_prompt(tools)
        if cached:
            self._tool_blocks.append(CacheableContent.cached(tool_text))
        else:
            self._tool_blocks.append(CacheableContent.uncached(tool_text))

        logger.debug(
            "tool_definitions_added",
            tool_count=len(tools),
            cached=cached,
        )
        return self

    def add_context(
        self,
        context: str,
        *,
        label: str | None = None,
        cached: bool = False,
    ) -> "CachedPromptBuilder":
        """Add additional context to the system prompt.

        Args:
            context: The context text to add.
            label: Optional label for the context section.
            cached: Whether to enable caching for this block.

        Returns:
            Self for method chaining.
        """
        text = f"\n\n## {label}\n{context}" if label else f"\n\n{context}"

        if cached:
            self._system_blocks.append(CacheableContent.cached(text))
        else:
            self._system_blocks.append(CacheableContent.uncached(text))

        logger.debug(
            "context_added",
            label=label,
            length=len(context),
            cached=cached,
        )
        return self

    def build_system(self) -> list[dict[str, Any]]:
        """Build the system parameter for the API call.

        Returns:
            List of content blocks for the system parameter.

        Note:
            Claude API accepts system as either a string or list of blocks.
            Using list format enables cache_control on individual blocks.
        """
        blocks = []
        for content in self._system_blocks:
            blocks.append(content.to_dict())
        for content in self._tool_blocks:
            blocks.append(content.to_dict())

        logger.debug(
            "system_prompt_built",
            block_count=len(blocks),
            cached_count=sum(1 for b in blocks if "cache_control" in b),
        )
        return blocks

    def _format_tools_for_prompt(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for inclusion in system prompt.

        Args:
            tools: List of tool definitions.

        Returns:
            Formatted string describing tools.
        """
        lines = ["\n## Available Tools\n"]
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            lines.append(f"- **{name}**: {description}")

            # Include input schema summary if available
            input_schema = tool.get("input_schema", {})
            if input_schema.get("properties"):
                props = input_schema["properties"]
                param_list = ", ".join(props.keys())
                lines.append(f"  Parameters: {param_list}")

        return "\n".join(lines)


def build_cached_system_prompt(
    agent_prompt: str,
    *,
    tool_definitions: list[dict[str, Any]] | None = None,
    memory_context: str | None = None,
    additional_context: str | None = None,
) -> list[dict[str, Any]]:
    """Build a cached system prompt with optimal cache placement.

    This is a convenience function for common use cases. For more control,
    use CachedPromptBuilder directly.

    Args:
        agent_prompt: The main agent system prompt (cached).
        tool_definitions: Optional tool definitions (cached).
        memory_context: Optional memory/history context (cached if stable).
        additional_context: Optional dynamic context (not cached).

    Returns:
        List of content blocks for the system parameter.

    Example:
        system = build_cached_system_prompt(
            agent_prompt=RESEARCH_AGENT_PROMPT,
            tool_definitions=tools,
            additional_context=f"User request: {user_query}"
        )
    """
    builder = CachedPromptBuilder()

    # Add main prompt (always cached - this is the static agent definition)
    builder.add_system_prompt(agent_prompt, cached=True)

    # Add tool definitions if provided (cached - these are static)
    if tool_definitions:
        builder.add_tool_definitions(tool_definitions, cached=True)

    # Add memory context if provided (cached - relatively stable)
    if memory_context:
        builder.add_context(memory_context, label="Memory Context", cached=True)

    # Add dynamic context if provided (not cached - changes per request)
    if additional_context:
        builder.add_context(additional_context, label="Current Context", cached=False)

    return builder.build_system()


def build_cached_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add cache control to tool definitions.

    Note: Tool caching is applied at the tool list level, not individual tools.
    The last tool in the list should have cache_control for optimal caching.

    Args:
        tools: List of tool definitions in Anthropic format.

    Returns:
        Tools with cache_control added to the last tool.
    """
    if not tools:
        return tools

    # Clone tools to avoid mutating originals
    cached_tools = []
    for i, tool in enumerate(tools):
        tool_copy = dict(tool)
        # Add cache_control to the last tool
        if i == len(tools) - 1:
            tool_copy["cache_control"] = {"type": "ephemeral"}
        cached_tools.append(tool_copy)

    logger.debug(
        "tools_cached",
        tool_count=len(cached_tools),
    )
    return cached_tools


def estimate_cache_tokens(text: str) -> int:
    """Estimate the number of tokens in text for cache planning.

    This is a rough estimate based on average token length.
    Actual token count may vary by ~20%.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.

    Note:
        Claude requires at least 1,024 tokens per cache checkpoint.
    """
    # Rough estimate: ~4 characters per token on average
    estimated = len(text) // 4
    logger.debug("tokens_estimated", text_length=len(text), estimated_tokens=estimated)
    return estimated


def is_cacheable(text: str, min_tokens: int = 1024) -> bool:
    """Check if text meets minimum token requirement for caching.

    Args:
        text: Text to check.
        min_tokens: Minimum tokens required (default 1024).

    Returns:
        True if text is long enough to be cached.
    """
    return estimate_cache_tokens(text) >= min_tokens
