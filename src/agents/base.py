"""Base agent class for all specialized agents.

This module provides the abstract base class that all agents inherit from,
integrated with Anthropic Claude and LangGraph for state management.

Prompt Caching:
    Agents support Claude's prompt caching feature which can reduce costs
    by up to 90% and latency by up to 85% for repeated system prompts.
    Enable via enable_cache=True in __init__ or set USE_PROMPT_CACHING=true.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog
from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.caching.metrics import CacheMetrics
from src.caching.prompt_cache import (
    build_cached_system_prompt,
    build_cached_tool_definitions,
)
from src.config import settings

logger = structlog.get_logger(__name__)


class AgentState(BaseModel):
    """Base state for agent workflows.

    This state is passed between agents in the workflow and accumulates
    results from each step.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]] = []
    context: dict[str, Any] = {}
    errors: list[str] = []


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Provides common functionality for LLM interaction, state management,
    and error handling. All specialized agents must inherit from this class.

    Attributes:
        llm: Anthropic client for Claude API calls
        model: Model identifier to use for completions
        max_tokens: Maximum tokens for LLM responses
        enable_cache: Whether to use prompt caching for reduced costs
        last_cache_metrics: Metrics from the last LLM call (if caching enabled)
    """

    def __init__(
        self,
        llm: Anthropic | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        enable_cache: bool | None = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            llm: Optional Anthropic client. If not provided, creates a new one.
            model: Model to use for completions.
            max_tokens: Maximum tokens for responses.
            enable_cache: Whether to enable prompt caching. If None, uses
                settings.USE_PROMPT_CACHING.
        """
        self.llm = llm or Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.enable_cache = enable_cache if enable_cache is not None else settings.USE_PROMPT_CACHING
        self.last_cache_metrics: CacheMetrics | None = None
        self._logger = logger.bind(agent=self.name)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's unique name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this agent does."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return the list of tools available to this agent.

        Override in subclasses to provide agent-specific tools.
        """
        return []

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        additional_context: str | None = None,
    ) -> dict[str, Any]:
        """Make a retry-enabled call to the LLM.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions.
            additional_context: Optional dynamic context to add to system prompt.

        Returns:
            The LLM response as a dict.

        Raises:
            TimeoutError: If the request times out after retries.
            ConnectionError: If connection fails after retries.
        """
        self._logger.debug(
            "calling_llm",
            message_count=len(messages),
            has_tools=bool(tools),
            caching_enabled=self.enable_cache,
        )

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        # Use cached system prompt structure if caching is enabled
        if self.enable_cache:
            # Build cached system prompt with cache_control breakpoints
            params["system"] = build_cached_system_prompt(
                agent_prompt=self.system_prompt,
                tool_definitions=tools,
                additional_context=additional_context,
            )
            # Add cache_control to tools as well
            if tools:
                params["tools"] = build_cached_tool_definitions(tools)
        else:
            # Standard string system prompt
            params["system"] = self.system_prompt
            if tools:
                params["tools"] = tools

        response = self.llm.messages.create(**params)

        # Extract cache metrics from response
        cache_metrics = CacheMetrics.from_response(
            response,
            model=self.model,
            agent_name=self.name,
        )
        self.last_cache_metrics = cache_metrics

        # Build usage dict with cache metrics
        usage_dict = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Add cache metrics if available
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage_dict["cache_creation_input_tokens"] = response.usage.cache_creation_input_tokens
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage_dict["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

        self._logger.debug(
            "llm_response_received",
            stop_reason=response.stop_reason,
            usage=usage_dict,
            cache_hit=cache_metrics.is_cache_hit if self.enable_cache else None,
            cache_hit_rate=f"{cache_metrics.cache_hit_rate:.1%}" if self.enable_cache else None,
        )

        return {
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
            "stop_reason": response.stop_reason,
            "usage": usage_dict,
        }

    @abstractmethod
    async def invoke(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state.

        This is the main entry point for agent execution. Subclasses must
        implement this method to define their specific behavior.

        Args:
            state: The current workflow state.

        Returns:
            The updated state after processing.
        """
        ...

    async def __call__(self, state: AgentState) -> AgentState:
        """Allow agent to be called as a function.

        This enables using the agent directly in LangGraph workflows.

        Args:
            state: The current workflow state.

        Returns:
            The updated state after processing.
        """
        self._logger.info("agent_invoked", state_keys=list(state.context.keys()))
        try:
            result = await self.invoke(state)
            self._logger.info("agent_completed", success=True)
            return result
        except Exception as e:
            self._logger.error("agent_failed", error=str(e))
            state.errors.append(f"{self.name}: {e!s}")
            return state
