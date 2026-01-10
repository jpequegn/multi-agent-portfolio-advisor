"""Tests for the base agent class."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.base import AgentState, BaseAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    @property
    def name(self) -> str:
        return "test_agent"

    @property
    def description(self) -> str:
        return "A test agent for unit testing"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."

    async def invoke(self, state: AgentState) -> AgentState:
        """Simple implementation that adds a message."""
        state.messages.append({"role": "assistant", "content": "Test response"})
        state.context["processed"] = True
        return state


class TestAgentState:
    """Tests for AgentState model."""

    def test_default_values(self) -> None:
        """Test that AgentState has correct defaults."""
        state = AgentState()
        assert state.messages == []
        assert state.context == {}
        assert state.errors == []

    def test_with_values(self) -> None:
        """Test AgentState with custom values."""
        state = AgentState(
            messages=[{"role": "user", "content": "Hello"}],
            context={"key": "value"},
            errors=["error1"],
        )
        assert len(state.messages) == 1
        assert state.context["key"] == "value"
        assert len(state.errors) == 1

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed in AgentState."""
        state = AgentState(custom_field="custom_value")  # type: ignore[call-arg]
        assert state.custom_field == "custom_value"  # type: ignore[attr-defined]


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_cannot_instantiate_base_agent(self) -> None:
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgent()  # type: ignore[abstract]

    def test_concrete_agent_instantiation(self) -> None:
        """Test that concrete implementations can be instantiated."""
        agent = ConcreteAgent()
        assert agent.name == "test_agent"
        assert agent.description == "A test agent for unit testing"
        assert agent.system_prompt == "You are a test agent."
        assert agent.tools == []

    def test_custom_llm_client(self) -> None:
        """Test that a custom LLM client can be provided."""
        mock_llm = MagicMock()
        agent = ConcreteAgent(llm=mock_llm)
        assert agent.llm is mock_llm

    def test_custom_model_and_tokens(self) -> None:
        """Test custom model and max_tokens configuration."""
        agent = ConcreteAgent(model="claude-3-opus-20240229", max_tokens=8192)
        assert agent.model == "claude-3-opus-20240229"
        assert agent.max_tokens == 8192

    @pytest.mark.asyncio
    async def test_invoke(self) -> None:
        """Test the invoke method."""
        agent = ConcreteAgent()
        state = AgentState()

        result = await agent.invoke(state)

        assert len(result.messages) == 1
        assert result.messages[0]["content"] == "Test response"
        assert result.context["processed"] is True

    @pytest.mark.asyncio
    async def test_call_method(self) -> None:
        """Test that agent can be called as a function."""
        agent = ConcreteAgent()
        state = AgentState()

        result = await agent(state)

        assert result.context["processed"] is True

    @pytest.mark.asyncio
    async def test_call_handles_errors(self) -> None:
        """Test that __call__ handles errors gracefully."""

        class FailingAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "failing_agent"

            @property
            def description(self) -> str:
                return "Agent that fails"

            @property
            def system_prompt(self) -> str:
                return "You are a failing agent."

            async def invoke(self, state: AgentState) -> AgentState:  # noqa: ARG002
                raise ValueError("Intentional failure")

        agent = FailingAgent()
        state = AgentState()

        result = await agent(state)

        assert len(result.errors) == 1
        assert "failing_agent: Intentional failure" in result.errors[0]

    @pytest.mark.asyncio
    async def test_call_llm(self) -> None:
        """Test the _call_llm method."""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "Hello"}
        mock_response.content = [mock_content]

        mock_llm = MagicMock()
        mock_llm.messages.create.return_value = mock_response

        # Disable caching to test standard flow
        agent = ConcreteAgent(llm=mock_llm, enable_cache=False)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hi"}]

        result = await agent._call_llm(messages)

        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20

        mock_llm.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="You are a test agent.",
            messages=messages,
        )

    @pytest.mark.asyncio
    async def test_call_llm_with_tools(self) -> None:
        """Test _call_llm includes tools when provided."""
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "tool_use", "name": "test_tool"}
        mock_response.content = [mock_content]

        mock_llm = MagicMock()
        mock_llm.messages.create.return_value = mock_response

        # Disable caching to test standard flow
        agent = ConcreteAgent(llm=mock_llm, enable_cache=False)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Use a tool"}]
        tools: list[dict[str, Any]] = [{"name": "test_tool", "description": "A test tool"}]

        result = await agent._call_llm(messages, tools=tools)

        assert result["stop_reason"] == "tool_use"
        mock_llm.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="You are a test agent.",
            messages=messages,
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_call_llm_with_caching_enabled(self) -> None:
        """Test _call_llm uses cached prompt structure when caching is enabled."""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_creation_input_tokens = 100
        mock_response.usage.cache_read_input_tokens = 0
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "Hello"}
        mock_response.content = [mock_content]

        mock_llm = MagicMock()
        mock_llm.messages.create.return_value = mock_response

        agent = ConcreteAgent(llm=mock_llm, enable_cache=True)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hi"}]

        result = await agent._call_llm(messages)

        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert agent.last_cache_metrics is not None
        assert agent.last_cache_metrics.cache_creation_tokens == 100

        # With caching enabled, system should be a list of blocks, not a string
        call_args = mock_llm.messages.create.call_args
        assert isinstance(call_args.kwargs["system"], list)
        assert "cache_control" in call_args.kwargs["system"][0]

    @pytest.mark.asyncio
    async def test_call_llm_cache_hit_metrics(self) -> None:
        """Test that cache hit metrics are properly tracked."""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 500  # Cache hit
        mock_content = MagicMock()
        mock_content.model_dump.return_value = {"type": "text", "text": "Cached response"}
        mock_response.content = [mock_content]

        mock_llm = MagicMock()
        mock_llm.messages.create.return_value = mock_response

        agent = ConcreteAgent(llm=mock_llm, enable_cache=True)
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Hi"}]

        await agent._call_llm(messages)

        assert agent.last_cache_metrics is not None
        assert agent.last_cache_metrics.is_cache_hit is True
        assert agent.last_cache_metrics.cache_read_tokens == 500
        assert agent.last_cache_metrics.cache_hit_rate > 0.9  # 500/(500+10) ≈ 98%
