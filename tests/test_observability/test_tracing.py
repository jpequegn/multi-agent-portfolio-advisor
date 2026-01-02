"""Tests for Langfuse tracing integration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.observability.tracing import (
    TraceContext,
    flush_traces,
    get_langfuse_client,
    shutdown_tracing,
    traced_agent,
    traced_generation,
    traced_tool,
)


class TestGetLangfuseClient:
    """Tests for get_langfuse_client function."""

    @patch.dict(
        os.environ,
        {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
            "LANGFUSE_HOST": "http://localhost:3000",
        },
    )
    @patch("src.observability.tracing.Langfuse")
    def test_creates_client_with_env_vars(self, mock_langfuse: MagicMock) -> None:
        """Test client creation with environment variables."""
        get_langfuse_client()

        mock_langfuse.assert_called_once_with(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://localhost:3000",
        )

    @patch.dict(os.environ, {}, clear=True)
    @patch("src.observability.tracing.get_client")
    def test_falls_back_to_get_client(self, mock_get_client: MagicMock) -> None:
        """Test fallback to get_client when env vars not set."""
        # Clear the env vars that might exist
        for key in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]:
            os.environ.pop(key, None)

        get_langfuse_client()

        mock_get_client.assert_called_once()


class TestTracedAgentDecorator:
    """Tests for traced_agent decorator."""

    @pytest.mark.asyncio
    async def test_async_function_decorated(self) -> None:
        """Test that async functions are properly decorated."""

        @traced_agent("test_agent")
        async def my_agent(value: int) -> int:
            return value * 2

        result = await my_agent(5)
        assert result == 10

    def test_sync_function_decorated(self) -> None:
        """Test that sync functions are properly decorated."""

        @traced_agent("test_agent")
        def my_agent(value: int) -> int:
            return value * 2

        result = my_agent(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Test that function metadata is preserved."""

        @traced_agent("test_agent")
        async def my_agent(value: int) -> int:
            """My agent docstring."""
            return value

        assert my_agent.__name__ == "my_agent"
        assert my_agent.__doc__ == "My agent docstring."


class TestTracedToolDecorator:
    """Tests for traced_tool decorator."""

    @pytest.mark.asyncio
    async def test_async_tool_decorated(self) -> None:
        """Test that async tools are properly decorated."""

        @traced_tool("market_data")
        async def get_data(symbol: str) -> dict[str, str]:
            return {"symbol": symbol}

        result = await get_data("AAPL")
        assert result == {"symbol": "AAPL"}

    def test_sync_tool_decorated(self) -> None:
        """Test that sync tools are properly decorated."""

        @traced_tool("market_data")
        def get_data(symbol: str) -> dict[str, str]:
            return {"symbol": symbol}

        result = get_data("AAPL")
        assert result == {"symbol": "AAPL"}


class TestTracedGenerationDecorator:
    """Tests for traced_generation decorator."""

    @pytest.mark.asyncio
    async def test_async_generation_decorated(self) -> None:
        """Test that async generations are properly decorated."""

        @traced_generation("llm_call", model="claude-sonnet-4-20250514")
        async def call_llm(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = await call_llm("Hello")
        assert result == "Response to: Hello"

    def test_sync_generation_decorated(self) -> None:
        """Test that sync generations are properly decorated."""

        @traced_generation("llm_call")
        def call_llm(prompt: str) -> str:
            return f"Response to: {prompt}"

        result = call_llm("Hello")
        assert result == "Response to: Hello"


class TestTraceContext:
    """Tests for TraceContext context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test TraceContext as async context manager."""
        async with TraceContext(
            session_id="test-session",
            user_id="test-user",
            metadata={"key": "value"},
            tags=["test"],
        ) as ctx:
            assert ctx.session_id == "test-session"
            assert ctx.user_id == "test-user"
            assert ctx.metadata == {"key": "value"}
            assert ctx.tags == ["test"]

    def test_sync_context_manager(self) -> None:
        """Test TraceContext as sync context manager."""
        with TraceContext(
            session_id="test-session",
            user_id="test-user",
        ) as ctx:
            assert ctx.session_id == "test-session"
            assert ctx.user_id == "test-user"

    def test_default_values(self) -> None:
        """Test TraceContext with default values."""
        ctx = TraceContext()
        assert ctx.session_id is None
        assert ctx.user_id is None
        assert ctx.metadata == {}
        assert ctx.tags == []


class TestFlushAndShutdown:
    """Tests for flush and shutdown functions."""

    @patch("src.observability.tracing.get_client")
    def test_flush_traces(self, mock_get_client: MagicMock) -> None:
        """Test flush_traces calls client.flush()."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        flush_traces()

        mock_client.flush.assert_called_once()

    @patch("src.observability.tracing.get_client")
    def test_flush_traces_handles_error(self, mock_get_client: MagicMock) -> None:
        """Test flush_traces handles errors gracefully."""
        mock_get_client.side_effect = Exception("Connection error")

        # Should not raise
        flush_traces()

    @patch("src.observability.tracing.get_client")
    def test_shutdown_tracing(self, mock_get_client: MagicMock) -> None:
        """Test shutdown_tracing calls client.shutdown()."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        shutdown_tracing()

        mock_client.shutdown.assert_called_once()

    @patch("src.observability.tracing.get_client")
    def test_shutdown_tracing_handles_error(self, mock_get_client: MagicMock) -> None:
        """Test shutdown_tracing handles errors gracefully."""
        mock_get_client.side_effect = Exception("Connection error")

        # Should not raise
        shutdown_tracing()
