"""Tests for Langfuse tracing integration."""

import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.observability.tracing import (
    SpanContext,
    SpanLevel,
    SpanMetadata,
    TraceContext,
    agent_span,
    flush_traces,
    generation_span,
    get_langfuse_client,
    planning_span,
    postprocessing_span,
    preprocessing_span,
    request_span,
    shutdown_tracing,
    tool_span,
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


# ============================================================================
# Span Hierarchy Tests
# ============================================================================


class TestSpanLevel:
    """Tests for SpanLevel enum."""

    def test_all_levels_defined(self) -> None:
        """Test that all expected span levels are defined."""
        assert SpanLevel.SESSION.value == "session"
        assert SpanLevel.REQUEST.value == "request"
        assert SpanLevel.PREPROCESSING.value == "preprocessing"
        assert SpanLevel.AGENT.value == "agent"
        assert SpanLevel.PLANNING.value == "planning"
        assert SpanLevel.TOOL.value == "tool"
        assert SpanLevel.GENERATION.value == "generation"
        assert SpanLevel.POSTPROCESSING.value == "postprocessing"

    def test_level_count(self) -> None:
        """Test that we have all 8 span levels."""
        assert len(SpanLevel) == 8


class TestSpanMetadata:
    """Tests for SpanMetadata dataclass."""

    def test_default_values(self) -> None:
        """Test SpanMetadata with default values."""
        metadata = SpanMetadata(level=SpanLevel.AGENT)

        assert metadata.level == SpanLevel.AGENT
        assert metadata.parent_span_id is None
        assert metadata.agent_name is None
        assert metadata.tool_name is None
        assert metadata.model is None
        assert metadata.input_data is None
        assert metadata.output_data is None
        assert metadata.reasoning == []
        assert isinstance(metadata.started_at, datetime)
        assert metadata.ended_at is None
        assert metadata.error is None

    def test_all_fields_set(self) -> None:
        """Test SpanMetadata with all fields set."""
        started = datetime.now(UTC)
        ended = datetime.now(UTC)

        metadata = SpanMetadata(
            level=SpanLevel.TOOL,
            parent_span_id="parent-123",
            agent_name="research_agent",
            tool_name="get_market_data",
            model="claude-sonnet-4-20250514",
            input_data={"symbol": "AAPL"},
            output_data={"price": 150.0},
            reasoning=["Step 1", "Step 2"],
            started_at=started,
            ended_at=ended,
            error="Test error",
        )

        assert metadata.level == SpanLevel.TOOL
        assert metadata.parent_span_id == "parent-123"
        assert metadata.agent_name == "research_agent"
        assert metadata.tool_name == "get_market_data"
        assert metadata.model == "claude-sonnet-4-20250514"
        assert metadata.input_data == {"symbol": "AAPL"}
        assert metadata.output_data == {"price": 150.0}
        assert metadata.reasoning == ["Step 1", "Step 2"]
        assert metadata.started_at == started
        assert metadata.ended_at == ended
        assert metadata.error == "Test error"

    def test_to_dict(self) -> None:
        """Test SpanMetadata.to_dict method."""
        metadata = SpanMetadata(
            level=SpanLevel.AGENT,
            parent_span_id="parent-123",
            agent_name="test_agent",
            input_data={"key": "value"},
            reasoning=["Reasoning step"],
        )

        result = metadata.to_dict()

        assert result["level"] == "agent"
        assert result["parent_span_id"] == "parent-123"
        assert result["agent_name"] == "test_agent"
        assert result["has_input"] is True
        assert result["has_output"] is False
        assert result["has_reasoning"] is True
        assert result["has_error"] is False
        assert "started_at" in result

    def test_to_dict_with_error(self) -> None:
        """Test SpanMetadata.to_dict with error set."""
        metadata = SpanMetadata(level=SpanLevel.TOOL, error="Something went wrong")

        result = metadata.to_dict()

        assert result["has_error"] is True


class TestSpanContext:
    """Tests for SpanContext context manager."""

    def test_initialization(self) -> None:
        """Test SpanContext initialization."""
        ctx = SpanContext(
            name="test_span",
            level=SpanLevel.AGENT,
            parent_span_id="parent-123",
            agent_name="test_agent",
            metadata={"key": "value"},
            tags=["tag1"],
        )

        assert ctx.name == "test_span"
        assert ctx.level == SpanLevel.AGENT
        assert ctx.parent_span_id == "parent-123"
        assert ctx.metadata == {"key": "value"}
        assert ctx.tags == ["tag1"]

    def test_add_reasoning(self) -> None:
        """Test adding reasoning traces."""
        ctx = SpanContext(name="test", level=SpanLevel.PLANNING)

        ctx.add_reasoning("Step 1: Analyze data")
        ctx.add_reasoning("Step 2: Make decision")

        assert len(ctx._span_metadata.reasoning) == 2
        assert ctx._span_metadata.reasoning[0] == "Step 1: Analyze data"
        assert ctx._span_metadata.reasoning[1] == "Step 2: Make decision"

    def test_set_input(self) -> None:
        """Test setting input data."""
        ctx = SpanContext(name="test", level=SpanLevel.TOOL)

        ctx.set_input({"symbol": "AAPL", "period": "1d"})

        assert ctx._span_metadata.input_data == {"symbol": "AAPL", "period": "1d"}

    def test_set_input_when_capture_disabled(self) -> None:
        """Test that input is not set when capture is disabled."""
        ctx = SpanContext(name="test", level=SpanLevel.TOOL, capture_input=False)

        ctx.set_input({"data": "value"})

        assert ctx._span_metadata.input_data is None

    def test_set_output(self) -> None:
        """Test setting output data."""
        ctx = SpanContext(name="test", level=SpanLevel.TOOL)

        ctx.set_output({"price": 150.0, "volume": 1000000})

        assert ctx._span_metadata.output_data == {"price": 150.0, "volume": 1000000}

    def test_set_output_when_capture_disabled(self) -> None:
        """Test that output is not set when capture is disabled."""
        ctx = SpanContext(name="test", level=SpanLevel.TOOL, capture_output=False)

        ctx.set_output({"data": "value"})

        assert ctx._span_metadata.output_data is None

    def test_set_error_with_string(self) -> None:
        """Test setting error with string message."""
        ctx = SpanContext(name="test", level=SpanLevel.TOOL)

        ctx.set_error("Something went wrong")

        assert ctx._span_metadata.error == "Something went wrong"

    def test_set_error_with_exception(self) -> None:
        """Test setting error with exception."""
        ctx = SpanContext(name="test", level=SpanLevel.TOOL)

        ctx.set_error(ValueError("Invalid value"))

        assert ctx._span_metadata.error == "Invalid value"

    @pytest.mark.asyncio
    @patch("src.observability.tracing.get_client")
    async def test_async_context_manager(self, mock_get_client: MagicMock) -> None:
        """Test SpanContext as async context manager."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_span.id = "span-123"
        mock_client.span.return_value = mock_span
        mock_get_client.return_value = mock_client

        async with SpanContext(name="test", level=SpanLevel.AGENT) as span:
            assert span.name == "test"
            span.add_reasoning("Test reasoning")

        mock_client.span.assert_called_once()
        mock_span.end.assert_called_once()

    @patch("src.observability.tracing.get_client")
    def test_sync_context_manager(self, mock_get_client: MagicMock) -> None:
        """Test SpanContext as sync context manager."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_span.id = "span-456"
        mock_client.span.return_value = mock_span
        mock_get_client.return_value = mock_client

        with SpanContext(name="test", level=SpanLevel.TOOL) as span:
            assert span.name == "test"
            span.set_input({"key": "value"})

        mock_client.span.assert_called_once()
        mock_span.end.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.observability.tracing.get_client")
    async def test_context_sets_error_on_exception(
        self, mock_get_client: MagicMock
    ) -> None:
        """Test that context manager captures exceptions."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_client.span.return_value = mock_span
        mock_get_client.return_value = mock_client

        with pytest.raises(ValueError):
            async with SpanContext(name="test", level=SpanLevel.TOOL) as span:
                raise ValueError("Test error")

        # Should still end the span even after exception
        mock_span.end.assert_called_once()

    @patch("src.observability.tracing.get_client")
    def test_span_id_property(self, mock_get_client: MagicMock) -> None:
        """Test that span_id is accessible."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_span.id = "span-789"
        mock_client.span.return_value = mock_span
        mock_get_client.return_value = mock_client

        with SpanContext(name="test", level=SpanLevel.AGENT) as span:
            assert span.span_id == "span-789"

    @patch("src.observability.tracing.get_client")
    def test_handles_client_errors_gracefully(self, mock_get_client: MagicMock) -> None:
        """Test that span context handles Langfuse client errors."""
        mock_get_client.side_effect = Exception("Connection error")

        # Should not raise, just log warning
        with SpanContext(name="test", level=SpanLevel.AGENT) as span:
            span.add_reasoning("Still works")

        assert span._span_metadata.reasoning == ["Still works"]


class TestConvenienceSpanFunctions:
    """Tests for convenience span creation functions."""

    def test_tool_span(self) -> None:
        """Test tool_span factory function."""
        span = tool_span(
            "get_market_data",
            input_data={"symbol": "AAPL"},
            metadata={"cache": True},
        )

        assert span.name == "get_market_data"
        assert span.level == SpanLevel.TOOL
        assert span._span_metadata.tool_name == "get_market_data"
        assert span._span_metadata.input_data == {"symbol": "AAPL"}

    def test_agent_span(self) -> None:
        """Test agent_span factory function."""
        span = agent_span("research_agent", metadata={"version": "1.0"})

        assert span.name == "research_agent"
        assert span.level == SpanLevel.AGENT
        assert span._span_metadata.agent_name == "research_agent"

    def test_generation_span(self) -> None:
        """Test generation_span factory function."""
        span = generation_span(
            "analyze_portfolio",
            model="claude-sonnet-4-20250514",
        )

        assert span.name == "analyze_portfolio"
        assert span.level == SpanLevel.GENERATION
        assert span._span_metadata.model == "claude-sonnet-4-20250514"

    def test_preprocessing_span(self) -> None:
        """Test preprocessing_span factory function."""
        span = preprocessing_span(metadata={"step": "validation"})

        assert span.name == "preprocessing"
        assert span.level == SpanLevel.PREPROCESSING

    def test_preprocessing_span_custom_name(self) -> None:
        """Test preprocessing_span with custom name."""
        span = preprocessing_span("input_validation")

        assert span.name == "input_validation"
        assert span.level == SpanLevel.PREPROCESSING

    def test_postprocessing_span(self) -> None:
        """Test postprocessing_span factory function."""
        span = postprocessing_span(metadata={"step": "formatting"})

        assert span.name == "postprocessing"
        assert span.level == SpanLevel.POSTPROCESSING

    def test_postprocessing_span_custom_name(self) -> None:
        """Test postprocessing_span with custom name."""
        span = postprocessing_span("format_response")

        assert span.name == "format_response"
        assert span.level == SpanLevel.POSTPROCESSING

    def test_planning_span(self) -> None:
        """Test planning_span factory function."""
        span = planning_span(metadata={"strategy": "risk-first"})

        assert span.name == "planning"
        assert span.level == SpanLevel.PLANNING

    def test_planning_span_custom_name(self) -> None:
        """Test planning_span with custom name."""
        span = planning_span("risk_evaluation")

        assert span.name == "risk_evaluation"
        assert span.level == SpanLevel.PLANNING

    def test_request_span(self) -> None:
        """Test request_span factory function."""
        span = request_span(
            "portfolio_analysis",
            session_id="sess-123",
            user_id="user-456",
        )

        assert span.name == "portfolio_analysis"
        assert span.level == SpanLevel.REQUEST
        assert span.metadata["session_id"] == "sess-123"
        assert span.metadata["user_id"] == "user-456"

    def test_request_span_without_ids(self) -> None:
        """Test request_span without session/user IDs."""
        span = request_span("simple_request")

        assert span.name == "simple_request"
        assert span.level == SpanLevel.REQUEST
        assert span.metadata == {}


class TestSpanHierarchyIntegration:
    """Integration tests for nested span hierarchy."""

    @pytest.mark.asyncio
    @patch("src.observability.tracing.get_client")
    async def test_nested_span_hierarchy(self, mock_get_client: MagicMock) -> None:
        """Test creating nested spans simulating real workflow."""
        mock_client = MagicMock()
        span_ids: list[str] = []
        call_count = 0

        def create_mock_span(**kwargs: MagicMock) -> MagicMock:
            nonlocal call_count
            mock_span = MagicMock()
            mock_span.id = f"span-{call_count}"
            span_ids.append(mock_span.id)
            call_count += 1
            return mock_span

        mock_client.span.side_effect = create_mock_span
        mock_get_client.return_value = mock_client

        # Simulate: Request → Agent → Tool
        async with request_span("portfolio_analysis", session_id="sess-1") as req:
            async with agent_span(
                "research_agent", parent_span_id=req.span_id
            ) as agent:
                agent.add_reasoning("Starting market research")

                async with tool_span(
                    "get_market_data",
                    parent_span_id=agent.span_id,
                    input_data={"symbol": "AAPL"},
                ) as tool:
                    tool.set_output({"price": 150.0})

        # Should have created 3 spans
        assert mock_client.span.call_count == 3

    @pytest.mark.asyncio
    @patch("src.observability.tracing.get_client")
    async def test_full_workflow_span_hierarchy(
        self, mock_get_client: MagicMock
    ) -> None:
        """Test complete workflow with preprocessing and postprocessing."""
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_span.id = "span-123"
        mock_client.span.return_value = mock_span
        mock_get_client.return_value = mock_client

        spans_created: list[str] = []

        async with request_span("workflow") as req:
            spans_created.append("request")

            async with preprocessing_span(parent_span_id=req.span_id) as pre:
                spans_created.append("preprocessing")
                pre.set_input({"raw": "data"})

            async with agent_span("agent", parent_span_id=req.span_id) as agent:
                spans_created.append("agent")

                async with planning_span(parent_span_id=agent.span_id) as plan:
                    spans_created.append("planning")
                    plan.add_reasoning("Analyzing...")

                async with generation_span(
                    "llm", model="claude", parent_span_id=agent.span_id
                ) as gen:
                    spans_created.append("generation")

            async with postprocessing_span(parent_span_id=req.span_id) as post:
                spans_created.append("postprocessing")
                post.set_output({"formatted": "result"})

        assert spans_created == [
            "request",
            "preprocessing",
            "agent",
            "planning",
            "generation",
            "postprocessing",
        ]
