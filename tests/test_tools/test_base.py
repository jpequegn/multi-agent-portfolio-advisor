"""Tests for the base tool interface."""

import pytest
from pydantic import ValidationError

from src.tools.base import (
    BaseTool,
    ToolExecutionError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


# Test input/output models
class SampleInput(ToolInput):
    """Sample input model for testing."""

    value: str


class SampleOutput(ToolOutput):
    """Sample output model for testing."""

    result: str


# Concrete tool for testing
class SampleTool(BaseTool[SampleInput, SampleOutput]):
    """Concrete tool implementation for testing."""

    name = "test_tool"
    description = "A test tool"

    def __init__(
        self,
        use_mock: bool | None = None,
        fallback_to_mock: bool | None = None,
        fail_real: bool = False,
    ) -> None:
        super().__init__(use_mock=use_mock, fallback_to_mock=fallback_to_mock)
        self.fail_real = fail_real
        self.real_called = False
        self.mock_called = False

    @property
    def input_schema(self) -> type[SampleInput]:
        return SampleInput

    @property
    def output_schema(self) -> type[SampleOutput]:
        return SampleOutput

    async def _execute_real(self, input_data: SampleInput) -> SampleOutput:
        self.real_called = True
        if self.fail_real:
            raise ConnectionError("API unavailable")
        return SampleOutput(result=f"real:{input_data.value}")

    async def _execute_mock(self, input_data: SampleInput) -> SampleOutput:
        self.mock_called = True
        return SampleOutput(result=f"mock:{input_data.value}")


class TestToolInput:
    """Tests for ToolInput base class."""

    def test_forbids_extra_fields(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            SampleInput(value="test", extra="not allowed")  # type: ignore[call-arg]

    def test_valid_input(self) -> None:
        """Test valid input creation."""
        input_data = SampleInput(value="test")
        assert input_data.value == "test"


class TestToolOutput:
    """Tests for ToolOutput base class."""

    def test_default_values(self) -> None:
        """Test default values."""
        output = SampleOutput(result="test")
        assert output.success is True
        assert output.error is None

    def test_allows_extra_fields(self) -> None:
        """Test that extra fields are allowed in output."""
        output = SampleOutput(result="test", extra_data="allowed")  # type: ignore[call-arg]
        assert output.extra_data == "allowed"  # type: ignore[attr-defined]


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    @pytest.mark.asyncio
    async def test_execute_real(self) -> None:
        """Test real execution."""
        tool = SampleTool()
        result = await tool.execute({"value": "test"})

        assert result.result == "real:test"
        assert result.success is True
        assert tool.real_called is True
        assert tool.mock_called is False

    @pytest.mark.asyncio
    async def test_execute_mock(self) -> None:
        """Test mock execution."""
        tool = SampleTool(use_mock=True)
        result = await tool.execute({"value": "test"})

        assert result.result == "mock:test"
        assert result.success is True
        assert tool.real_called is False
        assert tool.mock_called is True

    @pytest.mark.asyncio
    async def test_fallback_to_mock(self) -> None:
        """Test fallback to mock on real failure."""
        tool = SampleTool(fail_real=True, fallback_to_mock=True)
        result = await tool.execute({"value": "test"})

        assert result.result == "mock:test"
        assert result.success is True
        assert "Fallback to mock" in (result.error or "")
        assert tool.real_called is True
        assert tool.mock_called is True

    @pytest.mark.asyncio
    async def test_no_fallback_raises(self) -> None:
        """Test that error is raised when fallback is disabled."""
        tool = SampleTool(fail_real=True, fallback_to_mock=False)

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute({"value": "test"})

        assert "test_tool" in str(exc_info.value)
        assert tool.real_called is True
        assert tool.mock_called is False

    @pytest.mark.asyncio
    async def test_input_validation(self) -> None:
        """Test input validation."""
        tool = SampleTool()

        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.execute({"invalid_field": "test"})

        assert "Input validation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_accepts_model_input(self) -> None:
        """Test that validated model can be passed directly."""
        tool = SampleTool()
        input_data = SampleInput(value="test")
        result = await tool.execute(input_data)

        assert result.result == "real:test"

    def test_to_anthropic_tool(self) -> None:
        """Test conversion to Anthropic tool format."""
        tool = SampleTool()
        anthropic_tool = tool.to_anthropic_tool()

        assert anthropic_tool["name"] == "test_tool"
        assert anthropic_tool["description"] == "A test tool"
        assert "input_schema" in anthropic_tool
        assert "properties" in anthropic_tool["input_schema"]
        assert "value" in anthropic_tool["input_schema"]["properties"]


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)
        retrieved = registry.get("test_tool")

        assert retrieved is tool

    def test_register_duplicate_raises(self) -> None:
        """Test that registering duplicate raises error."""
        registry = ToolRegistry()
        tool1 = SampleTool()
        tool2 = SampleTool()

        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_get_unknown_raises(self) -> None:
        """Test that getting unknown tool raises error."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("unknown_tool")

    def test_list_tools(self) -> None:
        """Test listing registered tools."""
        registry = ToolRegistry()
        tool = SampleTool()

        assert registry.list_tools() == []

        registry.register(tool)
        assert registry.list_tools() == ["test_tool"]

    def test_to_anthropic_tools(self) -> None:
        """Test converting all tools to Anthropic format."""
        registry = ToolRegistry()
        tool = SampleTool()
        registry.register(tool)

        tools = registry.to_anthropic_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    def test_set_mock_mode(self) -> None:
        """Test setting mock mode for all tools."""
        registry = ToolRegistry()
        tool1 = SampleTool()
        tool2 = SampleTool()
        tool2.name = "test_tool_2"  # Change name to avoid duplicate

        registry.register(tool1)
        registry.register(tool2)

        assert tool1.use_mock is False
        assert tool2.use_mock is False

        registry.set_mock_mode(True)

        assert tool1.use_mock is True
        assert tool2.use_mock is True


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_error_message(self) -> None:
        """Test error message format."""
        error = ToolExecutionError("my_tool", "Something went wrong")
        assert str(error) == "my_tool: Something went wrong"

    def test_with_cause(self) -> None:
        """Test error with cause."""
        cause = ValueError("Original error")
        error = ToolExecutionError("my_tool", "Wrapper", cause=cause)

        assert error.cause is cause
        assert error.tool_name == "my_tool"
