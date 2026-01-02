"""Base tool interface for agent tools.

This module provides the abstract base class for all tools used by agents,
with support for real API execution and mock fallbacks.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _get_traced_tool_decorator() -> Callable[[str], Callable[[Callable[P, T]], Callable[P, T]]]:
    """Get the traced_tool decorator if available, otherwise return a no-op.

    This allows the tools module to work independently of the tracing module.
    """
    try:
        from src.observability.tracing import (  # type: ignore[import-untyped]
            traced_tool as _traced_tool,
        )

        return _traced_tool  # type: ignore[no-any-return]
    except ImportError:
        # Tracing not available, return no-op decorator
        def noop_decorator(
            name: str,  # noqa: ARG001
        ) -> Callable[[Callable[P, T]], Callable[P, T]]:
            def decorator(func: Callable[P, T]) -> Callable[P, T]:
                @wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    return func(*args, **kwargs)

                return wrapper

            return decorator

        return noop_decorator


traced_tool = _get_traced_tool_decorator()


class ToolInput(BaseModel):
    """Base class for tool input validation."""

    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    """Base class for tool output validation."""

    model_config = ConfigDict(extra="allow")

    success: bool = True
    error: str | None = None


InputT = TypeVar("InputT", bound=ToolInput)
OutputT = TypeVar("OutputT", bound=ToolOutput)


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str, cause: Exception | None = None) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"{tool_name}: {message}")


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all agent tools.

    Provides a consistent interface for tool execution with:
    - Input/output validation via Pydantic models
    - Automatic fallback to mock data on real API failures
    - Execution tracing via Langfuse
    - Structured logging

    Type Parameters:
        InputT: Pydantic model for input validation
        OutputT: Pydantic model for output validation

    Example:
        class MarketDataInput(ToolInput):
            symbol: str

        class MarketDataOutput(ToolOutput):
            price: float
            volume: int

        class MarketDataTool(BaseTool[MarketDataInput, MarketDataOutput]):
            name = "market_data"
            description = "Fetches current market data for a symbol"

            async def _execute_real(self, input: MarketDataInput) -> MarketDataOutput:
                # Call real API
                ...

            async def _execute_mock(self, input: MarketDataInput) -> MarketDataOutput:
                return MarketDataOutput(price=150.0, volume=1000000)
    """

    #: Unique name for the tool
    name: str
    #: Human-readable description of what the tool does
    description: str
    #: Whether to use mock data instead of real API
    use_mock: bool = False
    #: Whether to fallback to mock on real API failure
    fallback_to_mock: bool = True

    def __init__(
        self,
        use_mock: bool | None = None,
        fallback_to_mock: bool | None = None,
    ) -> None:
        """Initialize the tool.

        Args:
            use_mock: Override default mock setting.
            fallback_to_mock: Override default fallback setting.
        """
        if use_mock is not None:
            self.use_mock = use_mock
        if fallback_to_mock is not None:
            self.fallback_to_mock = fallback_to_mock

        self._logger = logger.bind(tool=self.name)

    @property
    @abstractmethod
    def input_schema(self) -> type[InputT]:
        """Return the Pydantic model for input validation."""
        ...

    @property
    @abstractmethod
    def output_schema(self) -> type[OutputT]:
        """Return the Pydantic model for output validation."""
        ...

    @abstractmethod
    async def _execute_real(self, input_data: InputT) -> OutputT:
        """Execute the tool with real API.

        Args:
            input_data: Validated input data.

        Returns:
            Tool output.

        Raises:
            Exception: If the real API call fails.
        """
        ...

    @abstractmethod
    async def _execute_mock(self, input_data: InputT) -> OutputT:
        """Execute the tool with mock data.

        Args:
            input_data: Validated input data.

        Returns:
            Mock tool output.
        """
        ...

    def _validate_input(self, input_data: dict[str, Any] | InputT) -> InputT:
        """Validate and parse input data.

        Args:
            input_data: Raw input dict or already validated model.

        Returns:
            Validated input model.

        Raises:
            ToolExecutionError: If validation fails.
        """
        if isinstance(input_data, self.input_schema):
            return input_data

        try:
            return self.input_schema.model_validate(input_data)
        except ValidationError as e:
            raise ToolExecutionError(
                self.name,
                f"Input validation failed: {e}",
                cause=e,
            ) from e

    def _validate_output(self, output_data: OutputT) -> OutputT:
        """Validate output data.

        Args:
            output_data: Output to validate.

        Returns:
            Validated output model.

        Raises:
            ToolExecutionError: If validation fails.
        """
        try:
            # Re-validate to ensure all fields are correct
            return self.output_schema.model_validate(output_data.model_dump())
        except ValidationError as e:
            raise ToolExecutionError(
                self.name,
                f"Output validation failed: {e}",
                cause=e,
            ) from e

    async def execute(self, input_data: dict[str, Any] | InputT) -> OutputT:
        """Execute the tool with input validation and fallback handling.

        This is the main entry point for tool execution. It:
        1. Validates input against the input schema
        2. Executes real API (or mock if use_mock=True)
        3. Falls back to mock on failure if fallback_to_mock=True
        4. Validates output against the output schema

        Args:
            input_data: Raw input dict or validated input model.

        Returns:
            Validated tool output.

        Raises:
            ToolExecutionError: If execution fails and fallback is disabled.
        """
        # Apply tracing decorator dynamically
        return await self._traced_execute(input_data)

    @traced_tool("tool_execution")
    async def _traced_execute(self, input_data: dict[str, Any] | InputT) -> OutputT:
        """Internal traced execution method."""
        validated_input = self._validate_input(input_data)

        self._logger.info(
            "tool_execute_start",
            use_mock=self.use_mock,
            input=validated_input.model_dump(),
        )

        output: OutputT
        if self.use_mock:
            output = await self._execute_mock(validated_input)
            self._logger.debug("tool_executed_mock")
        else:
            try:
                output = await self._execute_real(validated_input)
                self._logger.debug("tool_executed_real")
            except Exception as e:
                self._logger.warning(
                    "tool_real_execution_failed",
                    error=str(e),
                    fallback_to_mock=self.fallback_to_mock,
                )
                if self.fallback_to_mock:
                    output = await self._execute_mock(validated_input)
                    output.success = True
                    output.error = f"Fallback to mock: {e}"
                    self._logger.info("tool_fallback_to_mock")
                else:
                    raise ToolExecutionError(
                        self.name,
                        f"Execution failed: {e}",
                        cause=e,
                    ) from e

        validated_output = self._validate_output(output)
        self._logger.info(
            "tool_execute_complete",
            success=validated_output.success,
        )
        return validated_output

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert tool to Anthropic tool format for LLM function calling.

        Returns:
            Tool definition in Anthropic's expected format.
        """
        schema = self.input_schema.model_json_schema()
        # Remove title and description from schema as they go in tool definition
        schema.pop("title", None)
        schema.pop("description", None)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }


class ToolRegistry:
    """Registry for managing available tools.

    Provides a central place to register and retrieve tools by name.

    Example:
        registry = ToolRegistry()
        registry.register(MarketDataTool())
        tool = registry.get("market_data")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, BaseTool[Any, Any]] = {}
        self._logger = logger.bind(component="tool_registry")

    def register(self, tool: BaseTool[Any, Any]) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        self._logger.debug("tool_registered", tool=tool.name)

    def get(self, name: str) -> BaseTool[Any, Any]:
        """Get a tool by name.

        Args:
            name: Name of the tool to retrieve.

        Returns:
            The registered tool.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Convert all registered tools to Anthropic format.

        Returns:
            List of tool definitions in Anthropic's expected format.
        """
        return [tool.to_anthropic_tool() for tool in self._tools.values()]

    def set_mock_mode(self, use_mock: bool) -> None:
        """Set mock mode for all registered tools.

        Args:
            use_mock: Whether to use mock mode.
        """
        for tool in self._tools.values():
            tool.use_mock = use_mock
        self._logger.info("mock_mode_set", use_mock=use_mock, tool_count=len(self._tools))


# Global default registry
default_registry = ToolRegistry()
