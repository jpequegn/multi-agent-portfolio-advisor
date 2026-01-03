"""Error handling and recovery for multi-agent workflows.

This module provides:
- Custom exception hierarchy for workflow errors
- Retry logic with exponential backoff
- Fallback strategies per agent
- Recovery patterns for graceful degradation

Exception Hierarchy:
    PortfolioAdvisorError (base)
    ├── AgentError - Agent execution failures
    ├── ToolError - Tool execution failures
    ├── StateError - State validation failures
    ├── WorkflowTimeoutError - Operation timeouts
    └── RecoveryError - Recovery attempt failures
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ============================================================================
# Exception Hierarchy
# ============================================================================


class PortfolioAdvisorError(Exception):
    """Base exception for all Portfolio Advisor errors.

    Attributes:
        message: Human-readable error message.
        details: Additional error details.
        recoverable: Whether the error can be recovered from.
        trace_id: Optional trace ID for observability.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        self.trace_id = trace_id

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "trace_id": self.trace_id,
        }


class AgentError(PortfolioAdvisorError):
    """Agent execution failed.

    Raised when an agent encounters an error during execution.

    Attributes:
        agent_name: Name of the agent that failed.
        stage: Stage of execution where failure occurred.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_name: str,
        stage: str = "execution",
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            message, details=details, recoverable=recoverable, trace_id=trace_id
        )
        self.agent_name = agent_name
        self.stage = stage

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({"agent_name": self.agent_name, "stage": self.stage})
        return base


class ToolError(PortfolioAdvisorError):
    """Tool execution failed.

    Raised when a tool encounters an error during execution.

    Attributes:
        tool_name: Name of the tool that failed.
        input_data: Input that caused the failure (sanitized).
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        input_data: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            message, details=details, recoverable=recoverable, trace_id=trace_id
        )
        self.tool_name = tool_name
        self.input_data = input_data or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({"tool_name": self.tool_name, "input_data": self.input_data})
        return base


class StateError(PortfolioAdvisorError):
    """State validation or persistence failed.

    Raised when workflow state is invalid or cannot be persisted.

    Attributes:
        state_key: The state key that caused the error.
        expected: Expected value/type (if applicable).
        actual: Actual value/type found.
    """

    def __init__(
        self,
        message: str,
        *,
        state_key: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            message, details=details, recoverable=recoverable, trace_id=trace_id
        )
        self.state_key = state_key
        self.expected = expected
        self.actual = actual

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {"state_key": self.state_key, "expected": self.expected, "actual": self.actual}
        )
        return base


class WorkflowTimeoutError(PortfolioAdvisorError):
    """Operation timed out.

    Raised when an operation exceeds its timeout limit.

    Attributes:
        timeout_seconds: The timeout that was exceeded.
        operation: Name of the operation that timed out.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: float,
        operation: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            message, details=details, recoverable=recoverable, trace_id=trace_id
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {"timeout_seconds": self.timeout_seconds, "operation": self.operation}
        )
        return base


class RecoveryError(PortfolioAdvisorError):
    """Recovery attempt failed.

    Raised when error recovery fails after all retries/fallbacks.

    Attributes:
        original_error: The original error that triggered recovery.
        recovery_attempts: Number of recovery attempts made.
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: Exception | None = None,
        recovery_attempts: int = 0,
        details: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(
            message, details=details, recoverable=False, trace_id=trace_id
        )
        self.original_error = original_error
        self.recovery_attempts = recovery_attempts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "original_error": str(self.original_error) if self.original_error else None,
                "recovery_attempts": self.recovery_attempts,
            }
        )
        return base


# ============================================================================
# Retry Configuration
# ============================================================================


class RetryStrategy(Enum):
    """Retry strategies for different scenarios."""

    NONE = "none"  # No retries
    QUICK = "quick"  # 2 retries, short backoff
    STANDARD = "standard"  # 3 retries, standard backoff
    PERSISTENT = "persistent"  # 5 retries, longer backoff


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including initial).
        min_wait_seconds: Minimum wait between retries.
        max_wait_seconds: Maximum wait between retries.
        multiplier: Exponential backoff multiplier.
        retry_exceptions: Exception types to retry on.
    """

    max_attempts: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 30.0
    multiplier: float = 2.0
    retry_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (AgentError, ToolError, WorkflowTimeoutError)
    )

    @classmethod
    def from_strategy(cls, strategy: RetryStrategy) -> "RetryConfig":
        """Create config from a strategy preset.

        Args:
            strategy: The retry strategy to use.

        Returns:
            RetryConfig for the strategy.
        """
        configs = {
            RetryStrategy.NONE: cls(max_attempts=1),
            RetryStrategy.QUICK: cls(
                max_attempts=2, min_wait_seconds=0.5, max_wait_seconds=5.0
            ),
            RetryStrategy.STANDARD: cls(
                max_attempts=3, min_wait_seconds=1.0, max_wait_seconds=30.0
            ),
            RetryStrategy.PERSISTENT: cls(
                max_attempts=5, min_wait_seconds=2.0, max_wait_seconds=60.0
            ),
        }
        return configs[strategy]


def with_retry(
    config: RetryConfig | RetryStrategy | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for adding retry logic with exponential backoff.

    Args:
        config: Retry configuration or strategy preset.
        on_retry: Optional callback called on each retry.

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(RetryStrategy.STANDARD)
        async def fetch_data():
            return await api.get_data()

        @with_retry(RetryConfig(max_attempts=5))
        async def resilient_operation():
            ...
    """
    if config is None:
        retry_config = RetryConfig()
    elif isinstance(config, RetryStrategy):
        retry_config = RetryConfig.from_strategy(config)
    else:
        retry_config = config

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0

            async for attempt_context in AsyncRetrying(
                stop=stop_after_attempt(retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_config.multiplier,
                    min=retry_config.min_wait_seconds,
                    max=retry_config.max_wait_seconds,
                ),
                retry=retry_if_exception_type(retry_config.retry_exceptions),
                reraise=True,
            ):
                with attempt_context:
                    attempt += 1
                    if attempt > 1:
                        logger.info(
                            "retry_attempt",
                            function=fn.__name__,
                            attempt=attempt,
                            max_attempts=retry_config.max_attempts,
                        )
                        if on_retry and attempt_context.retry_state.outcome:
                            exc = attempt_context.retry_state.outcome.exception()
                            if exc:
                                on_retry(exc, attempt)

                    return await fn(*args, **kwargs)

            # This should not be reached due to reraise=True
            raise RuntimeError("Retry loop exited unexpectedly")

        return wrapper

    return decorator


# ============================================================================
# Fallback Strategies
# ============================================================================


class FallbackType(Enum):
    """Types of fallback behaviors."""

    CACHED_DATA = "cached_data"  # Use cached/stale data
    MOCK_DATA = "mock_data"  # Use mock/default data
    PARTIAL_RESULT = "partial_result"  # Return partial results
    SKIP = "skip"  # Skip the operation entirely
    ERROR = "error"  # Propagate the error


@dataclass
class FallbackResult:
    """Result from a fallback operation.

    Attributes:
        value: The fallback value.
        fallback_type: Type of fallback used.
        original_error: The error that triggered fallback.
        warning: Warning message for the user.
    """

    value: Any
    fallback_type: FallbackType
    original_error: Exception | None = None
    warning: str | None = None

    @property
    def is_fallback(self) -> bool:
        """Check if this is a fallback result."""
        return True


@dataclass
class AgentFallbackConfig:
    """Fallback configuration for an agent.

    Attributes:
        agent_name: Name of the agent.
        fallback_type: Type of fallback to use.
        fallback_fn: Function to generate fallback data.
        warning_message: Warning to show when fallback is used.
        max_retries: Maximum retries before fallback.
    """

    agent_name: str
    fallback_type: FallbackType = FallbackType.PARTIAL_RESULT
    fallback_fn: Callable[..., Any] | None = None
    warning_message: str | None = None
    max_retries: int = 3


# Default fallback configurations per agent
AGENT_FALLBACK_CONFIGS: dict[str, AgentFallbackConfig] = {
    "research_agent": AgentFallbackConfig(
        agent_name="research_agent",
        fallback_type=FallbackType.CACHED_DATA,
        warning_message="Using cached market data; prices may be outdated",
        max_retries=3,
    ),
    "analysis_agent": AgentFallbackConfig(
        agent_name="analysis_agent",
        fallback_type=FallbackType.PARTIAL_RESULT,
        warning_message="Using simplified analysis; some metrics unavailable",
        max_retries=2,
    ),
    "recommendation_agent": AgentFallbackConfig(
        agent_name="recommendation_agent",
        fallback_type=FallbackType.PARTIAL_RESULT,
        warning_message="Providing conservative recommendations due to data limitations",
        max_retries=2,
    ),
}


def get_agent_fallback_config(agent_name: str) -> AgentFallbackConfig:
    """Get fallback configuration for an agent.

    Args:
        agent_name: Name of the agent.

    Returns:
        Fallback configuration for the agent.
    """
    return AGENT_FALLBACK_CONFIGS.get(
        agent_name,
        AgentFallbackConfig(
            agent_name=agent_name,
            fallback_type=FallbackType.ERROR,
            max_retries=1,
        ),
    )


# ============================================================================
# Error Recovery
# ============================================================================


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt.

    Attributes:
        timestamp: When the attempt was made.
        strategy: Recovery strategy used.
        success: Whether recovery succeeded.
        error: Error if recovery failed.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    strategy: str = ""
    success: bool = False
    error: str | None = None


@dataclass
class RecoveryContext:
    """Context for error recovery operations.

    Tracks recovery attempts and provides utilities for recovery.

    Attributes:
        original_error: The error that triggered recovery.
        agent_name: Name of the agent being recovered.
        attempts: List of recovery attempts.
        max_attempts: Maximum recovery attempts allowed.
    """

    original_error: Exception
    agent_name: str | None = None
    attempts: list[RecoveryAttempt] = field(default_factory=list)
    max_attempts: int = 3

    def add_attempt(self, strategy: str, success: bool, error: str | None = None) -> None:
        """Record a recovery attempt."""
        self.attempts.append(
            RecoveryAttempt(strategy=strategy, success=success, error=error)
        )

    @property
    def attempt_count(self) -> int:
        """Number of recovery attempts made."""
        return len(self.attempts)

    @property
    def can_retry(self) -> bool:
        """Check if more recovery attempts are allowed."""
        return self.attempt_count < self.max_attempts

    @property
    def last_attempt(self) -> RecoveryAttempt | None:
        """Get the last recovery attempt."""
        return self.attempts[-1] if self.attempts else None


class ErrorRecoveryManager:
    """Manager for error recovery operations.

    Coordinates retry logic, fallback execution, and recovery tracking.

    Example:
        manager = ErrorRecoveryManager()

        try:
            result = await agent.execute(state)
        except AgentError as e:
            result = await manager.recover(
                error=e,
                agent_name="research_agent",
                fallback_fn=lambda: cached_data,
            )
    """

    def __init__(self) -> None:
        """Initialize the recovery manager."""
        self._active_recoveries: dict[str, RecoveryContext] = {}

    async def recover(
        self,
        error: Exception,
        agent_name: str,
        fallback_fn: Callable[[], Awaitable[T] | T] | None = None,
        state: dict[str, Any] | None = None,
    ) -> T:
        """Attempt to recover from an error.

        Args:
            error: The error to recover from.
            agent_name: Name of the agent that failed.
            fallback_fn: Function to generate fallback data.
            state: Current workflow state for context.

        Returns:
            Recovery result (either retry success or fallback data).

        Raises:
            RecoveryError: If all recovery attempts fail.
        """
        config = get_agent_fallback_config(agent_name)
        context = RecoveryContext(
            original_error=error,
            agent_name=agent_name,
            max_attempts=config.max_retries,
        )
        self._active_recoveries[agent_name] = context

        logger.warning(
            "recovery_started",
            agent_name=agent_name,
            error_type=type(error).__name__,
            error_message=str(error),
        )

        try:
            # Check if error is recoverable
            if isinstance(error, PortfolioAdvisorError) and not error.recoverable:
                raise RecoveryError(
                    f"Non-recoverable error in {agent_name}: {error}",
                    original_error=error,
                    recovery_attempts=0,
                )

            # Try fallback
            if fallback_fn or config.fallback_fn:
                fn = fallback_fn or config.fallback_fn
                if fn is None:
                    raise RecoveryError(
                        f"No fallback function for {agent_name}",
                        original_error=error,
                    )
                try:
                    result = fn()
                    if asyncio.iscoroutine(result):
                        result = await result

                    context.add_attempt("fallback", success=True)
                    logger.info(
                        "recovery_succeeded",
                        agent_name=agent_name,
                        strategy="fallback",
                        warning=config.warning_message,
                    )
                    return result  # type: ignore[return-value]

                except Exception as fallback_error:
                    context.add_attempt(
                        "fallback", success=False, error=str(fallback_error)
                    )
                    raise RecoveryError(
                        f"Fallback failed for {agent_name}: {fallback_error}",
                        original_error=error,
                        recovery_attempts=context.attempt_count,
                    ) from fallback_error

            # No fallback available
            raise RecoveryError(
                f"No recovery strategy for {agent_name}",
                original_error=error,
                recovery_attempts=context.attempt_count,
            )

        finally:
            del self._active_recoveries[agent_name]

    def get_active_recovery(self, agent_name: str) -> RecoveryContext | None:
        """Get active recovery context for an agent."""
        return self._active_recoveries.get(agent_name)


# Global recovery manager instance
_recovery_manager: ErrorRecoveryManager | None = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get the global recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


def reset_recovery_manager() -> None:
    """Reset the global recovery manager (for testing)."""
    global _recovery_manager
    _recovery_manager = None


# ============================================================================
# Utility Functions
# ============================================================================


async def execute_with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    operation_name: str = "operation",
) -> T:
    """Execute a coroutine with a timeout.

    Args:
        coro: The coroutine to execute.
        timeout_seconds: Maximum time to wait.
        operation_name: Name of operation for error messages.

    Returns:
        The coroutine result.

    Raises:
        WorkflowTimeoutError: If the operation times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError as e:
        raise WorkflowTimeoutError(
            f"{operation_name} timed out after {timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            operation=operation_name,
        ) from e


def classify_error(error: Exception) -> tuple[str, bool]:
    """Classify an error for handling.

    Args:
        error: The error to classify.

    Returns:
        Tuple of (error_category, is_recoverable).
    """
    if isinstance(error, PortfolioAdvisorError):
        return (type(error).__name__, error.recoverable)

    # Map common exceptions to categories
    error_map: dict[type[Exception], tuple[str, bool]] = {
        TimeoutError: ("timeout", True),
        asyncio.TimeoutError: ("timeout", True),
        ConnectionError: ("connection", True),
        ValueError: ("validation", False),
        KeyError: ("state", False),
    }

    for exc_type, (category, recoverable) in error_map.items():
        if isinstance(error, exc_type):
            return (category, recoverable)

    return ("unknown", False)


def is_critical_error(error: Exception) -> bool:
    """Check if an error is critical (should stop the workflow).

    Args:
        error: The error to check.

    Returns:
        True if the error is critical.
    """
    # Check for non-recoverable Portfolio errors
    if isinstance(error, PortfolioAdvisorError):
        return not error.recoverable

    # Check for state errors
    if isinstance(error, (StateError, RecoveryError)):
        return True

    # Check error message for critical indicators
    error_str = str(error).lower()
    critical_keywords = ["critical", "fatal", "unrecoverable", "authentication"]
    return any(keyword in error_str for keyword in critical_keywords)
