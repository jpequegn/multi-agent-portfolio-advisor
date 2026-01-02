"""Circuit breaker pattern implementation for resilience.

This module implements the circuit breaker pattern to prevent cascade failures
when external services (LLM, APIs) are experiencing issues.

Circuit States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed
"""

import asyncio
import functools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ParamSpec, TypeVar, cast

import structlog

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, name: str, recovery_time: float) -> None:
        self.name = name
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit '{name}' is open. Recovery in {recovery_time:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before attempting recovery.
        half_open_max_calls: Max calls allowed in half-open state.
        success_threshold: Successes needed in half-open to close circuit.
        excluded_exceptions: Exceptions that don't count as failures.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    excluded_exceptions: tuple[type[Exception], ...] = ()


# Predefined configurations for common services
LLM_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_max_calls=2,
    success_threshold=2,
)

API_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    half_open_max_calls=3,
    success_threshold=2,
)

MARKET_DATA_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=45.0,
    half_open_max_calls=2,
    success_threshold=2,
)


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation with async support.

    Thread-safe circuit breaker that tracks failures and opens
    the circuit when the failure threshold is reached.

    Example:
        breaker = CircuitBreaker("llm", LLM_CIRCUIT_CONFIG)

        try:
            async with breaker:
                result = await call_llm(prompt)
        except CircuitOpenError:
            # Handle circuit open - use fallback
            pass
    """

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # State tracking
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery."""
        if self._state == CircuitState.OPEN and self._should_attempt_recovery():
            return CircuitState.HALF_OPEN
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN and not self._should_attempt_recovery()

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def time_until_recovery(self) -> float:
        """Get seconds until recovery attempt is allowed."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        remaining = self.config.recovery_timeout - elapsed
        return max(0.0, remaining)

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            logger.info(
                "circuit_state_changed",
                circuit=self.name,
                from_state=old_state.value,
                to_state=new_state.value,
                failure_count=self._failure_count,
            )

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    "circuit_half_open_success",
                    circuit=self.name,
                    success_count=self._success_count,
                    threshold=self.config.success_threshold,
                )
                if self._success_count >= self.config.success_threshold:
                    await self._close()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                if self._failure_count > 0:
                    self._failure_count = 0
                    logger.debug(
                        "circuit_failure_count_reset",
                        circuit=self.name,
                    )

    async def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            logger.debug(
                "circuit_excluded_exception",
                circuit=self.name,
                exception_type=type(exception).__name__,
            )
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            logger.warning(
                "circuit_failure_recorded",
                circuit=self.name,
                failure_count=self._failure_count,
                threshold=self.config.failure_threshold,
                exception_type=type(exception).__name__,
                exception_msg=str(exception)[:200],
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                await self._open()
            elif self._failure_count >= self.config.failure_threshold:
                await self._open()

    async def _open(self) -> None:
        """Open the circuit."""
        await self._transition_to(CircuitState.OPEN)
        self._half_open_calls = 0
        self._success_count = 0
        logger.warning(
            "circuit_opened",
            circuit=self.name,
            failure_count=self._failure_count,
            recovery_timeout=self.config.recovery_timeout,
        )

    async def _close(self) -> None:
        """Close the circuit (normal operation)."""
        await self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        logger.info(
            "circuit_closed",
            circuit=self.name,
        )

    async def _enter_half_open(self) -> None:
        """Enter half-open state for recovery testing."""
        await self._transition_to(CircuitState.HALF_OPEN)
        self._success_count = 0
        self._half_open_calls = 0

    async def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if rejected.

        Raises:
            CircuitOpenError: If circuit is open and rejecting requests.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.CLOSED:
                return True

            if current_state == CircuitState.OPEN:
                raise CircuitOpenError(self.name, self.time_until_recovery)

            # Half-open state
            if self._state != CircuitState.HALF_OPEN:
                await self._enter_half_open()

            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitOpenError(self.name, self.time_until_recovery)

            self._half_open_calls += 1
            logger.debug(
                "circuit_half_open_request",
                circuit=self.name,
                call_number=self._half_open_calls,
                max_calls=self.config.half_open_max_calls,
            )
            return True

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry - check if request is allowed."""
        await self.allow_request()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Async context manager exit - record success or failure."""
        if exc_val is None:
            await self.record_success()
        elif isinstance(exc_val, Exception):
            await self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status for observability."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.config.failure_threshold,
            "time_until_recovery": self.time_until_recovery,
            "recovery_timeout": self.config.recovery_timeout,
        }


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Args:
        name: Unique name for the circuit breaker.
        config: Configuration for the circuit breaker. Only used
            when creating a new circuit breaker.

    Returns:
        Circuit breaker instance.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            config=config or CircuitBreakerConfig(),
        )
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    _circuit_breakers.clear()


def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Name of the circuit breaker.
        config: Configuration for the circuit breaker.
        fallback: Optional fallback function when circuit is open.

    Returns:
        Decorated function.

    Example:
        @circuit_breaker("llm", LLM_CIRCUIT_CONFIG)
        async def call_llm(prompt: str) -> str:
            return await llm.invoke(prompt)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        breaker = get_circuit_breaker(name, config)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                async with breaker:
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
            except CircuitOpenError:
                if fallback is not None:
                    logger.info(
                        "circuit_using_fallback",
                        circuit=name,
                        function=func.__name__,
                    )
                    fallback_result = fallback(*args, **kwargs)
                    if asyncio.iscoroutine(fallback_result):
                        return await fallback_result
                    return fallback_result
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions, we need to run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, T], async_wrapper)
        return cast(Callable[P, T], sync_wrapper)

    return decorator


# Convenience decorators for common services
def llm_circuit_breaker(
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Circuit breaker decorator for LLM calls.

    Example:
        @llm_circuit_breaker()
        async def generate_response(prompt: str) -> str:
            return await llm.invoke(prompt)
    """
    return circuit_breaker("llm", LLM_CIRCUIT_CONFIG, fallback)


def api_circuit_breaker(
    name: str,
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Circuit breaker decorator for external API calls.

    Args:
        name: Unique name for this API's circuit breaker.
        fallback: Optional fallback function.

    Example:
        @api_circuit_breaker("market_data")
        async def fetch_stock_price(symbol: str) -> float:
            return await api.get_price(symbol)
    """
    return circuit_breaker(f"api_{name}", API_CIRCUIT_CONFIG, fallback)


def market_data_circuit_breaker(
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Circuit breaker decorator for market data API calls.

    Example:
        @market_data_circuit_breaker()
        async def fetch_market_data(symbol: str) -> dict:
            return await yfinance.get_data(symbol)
    """
    return circuit_breaker("market_data", MARKET_DATA_CIRCUIT_CONFIG, fallback)
