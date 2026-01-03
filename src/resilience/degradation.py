"""Graceful degradation for service continuity during partial outages.

This module provides:
- Degradation levels for different failure scenarios
- Component status tracking
- Degraded response wrappers
- Degradation monitoring and alerting

Features:
- Automatic degradation based on component health
- User-facing warnings about degraded state
- Fallback strategies per component
- Degradation metrics for observability
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, ParamSpec, TypeVar, cast

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class DegradationLevel(Enum):
    """Degradation levels from best to worst service quality.

    Each level represents a different quality of service based on
    which components are available.
    """

    FULL = "full"  # All features working
    CACHED_DATA = "cached_data"  # Using cached market data
    MOCK_DATA = "mock_data"  # Using mock data
    BASIC_ANALYSIS = "basic_analysis"  # Only basic metrics
    RECOMMENDATIONS_ONLY = "recommendations_only"  # Skip analysis details
    UNAVAILABLE = "unavailable"  # Service unavailable

    def is_degraded(self) -> bool:
        """Check if this level represents degraded service."""
        return self != DegradationLevel.FULL

    def severity(self) -> int:
        """Get severity level (higher = more degraded)."""
        severity_map = {
            DegradationLevel.FULL: 0,
            DegradationLevel.CACHED_DATA: 1,
            DegradationLevel.MOCK_DATA: 2,
            DegradationLevel.BASIC_ANALYSIS: 3,
            DegradationLevel.RECOMMENDATIONS_ONLY: 4,
            DegradationLevel.UNAVAILABLE: 5,
        }
        return severity_map[self]


class ComponentType(Enum):
    """Types of components that can fail."""

    MARKET_DATA_API = "market_data_api"
    NEWS_API = "news_api"
    LLM = "llm"
    REDIS = "redis"
    POSTGRES = "postgres"


class ComponentHealth(Enum):
    """Health status of a component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentStatus:
    """Status of a single component.

    Attributes:
        component: Type of component.
        health: Current health status.
        last_check: Timestamp of last health check.
        last_healthy: Timestamp when last healthy.
        failure_count: Consecutive failure count.
        error_message: Last error message if unhealthy.
    """

    component: ComponentType
    health: ComponentHealth = ComponentHealth.UNKNOWN
    last_check: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_healthy: datetime | None = None
    failure_count: int = 0
    error_message: str | None = None

    def mark_healthy(self) -> None:
        """Mark component as healthy."""
        self.health = ComponentHealth.HEALTHY
        self.last_check = datetime.now(UTC)
        self.last_healthy = self.last_check
        self.failure_count = 0
        self.error_message = None

    def mark_unhealthy(self, error: str | None = None) -> None:
        """Mark component as unhealthy."""
        self.health = ComponentHealth.UNHEALTHY
        self.last_check = datetime.now(UTC)
        self.failure_count += 1
        self.error_message = error

    def mark_degraded(self, error: str | None = None) -> None:
        """Mark component as degraded (partially working)."""
        self.health = ComponentHealth.DEGRADED
        self.last_check = datetime.now(UTC)
        self.error_message = error

    def is_available(self) -> bool:
        """Check if component is available (healthy or degraded)."""
        return self.health in (ComponentHealth.HEALTHY, ComponentHealth.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component.value,
            "health": self.health.value,
            "last_check": self.last_check.isoformat(),
            "last_healthy": self.last_healthy.isoformat() if self.last_healthy else None,
            "failure_count": self.failure_count,
            "error_message": self.error_message,
        }


@dataclass
class DegradedResponse(Generic[T]):
    """Response wrapper that includes degradation information.

    Attributes:
        result: The actual response data.
        degradation_level: Current degradation level.
        warnings: List of warnings about degraded state.
        timestamp: When the response was generated.
        cache_age_seconds: Age of cached data if using cache.
    """

    result: T
    degradation_level: DegradationLevel = DegradationLevel.FULL
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    cache_age_seconds: float | None = None

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def is_degraded(self) -> bool:
        """Check if response is degraded."""
        return self.degradation_level.is_degraded()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result": self.result,
            "degradation_level": self.degradation_level.value,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "cache_age_seconds": self.cache_age_seconds,
        }


@dataclass
class DegradationMetrics:
    """Metrics for degradation monitoring.

    Attributes:
        degradation_events: Count of degradation events by level.
        component_failures: Count of failures by component.
        total_degraded_requests: Total requests served in degraded state.
        total_full_requests: Total requests served at full capacity.
    """

    degradation_events: dict[DegradationLevel, int] = field(
        default_factory=lambda: dict.fromkeys(DegradationLevel, 0)
    )
    component_failures: dict[ComponentType, int] = field(
        default_factory=lambda: dict.fromkeys(ComponentType, 0)
    )
    total_degraded_requests: int = 0
    total_full_requests: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def record_degradation(self, level: DegradationLevel) -> None:
        """Record a degradation event."""
        async with self._lock:
            self.degradation_events[level] += 1
            if level.is_degraded():
                self.total_degraded_requests += 1
            else:
                self.total_full_requests += 1

    async def record_component_failure(self, component: ComponentType) -> None:
        """Record a component failure."""
        async with self._lock:
            self.component_failures[component] += 1

    @property
    def degraded_ratio(self) -> float:
        """Ratio of degraded to total requests."""
        total = self.total_degraded_requests + self.total_full_requests
        if total == 0:
            return 0.0
        return self.total_degraded_requests / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "degradation_events": {k.value: v for k, v in self.degradation_events.items()},
            "component_failures": {k.value: v for k, v in self.component_failures.items()},
            "total_degraded_requests": self.total_degraded_requests,
            "total_full_requests": self.total_full_requests,
            "degraded_ratio": round(self.degraded_ratio, 4),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.degradation_events = dict.fromkeys(DegradationLevel, 0)
        self.component_failures = dict.fromkeys(ComponentType, 0)
        self.total_degraded_requests = 0
        self.total_full_requests = 0


class DegradationStrategy:
    """Strategy for handling component failures.

    Maps component failures to degradation levels and fallback behaviors.
    """

    # Default degradation mapping for component failures
    DEFAULT_DEGRADATION_MAP: dict[ComponentType, DegradationLevel] = {
        ComponentType.MARKET_DATA_API: DegradationLevel.CACHED_DATA,
        ComponentType.NEWS_API: DegradationLevel.CACHED_DATA,
        ComponentType.LLM: DegradationLevel.BASIC_ANALYSIS,
        ComponentType.REDIS: DegradationLevel.FULL,  # Can bypass cache
        ComponentType.POSTGRES: DegradationLevel.RECOMMENDATIONS_ONLY,
    }

    # Warning messages for each degradation scenario
    DEFAULT_WARNINGS: dict[ComponentType, str] = {
        ComponentType.MARKET_DATA_API: "Market data may be outdated",
        ComponentType.NEWS_API: "News analysis unavailable",
        ComponentType.LLM: "Using simplified analysis",
        ComponentType.REDIS: "Responses may be slower (cache bypassed)",
        ComponentType.POSTGRES: "Read-only mode, some features limited",
    }

    def __init__(
        self,
        degradation_map: dict[ComponentType, DegradationLevel] | None = None,
        warnings: dict[ComponentType, str] | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            degradation_map: Custom component to degradation level mapping.
            warnings: Custom warning messages per component.
        """
        self.degradation_map = degradation_map or self.DEFAULT_DEGRADATION_MAP.copy()
        self.warnings = warnings or self.DEFAULT_WARNINGS.copy()

    def get_degradation_level(
        self, failed_components: set[ComponentType]
    ) -> DegradationLevel:
        """Determine degradation level based on failed components.

        Returns the worst (highest severity) degradation level.

        Args:
            failed_components: Set of failed component types.

        Returns:
            Worst degradation level required.
        """
        if not failed_components:
            return DegradationLevel.FULL

        levels = [
            self.degradation_map.get(comp, DegradationLevel.UNAVAILABLE)
            for comp in failed_components
        ]
        return max(levels, key=lambda x: x.severity())

    def get_warnings(self, failed_components: set[ComponentType]) -> list[str]:
        """Get warning messages for failed components.

        Args:
            failed_components: Set of failed component types.

        Returns:
            List of warning messages.
        """
        return [
            self.warnings[comp]
            for comp in failed_components
            if comp in self.warnings
        ]


class DegradationManager:
    """Manager for graceful degradation.

    Coordinates component health tracking, degradation level determination,
    and response wrapping.

    Example:
        manager = DegradationManager()

        # Report component health
        await manager.mark_healthy(ComponentType.MARKET_DATA_API)
        await manager.mark_unhealthy(ComponentType.NEWS_API, "API timeout")

        # Get current degradation level
        level = manager.get_degradation_level()

        # Wrap a response with degradation info
        response = manager.wrap_response(data, cache_age=300)
    """

    def __init__(
        self,
        strategy: DegradationStrategy | None = None,
        alert_callback: Callable[[DegradationLevel, list[str]], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize degradation manager.

        Args:
            strategy: Degradation strategy to use.
            alert_callback: Async callback for alerts on degradation changes.
        """
        self.strategy = strategy or DegradationStrategy()
        self.alert_callback = alert_callback
        self.metrics = DegradationMetrics()
        self._components: dict[ComponentType, ComponentStatus] = {}
        for comp in ComponentType:
            status = ComponentStatus(component=comp)
            status.health = ComponentHealth.HEALTHY  # Start all components as healthy
            self._components[comp] = status
        self._current_level = DegradationLevel.FULL
        self._lock = asyncio.Lock()

    async def mark_healthy(self, component: ComponentType) -> None:
        """Mark a component as healthy.

        Args:
            component: Component type to mark healthy.
        """
        async with self._lock:
            self._components[component].mark_healthy()
            await self._update_degradation_level()

        logger.debug("component_healthy", component=component.value)

    async def mark_unhealthy(
        self, component: ComponentType, error: str | None = None
    ) -> None:
        """Mark a component as unhealthy.

        Args:
            component: Component type to mark unhealthy.
            error: Error message describing the failure.
        """
        async with self._lock:
            self._components[component].mark_unhealthy(error)
            await self.metrics.record_component_failure(component)
            await self._update_degradation_level()

        logger.warning(
            "component_unhealthy",
            component=component.value,
            error=error,
            failure_count=self._components[component].failure_count,
        )

    async def mark_degraded(
        self, component: ComponentType, error: str | None = None
    ) -> None:
        """Mark a component as degraded (partially working).

        Args:
            component: Component type to mark degraded.
            error: Error message describing the issue.
        """
        async with self._lock:
            self._components[component].mark_degraded(error)
            await self._update_degradation_level()

        logger.info("component_degraded", component=component.value, error=error)

    async def _update_degradation_level(self) -> None:
        """Update the current degradation level based on component health."""
        failed = {
            comp
            for comp, status in self._components.items()
            if not status.is_available()
        }

        new_level = self.strategy.get_degradation_level(failed)

        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level

            await self.metrics.record_degradation(new_level)

            logger.info(
                "degradation_level_changed",
                old_level=old_level.value,
                new_level=new_level.value,
                failed_components=[c.value for c in failed],
            )

            # Trigger alert callback if degradation worsened
            if new_level.severity() > old_level.severity() and self.alert_callback:
                warnings = self.strategy.get_warnings(failed)
                await self.alert_callback(new_level, warnings)

    def get_degradation_level(self) -> DegradationLevel:
        """Get current degradation level.

        Returns:
            Current degradation level.
        """
        return self._current_level

    def get_component_status(self, component: ComponentType) -> ComponentStatus:
        """Get status of a specific component.

        Args:
            component: Component type.

        Returns:
            Component status.
        """
        return self._components[component]

    def get_all_component_statuses(self) -> dict[ComponentType, ComponentStatus]:
        """Get status of all components.

        Returns:
            Dictionary of component statuses.
        """
        return self._components.copy()

    def get_failed_components(self) -> set[ComponentType]:
        """Get set of currently failed components.

        Returns:
            Set of failed component types.
        """
        return {
            comp
            for comp, status in self._components.items()
            if not status.is_available()
        }

    def wrap_response(
        self,
        result: T,
        cache_age: float | None = None,
    ) -> DegradedResponse[T]:
        """Wrap a result with degradation information.

        Args:
            result: The result to wrap.
            cache_age: Age of cached data in seconds.

        Returns:
            DegradedResponse with degradation info.
        """
        failed = self.get_failed_components()
        warnings = self.strategy.get_warnings(failed)

        if cache_age is not None and cache_age > 60:
            warnings.append(f"Data from cache ({int(cache_age)} seconds old)")

        return DegradedResponse(
            result=result,
            degradation_level=self._current_level,
            warnings=warnings,
            cache_age_seconds=cache_age,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get degradation metrics.

        Returns:
            Dictionary of metrics.
        """
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset degradation metrics."""
        self.metrics.reset()

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of system health.

        Returns:
            Dictionary with health summary.
        """
        return {
            "degradation_level": self._current_level.value,
            "is_degraded": self._current_level.is_degraded(),
            "components": {
                comp.value: status.to_dict()
                for comp, status in self._components.items()
            },
            "failed_components": [c.value for c in self.get_failed_components()],
            "metrics": self.get_metrics(),
        }


def with_fallback(
    fallback_fn: Callable[P, T | Awaitable[T]],
    on_error: Callable[[Exception], None] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to provide fallback on failure.

    Args:
        fallback_fn: Function to call if primary fails.
        on_error: Optional callback when error occurs.

    Returns:
        Decorator function.

    Example:
        @with_fallback(lambda symbol: {"price": 0.0}, on_error=log_error)
        async def fetch_price(symbol: str) -> dict:
            return await market_api.get_price(symbol)
    """

    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if on_error:
                    on_error(e)

                logger.warning(
                    "fallback_activated",
                    function=fn.__name__,
                    error=str(e),
                )

                fallback_result = fallback_fn(*args, **kwargs)
                if asyncio.iscoroutine(fallback_result):
                    return cast(T, await fallback_result)
                return cast(T, fallback_result)

        return wrapper

    return decorator


class FallbackChain(Generic[T]):
    """Chain of fallback data sources.

    Tries each source in order until one succeeds.

    Example:
        chain = FallbackChain[dict]()
        chain.add("primary", fetch_from_api)
        chain.add("cache", fetch_from_cache)
        chain.add("mock", lambda: {"price": 0.0})

        result, source = await chain.execute()
    """

    def __init__(self) -> None:
        """Initialize fallback chain."""
        self._sources: list[tuple[str, Callable[[], Awaitable[T] | T]]] = []

    def add(self, name: str, source: Callable[[], Awaitable[T] | T]) -> "FallbackChain[T]":
        """Add a fallback source.

        Args:
            name: Name of the source for logging.
            source: Callable that returns data.

        Returns:
            Self for chaining.
        """
        self._sources.append((name, source))
        return self

    async def execute(self) -> tuple[T, str]:
        """Execute the fallback chain.

        Tries each source in order until one succeeds.

        Returns:
            Tuple of (result, source_name).

        Raises:
            RuntimeError: If all sources fail.
        """
        errors: list[tuple[str, Exception]] = []

        for name, source in self._sources:
            try:
                source_result = source()
                if asyncio.iscoroutine(source_result):
                    source_result = await source_result

                logger.debug("fallback_chain_success", source=name)
                return cast(T, source_result), name

            except Exception as e:
                errors.append((name, e))
                logger.debug(
                    "fallback_chain_source_failed",
                    source=name,
                    error=str(e),
                )
                continue

        # All sources failed
        error_summary = "; ".join(f"{name}: {e}" for name, e in errors)
        raise RuntimeError(f"All fallback sources failed: {error_summary}")


# Global degradation manager instance
_degradation_manager: DegradationManager | None = None


def get_degradation_manager() -> DegradationManager | None:
    """Get the global degradation manager instance.

    Returns:
        Degradation manager or None if not initialized.
    """
    return _degradation_manager


def set_degradation_manager(manager: DegradationManager) -> None:
    """Set the global degradation manager instance.

    Args:
        manager: Degradation manager to set as global.
    """
    global _degradation_manager
    _degradation_manager = manager


def reset_degradation_manager() -> None:
    """Reset the global degradation manager (for testing)."""
    global _degradation_manager
    _degradation_manager = None
