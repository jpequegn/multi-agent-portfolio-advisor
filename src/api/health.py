"""Health check endpoints for monitoring and orchestration.

This module provides:
- /health (liveness): Basic check that the service is running
- /health/live (liveness): Alias for Kubernetes compatibility
- /health/ready (readiness): Full check of all dependencies

Features:
- Individual component health checks
- Configurable timeouts
- Latency tracking
- Integration with DegradationManager
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status values."""

    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceStatus(Enum):
    """Overall service status."""

    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"


@dataclass
class ComponentCheck:
    """Result of a component health check.

    Attributes:
        name: Component name.
        status: Health status.
        latency_ms: Check latency in milliseconds.
        error: Error message if unhealthy.
        details: Additional details.
    """

    name: str
    status: HealthStatus
    latency_ms: float
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.error:
            result["error"] = self.error
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class HealthCheckConfig:
    """Configuration for health checks.

    Attributes:
        timeout_seconds: Default timeout for checks.
        postgresql_timeout: Timeout for PostgreSQL check.
        redis_timeout: Timeout for Redis check.
        llm_timeout: Timeout for LLM check.
        api_timeout: Timeout for external API checks.
    """

    timeout_seconds: float = 5.0
    postgresql_timeout: float = 2.0
    redis_timeout: float = 1.0
    llm_timeout: float = 10.0
    api_timeout: float = 5.0


DEFAULT_HEALTH_CONFIG = HealthCheckConfig()


class HealthChecker:
    """Base class for component health checkers."""

    def __init__(self, name: str, timeout: float = 5.0) -> None:
        """Initialize health checker.

        Args:
            name: Component name.
            timeout: Check timeout in seconds.
        """
        self.name = name
        self.timeout = timeout

    async def check(self) -> ComponentCheck:
        """Run health check.

        Returns:
            ComponentCheck result.
        """
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self._do_check(),
                timeout=self.timeout,
            )
            latency_ms = (time.monotonic() - start) * 1000
            return ComponentCheck(
                name=self.name,
                status=result.status,
                latency_ms=latency_ms,
                error=result.error,
                details=result.details,
            )
        except TimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _do_check(self) -> ComponentCheck:
        """Implement the actual health check.

        Returns:
            ComponentCheck result.
        """
        raise NotImplementedError


class PostgreSQLHealthChecker(HealthChecker):
    """Health checker for PostgreSQL database."""

    def __init__(
        self,
        pool: Any | None = None,
        timeout: float = DEFAULT_HEALTH_CONFIG.postgresql_timeout,
    ) -> None:
        """Initialize PostgreSQL health checker.

        Args:
            pool: Database connection pool.
            timeout: Check timeout.
        """
        super().__init__("postgresql", timeout)
        self.pool = pool

    async def _do_check(self) -> ComponentCheck:
        """Check PostgreSQL connectivity."""
        if self.pool is None:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error="No connection pool configured",
            )

        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    return ComponentCheck(
                        name=self.name,
                        status=HealthStatus.OK,
                        latency_ms=0,
                    )
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    error="Unexpected query result",
                )
        except Exception as e:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error=str(e),
            )


class RedisHealthChecker(HealthChecker):
    """Health checker for Redis cache."""

    def __init__(
        self,
        redis: Any | None = None,
        timeout: float = DEFAULT_HEALTH_CONFIG.redis_timeout,
    ) -> None:
        """Initialize Redis health checker.

        Args:
            redis: Redis client.
            timeout: Check timeout.
        """
        super().__init__("redis", timeout)
        self.redis = redis

    async def _do_check(self) -> ComponentCheck:
        """Check Redis connectivity."""
        if self.redis is None:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error="No Redis client configured",
            )

        try:
            result = await self.redis.ping()
            if result:
                info = await self.redis.info("memory")
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=0,
                    details={
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                    },
                )
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error="Ping failed",
            )
        except Exception as e:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error=str(e),
            )


class LLMHealthChecker(HealthChecker):
    """Health checker for LLM service."""

    def __init__(
        self,
        client: Any | None = None,
        timeout: float = DEFAULT_HEALTH_CONFIG.llm_timeout,
    ) -> None:
        """Initialize LLM health checker.

        Args:
            client: LLM client (e.g., Anthropic client).
            timeout: Check timeout.
        """
        super().__init__("llm", timeout)
        self.client = client

    async def _do_check(self) -> ComponentCheck:
        """Check LLM service availability."""
        if self.client is None:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error="No LLM client configured",
            )

        try:
            # Use a minimal API call to check connectivity
            # For Anthropic, we could use the messages API with minimal tokens
            # For now, just verify the client is configured
            if hasattr(self.client, "api_key") and self.client.api_key:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=0,
                    details={"configured": True},
                )
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error="API key not configured",
            )
        except Exception as e:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error=str(e),
            )


class ExternalAPIHealthChecker(HealthChecker):
    """Health checker for external APIs."""

    def __init__(
        self,
        name: str,
        check_fn: Any | None = None,
        timeout: float = DEFAULT_HEALTH_CONFIG.api_timeout,
    ) -> None:
        """Initialize external API health checker.

        Args:
            name: API name (e.g., "market_data_api").
            check_fn: Async function that returns True if healthy.
            timeout: Check timeout.
        """
        super().__init__(name, timeout)
        self.check_fn = check_fn

    async def _do_check(self) -> ComponentCheck:
        """Check external API availability."""
        if self.check_fn is None:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error="No health check function configured",
            )

        try:
            is_healthy = await self.check_fn()
            if is_healthy:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=0,
                )
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error="API returned unhealthy",
            )
        except Exception as e:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=0,
                error=str(e),
            )


class LangfuseHealthChecker(HealthChecker):
    """Health checker for Langfuse observability."""

    def __init__(
        self,
        client: Any | None = None,
        timeout: float = DEFAULT_HEALTH_CONFIG.api_timeout,
    ) -> None:
        """Initialize Langfuse health checker.

        Args:
            client: Langfuse client.
            timeout: Check timeout.
        """
        super().__init__("langfuse", timeout)
        self.client = client

    async def _do_check(self) -> ComponentCheck:
        """Check Langfuse connectivity."""
        if self.client is None:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error="No Langfuse client configured",
                details={"optional": True},
            )

        try:
            # Langfuse is optional, so degraded if not working
            if hasattr(self.client, "flush"):
                await asyncio.to_thread(self.client.flush)
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.OK,
                latency_ms=0,
            )
        except Exception as e:
            return ComponentCheck(
                name=self.name,
                status=HealthStatus.DEGRADED,
                latency_ms=0,
                error=str(e),
                details={"optional": True},
            )


@dataclass
class HealthCheckResult:
    """Result of full health check.

    Attributes:
        status: Overall service status.
        checks: Individual component checks.
        timestamp: When the check was performed.
        version: Service version.
    """

    status: ServiceStatus
    checks: dict[str, dict[str, Any]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }


class HealthService:
    """Service for running health checks.

    Coordinates all component health checkers and provides
    endpoints for liveness and readiness probes.

    Example:
        service = HealthService()
        service.register_checker(PostgreSQLHealthChecker(pool=db_pool))
        service.register_checker(RedisHealthChecker(redis=redis_client))

        # Basic liveness check
        result = await service.liveness()

        # Full readiness check
        result = await service.readiness()
    """

    def __init__(self, version: str = "1.0.0") -> None:
        """Initialize health service.

        Args:
            version: Service version to include in responses.
        """
        self.version = version
        self._checkers: list[HealthChecker] = []

    def register_checker(self, checker: HealthChecker) -> None:
        """Register a health checker.

        Args:
            checker: Health checker to register.
        """
        self._checkers.append(checker)
        logger.debug("health_checker_registered", name=checker.name)

    def unregister_checker(self, name: str) -> bool:
        """Unregister a health checker by name.

        Args:
            name: Name of checker to unregister.

        Returns:
            True if checker was found and removed.
        """
        for i, checker in enumerate(self._checkers):
            if checker.name == name:
                self._checkers.pop(i)
                logger.debug("health_checker_unregistered", name=name)
                return True
        return False

    async def liveness(self) -> dict[str, Any]:
        """Basic liveness check.

        Fast check that the service is running.
        Does not check dependencies.

        Returns:
            Simple status response.
        """
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def readiness(self) -> HealthCheckResult:
        """Full readiness check.

        Checks all registered components in parallel.

        Returns:
            Comprehensive health check result.
        """
        if not self._checkers:
            return HealthCheckResult(
                status=ServiceStatus.READY,
                checks={},
                version=self.version,
            )

        # Run all checks in parallel
        check_tasks = [checker.check() for checker in self._checkers]
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        checks: dict[str, dict[str, Any]] = {}
        all_ok = True
        any_unhealthy = False

        for result in results:
            if isinstance(result, BaseException):
                # Handle unexpected exceptions
                checks["unknown"] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(result),
                }
                any_unhealthy = True
            elif isinstance(result, ComponentCheck):
                checks[result.name] = result.to_dict()
                if result.status == HealthStatus.UNHEALTHY:
                    any_unhealthy = True
                    all_ok = False
                elif result.status == HealthStatus.DEGRADED:
                    all_ok = False

        # Determine overall status
        if any_unhealthy:
            status = ServiceStatus.NOT_READY
        elif all_ok:
            status = ServiceStatus.READY
        else:
            status = ServiceStatus.DEGRADED

        logger.info(
            "health_check_completed",
            status=status.value,
            checks_count=len(checks),
        )

        return HealthCheckResult(
            status=status,
            checks=checks,
            version=self.version,
        )

    async def check_component(self, name: str) -> ComponentCheck | None:
        """Check a specific component.

        Args:
            name: Component name.

        Returns:
            ComponentCheck or None if not found.
        """
        for checker in self._checkers:
            if checker.name == name:
                return await checker.check()
        return None


# Global health service instance
_health_service: HealthService | None = None


def get_health_service() -> HealthService | None:
    """Get the global health service instance.

    Returns:
        Health service or None if not initialized.
    """
    return _health_service


def set_health_service(service: HealthService) -> None:
    """Set the global health service instance.

    Args:
        service: Health service to set as global.
    """
    global _health_service
    _health_service = service


def reset_health_service() -> None:
    """Reset the global health service (for testing)."""
    global _health_service
    _health_service = None


def create_health_service(
    version: str = "1.0.0",
    postgresql_pool: Any | None = None,
    redis_client: Any | None = None,
    llm_client: Any | None = None,
    langfuse_client: Any | None = None,
    config: HealthCheckConfig | None = None,
) -> HealthService:
    """Create a configured health service.

    Factory function to create a health service with common checkers.

    Args:
        version: Service version.
        postgresql_pool: Database connection pool.
        redis_client: Redis client.
        llm_client: LLM client.
        langfuse_client: Langfuse client.
        config: Health check configuration.

    Returns:
        Configured HealthService.
    """
    config = config or DEFAULT_HEALTH_CONFIG
    service = HealthService(version=version)

    if postgresql_pool is not None:
        service.register_checker(
            PostgreSQLHealthChecker(pool=postgresql_pool, timeout=config.postgresql_timeout)
        )

    if redis_client is not None:
        service.register_checker(
            RedisHealthChecker(redis=redis_client, timeout=config.redis_timeout)
        )

    if llm_client is not None:
        service.register_checker(
            LLMHealthChecker(client=llm_client, timeout=config.llm_timeout)
        )

    if langfuse_client is not None:
        service.register_checker(
            LangfuseHealthChecker(client=langfuse_client, timeout=config.api_timeout)
        )

    return service
