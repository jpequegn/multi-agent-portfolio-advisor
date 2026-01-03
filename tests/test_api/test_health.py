"""Tests for health check module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.health import (
    ComponentCheck,
    ExternalAPIHealthChecker,
    HealthCheckConfig,
    HealthChecker,
    HealthCheckResult,
    HealthService,
    HealthStatus,
    LangfuseHealthChecker,
    LLMHealthChecker,
    PostgreSQLHealthChecker,
    RedisHealthChecker,
    ServiceStatus,
    create_health_service,
    get_health_service,
    reset_health_service,
    set_health_service,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_values(self) -> None:
        """Should have expected values."""
        assert HealthStatus.OK.value == "ok"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestServiceStatus:
    """Tests for ServiceStatus enum."""

    def test_values(self) -> None:
        """Should have expected values."""
        assert ServiceStatus.READY.value == "ready"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.NOT_READY.value == "not_ready"


class TestComponentCheck:
    """Tests for ComponentCheck."""

    def test_to_dict_basic(self) -> None:
        """Should convert basic check to dict."""
        check = ComponentCheck(
            name="test",
            status=HealthStatus.OK,
            latency_ms=5.123,
        )

        data = check.to_dict()
        assert data["status"] == "ok"
        assert data["latency_ms"] == 5.12
        assert "error" not in data
        assert "details" not in data

    def test_to_dict_with_error(self) -> None:
        """Should include error when present."""
        check = ComponentCheck(
            name="test",
            status=HealthStatus.UNHEALTHY,
            latency_ms=10.0,
            error="Connection refused",
        )

        data = check.to_dict()
        assert data["error"] == "Connection refused"

    def test_to_dict_with_details(self) -> None:
        """Should include details when present."""
        check = ComponentCheck(
            name="test",
            status=HealthStatus.OK,
            latency_ms=5.0,
            details={"version": "1.0"},
        )

        data = check.to_dict()
        assert data["details"] == {"version": "1.0"}


class TestHealthCheckConfig:
    """Tests for HealthCheckConfig."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = HealthCheckConfig()
        assert config.timeout_seconds == 5.0
        assert config.postgresql_timeout == 2.0
        assert config.redis_timeout == 1.0
        assert config.llm_timeout == 10.0
        assert config.api_timeout == 5.0

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = HealthCheckConfig(
            timeout_seconds=10.0,
            postgresql_timeout=5.0,
        )
        assert config.timeout_seconds == 10.0
        assert config.postgresql_timeout == 5.0


class TestHealthChecker:
    """Tests for base HealthChecker."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Should handle timeout."""

        class SlowChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                await asyncio.sleep(10)  # Will timeout
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=0,
                )

        checker = SlowChecker("slow", timeout=0.1)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_exception_handling(self) -> None:
        """Should handle exceptions."""

        class FailingChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                raise RuntimeError("Check failed")

        checker = FailingChecker("failing", timeout=1.0)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_latency_tracking(self) -> None:
        """Should track latency."""

        class DelayedChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                await asyncio.sleep(0.05)
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=0,
                )

        checker = DelayedChecker("delayed", timeout=1.0)
        result = await checker.check()

        assert result.latency_ms >= 50  # At least 50ms


class TestPostgreSQLHealthChecker:
    """Tests for PostgreSQLHealthChecker."""

    @pytest.mark.asyncio
    async def test_no_pool(self) -> None:
        """Should return unhealthy when no pool."""
        checker = PostgreSQLHealthChecker(pool=None)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "No connection pool" in (result.error or "")

    @pytest.mark.asyncio
    async def test_successful_check(self) -> None:
        """Should return OK on successful query."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        checker = PostgreSQLHealthChecker(pool=mock_pool)
        result = await checker.check()

        assert result.status == HealthStatus.OK

    @pytest.mark.asyncio
    async def test_query_failure(self) -> None:
        """Should return unhealthy on query failure."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=Exception("Connection lost"))

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

        checker = PostgreSQLHealthChecker(pool=mock_pool)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY


class TestRedisHealthChecker:
    """Tests for RedisHealthChecker."""

    @pytest.mark.asyncio
    async def test_no_client(self) -> None:
        """Should return unhealthy when no client."""
        checker = RedisHealthChecker(redis=None)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "No Redis client" in (result.error or "")

    @pytest.mark.asyncio
    async def test_successful_check(self) -> None:
        """Should return OK on successful ping."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.info = AsyncMock(return_value={"used_memory_human": "1M"})

        checker = RedisHealthChecker(redis=mock_redis)
        result = await checker.check()

        assert result.status == HealthStatus.OK
        assert result.details.get("used_memory_human") == "1M"

    @pytest.mark.asyncio
    async def test_ping_failure(self) -> None:
        """Should return unhealthy on ping failure."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=False)

        checker = RedisHealthChecker(redis=mock_redis)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY


class TestLLMHealthChecker:
    """Tests for LLMHealthChecker."""

    @pytest.mark.asyncio
    async def test_no_client(self) -> None:
        """Should return unhealthy when no client."""
        checker = LLMHealthChecker(client=None)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "No LLM client" in (result.error or "")

    @pytest.mark.asyncio
    async def test_configured_client(self) -> None:
        """Should return OK when client has API key."""
        mock_client = MagicMock()
        mock_client.api_key = "test-key"

        checker = LLMHealthChecker(client=mock_client)
        result = await checker.check()

        assert result.status == HealthStatus.OK

    @pytest.mark.asyncio
    async def test_no_api_key(self) -> None:
        """Should return unhealthy when no API key."""
        mock_client = MagicMock()
        mock_client.api_key = None

        checker = LLMHealthChecker(client=mock_client)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY


class TestExternalAPIHealthChecker:
    """Tests for ExternalAPIHealthChecker."""

    @pytest.mark.asyncio
    async def test_no_check_fn(self) -> None:
        """Should return degraded when no check function."""
        checker = ExternalAPIHealthChecker(name="api", check_fn=None)
        result = await checker.check()

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_healthy_api(self) -> None:
        """Should return OK when API is healthy."""
        check_fn = AsyncMock(return_value=True)
        checker = ExternalAPIHealthChecker(name="api", check_fn=check_fn)
        result = await checker.check()

        assert result.status == HealthStatus.OK

    @pytest.mark.asyncio
    async def test_unhealthy_api(self) -> None:
        """Should return degraded when API returns false."""
        check_fn = AsyncMock(return_value=False)
        checker = ExternalAPIHealthChecker(name="api", check_fn=check_fn)
        result = await checker.check()

        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_api_exception(self) -> None:
        """Should return unhealthy on exception."""
        check_fn = AsyncMock(side_effect=Exception("API error"))
        checker = ExternalAPIHealthChecker(name="api", check_fn=check_fn)
        result = await checker.check()

        assert result.status == HealthStatus.UNHEALTHY


class TestLangfuseHealthChecker:
    """Tests for LangfuseHealthChecker."""

    @pytest.mark.asyncio
    async def test_no_client(self) -> None:
        """Should return degraded when no client (optional)."""
        checker = LangfuseHealthChecker(client=None)
        result = await checker.check()

        assert result.status == HealthStatus.DEGRADED
        assert result.details.get("optional") is True

    @pytest.mark.asyncio
    async def test_successful_flush(self) -> None:
        """Should return OK on successful flush."""
        mock_client = MagicMock()
        mock_client.flush = MagicMock()

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = None
            checker = LangfuseHealthChecker(client=mock_client)
            result = await checker.check()

        assert result.status == HealthStatus.OK


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_to_dict(self) -> None:
        """Should convert to dict."""
        result = HealthCheckResult(
            status=ServiceStatus.READY,
            checks={"db": {"status": "ok"}},
            version="1.0.0",
        )

        data = result.to_dict()
        assert data["status"] == "ready"
        assert data["checks"] == {"db": {"status": "ok"}}
        assert data["version"] == "1.0.0"
        assert "timestamp" in data


class TestHealthService:
    """Tests for HealthService."""

    @pytest.fixture
    def service(self) -> HealthService:
        """Create fresh health service."""
        return HealthService(version="1.0.0")

    @pytest.mark.asyncio
    async def test_liveness(self, service: HealthService) -> None:
        """Should return basic status."""
        result = await service.liveness()

        assert result["status"] == "ok"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_readiness_no_checkers(self, service: HealthService) -> None:
        """Should return ready when no checkers."""
        result = await service.readiness()

        assert result.status == ServiceStatus.READY
        assert result.checks == {}

    @pytest.mark.asyncio
    async def test_register_checker(self, service: HealthService) -> None:
        """Should register and run checker."""

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=1.0,
                )

        service.register_checker(OKChecker("test"))
        result = await service.readiness()

        assert "test" in result.checks
        assert result.checks["test"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unregister_checker(self, service: HealthService) -> None:
        """Should unregister checker."""

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=1.0,
                )

        service.register_checker(OKChecker("test"))
        assert service.unregister_checker("test") is True
        assert service.unregister_checker("nonexistent") is False

        result = await service.readiness()
        assert "test" not in result.checks

    @pytest.mark.asyncio
    async def test_all_healthy_returns_ready(self, service: HealthService) -> None:
        """Should return READY when all healthy."""

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=1.0,
                )

        service.register_checker(OKChecker("db"))
        service.register_checker(OKChecker("cache"))

        result = await service.readiness()
        assert result.status == ServiceStatus.READY

    @pytest.mark.asyncio
    async def test_degraded_returns_degraded(self, service: HealthService) -> None:
        """Should return DEGRADED when some degraded."""

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=1.0,
                )

        class DegradedChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    latency_ms=1.0,
                )

        service.register_checker(OKChecker("db"))
        service.register_checker(DegradedChecker("cache"))

        result = await service.readiness()
        assert result.status == ServiceStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_returns_not_ready(self, service: HealthService) -> None:
        """Should return NOT_READY when any unhealthy."""

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=1.0,
                )

        class UnhealthyChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=1.0,
                    error="DB down",
                )

        service.register_checker(OKChecker("cache"))
        service.register_checker(UnhealthyChecker("db"))

        result = await service.readiness()
        assert result.status == ServiceStatus.NOT_READY

    @pytest.mark.asyncio
    async def test_check_component(self, service: HealthService) -> None:
        """Should check specific component."""

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=1.0,
                )

        service.register_checker(OKChecker("db"))

        result = await service.check_component("db")
        assert result is not None
        assert result.status == HealthStatus.OK

        result = await service.check_component("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_parallel_checks(self, service: HealthService) -> None:
        """Should run checks in parallel."""

        class SlowChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                await asyncio.sleep(0.1)
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=100,
                )

        service.register_checker(SlowChecker("check1"))
        service.register_checker(SlowChecker("check2"))
        service.register_checker(SlowChecker("check3"))

        import time

        start = time.monotonic()
        result = await service.readiness()
        elapsed = time.monotonic() - start

        # Should take ~100ms (parallel), not ~300ms (sequential)
        assert elapsed < 0.2
        assert len(result.checks) == 3


class TestGlobalService:
    """Tests for global service functions."""

    def test_initial_state(self) -> None:
        """Should be None initially."""
        reset_health_service()
        assert get_health_service() is None

    def test_set_and_get(self) -> None:
        """Should set and get service."""
        reset_health_service()
        service = HealthService()
        set_health_service(service)

        assert get_health_service() is service

    def test_reset(self) -> None:
        """Should reset service."""
        service = HealthService()
        set_health_service(service)
        reset_health_service()

        assert get_health_service() is None


class TestCreateHealthService:
    """Tests for create_health_service factory."""

    def test_empty_service(self) -> None:
        """Should create service with no checkers."""
        service = create_health_service()
        assert len(service._checkers) == 0

    def test_with_postgresql(self) -> None:
        """Should add PostgreSQL checker."""
        mock_pool = MagicMock()
        service = create_health_service(postgresql_pool=mock_pool)

        checker_names = [c.name for c in service._checkers]
        assert "postgresql" in checker_names

    def test_with_redis(self) -> None:
        """Should add Redis checker."""
        mock_redis = MagicMock()
        service = create_health_service(redis_client=mock_redis)

        checker_names = [c.name for c in service._checkers]
        assert "redis" in checker_names

    def test_with_llm(self) -> None:
        """Should add LLM checker."""
        mock_llm = MagicMock()
        service = create_health_service(llm_client=mock_llm)

        checker_names = [c.name for c in service._checkers]
        assert "llm" in checker_names

    def test_with_langfuse(self) -> None:
        """Should add Langfuse checker."""
        mock_langfuse = MagicMock()
        service = create_health_service(langfuse_client=mock_langfuse)

        checker_names = [c.name for c in service._checkers]
        assert "langfuse" in checker_names

    def test_with_custom_config(self) -> None:
        """Should use custom config."""
        config = HealthCheckConfig(postgresql_timeout=10.0)
        mock_pool = MagicMock()
        service = create_health_service(
            postgresql_pool=mock_pool,
            config=config,
        )

        pg_checker = next(c for c in service._checkers if c.name == "postgresql")
        assert pg_checker.timeout == 10.0

    def test_with_version(self) -> None:
        """Should set version."""
        service = create_health_service(version="2.0.0")
        assert service.version == "2.0.0"


class TestIntegration:
    """Integration tests for health checks."""

    @pytest.mark.asyncio
    async def test_full_health_check_flow(self) -> None:
        """Test complete health check flow."""
        # Create mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.info = AsyncMock(return_value={"used_memory_human": "1M"})

        mock_llm = MagicMock()
        mock_llm.api_key = "test-key"

        # Create service with multiple checkers
        service = create_health_service(
            version="1.0.0",
            redis_client=mock_redis,
            llm_client=mock_llm,
        )

        # Run liveness check
        liveness = await service.liveness()
        assert liveness["status"] == "ok"

        # Run readiness check
        readiness = await service.readiness()
        assert readiness.status == ServiceStatus.READY
        assert "redis" in readiness.checks
        assert "llm" in readiness.checks

    @pytest.mark.asyncio
    async def test_degraded_service_detection(self) -> None:
        """Test that degraded services are detected."""
        # Create service with failing checker
        service = HealthService()

        class FailingChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    error="Service down",
                )

        class OKChecker(HealthChecker):
            async def _do_check(self) -> ComponentCheck:
                return ComponentCheck(
                    name=self.name,
                    status=HealthStatus.OK,
                    latency_ms=0,
                )

        service.register_checker(OKChecker("redis"))
        service.register_checker(FailingChecker("postgresql"))

        result = await service.readiness()
        assert result.status == ServiceStatus.NOT_READY
        assert result.checks["postgresql"]["status"] == "unhealthy"
        assert result.checks["redis"]["status"] == "ok"
