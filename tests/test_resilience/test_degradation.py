"""Tests for graceful degradation module."""

import asyncio

import pytest

from src.resilience.degradation import (
    ComponentHealth,
    ComponentStatus,
    ComponentType,
    DegradationLevel,
    DegradationManager,
    DegradationMetrics,
    DegradationStrategy,
    DegradedResponse,
    FallbackChain,
    get_degradation_manager,
    reset_degradation_manager,
    set_degradation_manager,
    with_fallback,
)


class TestDegradationLevel:
    """Tests for DegradationLevel enum."""

    def test_full_is_not_degraded(self) -> None:
        """FULL level should not be degraded."""
        assert not DegradationLevel.FULL.is_degraded()

    def test_other_levels_are_degraded(self) -> None:
        """All non-FULL levels should be degraded."""
        assert DegradationLevel.CACHED_DATA.is_degraded()
        assert DegradationLevel.MOCK_DATA.is_degraded()
        assert DegradationLevel.BASIC_ANALYSIS.is_degraded()
        assert DegradationLevel.RECOMMENDATIONS_ONLY.is_degraded()
        assert DegradationLevel.UNAVAILABLE.is_degraded()

    def test_severity_ordering(self) -> None:
        """Severity should increase with worse degradation."""
        assert DegradationLevel.FULL.severity() < DegradationLevel.CACHED_DATA.severity()
        assert DegradationLevel.CACHED_DATA.severity() < DegradationLevel.MOCK_DATA.severity()
        assert DegradationLevel.MOCK_DATA.severity() < DegradationLevel.BASIC_ANALYSIS.severity()
        assert (
            DegradationLevel.BASIC_ANALYSIS.severity()
            < DegradationLevel.RECOMMENDATIONS_ONLY.severity()
        )
        assert (
            DegradationLevel.RECOMMENDATIONS_ONLY.severity()
            < DegradationLevel.UNAVAILABLE.severity()
        )


class TestComponentStatus:
    """Tests for ComponentStatus."""

    def test_initial_state(self) -> None:
        """New status should have unknown health."""
        status = ComponentStatus(component=ComponentType.MARKET_DATA_API)
        assert status.health == ComponentHealth.UNKNOWN
        assert status.failure_count == 0
        assert status.error_message is None

    def test_mark_healthy(self) -> None:
        """Marking healthy should update status."""
        status = ComponentStatus(component=ComponentType.MARKET_DATA_API)
        status.mark_unhealthy("Error")
        status.mark_healthy()

        assert status.health == ComponentHealth.HEALTHY
        assert status.failure_count == 0
        assert status.error_message is None
        assert status.last_healthy is not None

    def test_mark_unhealthy(self) -> None:
        """Marking unhealthy should track failures."""
        status = ComponentStatus(component=ComponentType.LLM)
        status.mark_unhealthy("Connection timeout")
        status.mark_unhealthy("Connection refused")

        assert status.health == ComponentHealth.UNHEALTHY
        assert status.failure_count == 2
        assert status.error_message == "Connection refused"

    def test_mark_degraded(self) -> None:
        """Marking degraded should set appropriate state."""
        status = ComponentStatus(component=ComponentType.REDIS)
        status.mark_degraded("High latency")

        assert status.health == ComponentHealth.DEGRADED
        assert status.error_message == "High latency"

    def test_is_available(self) -> None:
        """Available should be true for healthy or degraded."""
        status = ComponentStatus(component=ComponentType.POSTGRES)

        status.mark_healthy()
        assert status.is_available()

        status.mark_degraded()
        assert status.is_available()

        status.mark_unhealthy()
        assert not status.is_available()

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        status = ComponentStatus(component=ComponentType.NEWS_API)
        status.mark_healthy()

        data = status.to_dict()
        assert data["component"] == "news_api"
        assert data["health"] == "healthy"
        assert "last_check" in data
        assert data["failure_count"] == 0


class TestDegradedResponse:
    """Tests for DegradedResponse."""

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        response: DegradedResponse[str] = DegradedResponse(result="data")
        assert response.result == "data"
        assert response.degradation_level == DegradationLevel.FULL
        assert response.warnings == []
        assert response.cache_age_seconds is None
        assert not response.is_degraded()

    def test_degraded_response(self) -> None:
        """Should track degradation state."""
        response: DegradedResponse[dict[str, int]] = DegradedResponse(
            result={"value": 42},
            degradation_level=DegradationLevel.CACHED_DATA,
            warnings=["Using cached data"],
            cache_age_seconds=120.5,
        )

        assert response.is_degraded()
        assert len(response.warnings) == 1
        assert response.cache_age_seconds == 120.5

    def test_add_warning(self) -> None:
        """Should add warnings."""
        response: DegradedResponse[str] = DegradedResponse(result="data")
        response.add_warning("Warning 1")
        response.add_warning("Warning 2")

        assert len(response.warnings) == 2
        assert "Warning 1" in response.warnings

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        response: DegradedResponse[str] = DegradedResponse(
            result="test",
            degradation_level=DegradationLevel.MOCK_DATA,
        )

        data = response.to_dict()
        assert data["result"] == "test"
        assert data["degradation_level"] == "mock_data"
        assert "timestamp" in data


class TestDegradationMetrics:
    """Tests for DegradationMetrics."""

    @pytest.fixture
    def metrics(self) -> DegradationMetrics:
        """Create fresh metrics."""
        return DegradationMetrics()

    @pytest.mark.asyncio
    async def test_record_degradation(self, metrics: DegradationMetrics) -> None:
        """Should record degradation events."""
        await metrics.record_degradation(DegradationLevel.FULL)
        await metrics.record_degradation(DegradationLevel.CACHED_DATA)
        await metrics.record_degradation(DegradationLevel.CACHED_DATA)

        assert metrics.degradation_events[DegradationLevel.FULL] == 1
        assert metrics.degradation_events[DegradationLevel.CACHED_DATA] == 2
        assert metrics.total_full_requests == 1
        assert metrics.total_degraded_requests == 2

    @pytest.mark.asyncio
    async def test_record_component_failure(self, metrics: DegradationMetrics) -> None:
        """Should record component failures."""
        await metrics.record_component_failure(ComponentType.LLM)
        await metrics.record_component_failure(ComponentType.LLM)
        await metrics.record_component_failure(ComponentType.REDIS)

        assert metrics.component_failures[ComponentType.LLM] == 2
        assert metrics.component_failures[ComponentType.REDIS] == 1

    @pytest.mark.asyncio
    async def test_degraded_ratio(self, metrics: DegradationMetrics) -> None:
        """Should calculate degraded ratio."""
        assert metrics.degraded_ratio == 0.0

        await metrics.record_degradation(DegradationLevel.FULL)
        await metrics.record_degradation(DegradationLevel.FULL)
        await metrics.record_degradation(DegradationLevel.CACHED_DATA)
        await metrics.record_degradation(DegradationLevel.MOCK_DATA)

        assert metrics.degraded_ratio == 0.5  # 2 degraded out of 4

    def test_reset(self, metrics: DegradationMetrics) -> None:
        """Should reset all metrics."""
        metrics.total_degraded_requests = 100
        metrics.total_full_requests = 200
        metrics.reset()

        assert metrics.total_degraded_requests == 0
        assert metrics.total_full_requests == 0
        assert all(v == 0 for v in metrics.degradation_events.values())

    @pytest.mark.asyncio
    async def test_to_dict(self, metrics: DegradationMetrics) -> None:
        """Should convert to dictionary."""
        await metrics.record_degradation(DegradationLevel.BASIC_ANALYSIS)

        data = metrics.to_dict()
        assert "degradation_events" in data
        assert "component_failures" in data
        assert "degraded_ratio" in data


class TestDegradationStrategy:
    """Tests for DegradationStrategy."""

    def test_default_strategy(self) -> None:
        """Should use default mappings."""
        strategy = DegradationStrategy()

        # Market data failure -> cached data
        level = strategy.get_degradation_level({ComponentType.MARKET_DATA_API})
        assert level == DegradationLevel.CACHED_DATA

        # LLM failure -> basic analysis
        level = strategy.get_degradation_level({ComponentType.LLM})
        assert level == DegradationLevel.BASIC_ANALYSIS

    def test_no_failures_returns_full(self) -> None:
        """Should return FULL when no components failed."""
        strategy = DegradationStrategy()
        level = strategy.get_degradation_level(set())
        assert level == DegradationLevel.FULL

    def test_multiple_failures_returns_worst(self) -> None:
        """Should return worst degradation level."""
        strategy = DegradationStrategy()

        # Market data (cached) + LLM (basic analysis) -> basic analysis (worse)
        level = strategy.get_degradation_level(
            {ComponentType.MARKET_DATA_API, ComponentType.LLM}
        )
        assert level == DegradationLevel.BASIC_ANALYSIS

    def test_custom_degradation_map(self) -> None:
        """Should use custom degradation mapping."""
        custom_map = {
            ComponentType.REDIS: DegradationLevel.UNAVAILABLE,
        }
        strategy = DegradationStrategy(degradation_map=custom_map)

        level = strategy.get_degradation_level({ComponentType.REDIS})
        assert level == DegradationLevel.UNAVAILABLE

    def test_get_warnings(self) -> None:
        """Should return warnings for failed components."""
        strategy = DegradationStrategy()

        warnings = strategy.get_warnings({ComponentType.MARKET_DATA_API, ComponentType.LLM})
        assert len(warnings) == 2
        assert any("market data" in w.lower() for w in warnings)
        assert any("analysis" in w.lower() for w in warnings)

    def test_get_warnings_empty_for_no_failures(self) -> None:
        """Should return empty list when no failures."""
        strategy = DegradationStrategy()
        warnings = strategy.get_warnings(set())
        assert warnings == []


class TestDegradationManager:
    """Tests for DegradationManager."""

    @pytest.fixture
    def manager(self) -> DegradationManager:
        """Create fresh manager."""
        return DegradationManager()

    @pytest.mark.asyncio
    async def test_initial_state(self, manager: DegradationManager) -> None:
        """Should start at FULL degradation."""
        assert manager.get_degradation_level() == DegradationLevel.FULL

    @pytest.mark.asyncio
    async def test_mark_healthy(self, manager: DegradationManager) -> None:
        """Should mark component healthy."""
        await manager.mark_healthy(ComponentType.MARKET_DATA_API)

        status = manager.get_component_status(ComponentType.MARKET_DATA_API)
        assert status.health == ComponentHealth.HEALTHY
        assert manager.get_degradation_level() == DegradationLevel.FULL

    @pytest.mark.asyncio
    async def test_mark_unhealthy_degrades(self, manager: DegradationManager) -> None:
        """Should degrade when component fails."""
        await manager.mark_unhealthy(ComponentType.LLM, "Rate limited")

        assert manager.get_degradation_level() == DegradationLevel.BASIC_ANALYSIS
        status = manager.get_component_status(ComponentType.LLM)
        assert status.failure_count == 1

    @pytest.mark.asyncio
    async def test_mark_degraded(self, manager: DegradationManager) -> None:
        """Should handle degraded component."""
        await manager.mark_degraded(ComponentType.REDIS, "High latency")

        status = manager.get_component_status(ComponentType.REDIS)
        assert status.health == ComponentHealth.DEGRADED
        # Degraded is still available, so no level change
        assert manager.get_degradation_level() == DegradationLevel.FULL

    @pytest.mark.asyncio
    async def test_recovery(self, manager: DegradationManager) -> None:
        """Should recover when component becomes healthy."""
        await manager.mark_unhealthy(ComponentType.NEWS_API)
        assert manager.get_degradation_level().is_degraded()

        await manager.mark_healthy(ComponentType.NEWS_API)
        assert manager.get_degradation_level() == DegradationLevel.FULL

    @pytest.mark.asyncio
    async def test_get_failed_components(self, manager: DegradationManager) -> None:
        """Should track failed components."""
        assert manager.get_failed_components() == set()

        await manager.mark_unhealthy(ComponentType.LLM)
        await manager.mark_unhealthy(ComponentType.REDIS)

        failed = manager.get_failed_components()
        assert ComponentType.LLM in failed
        assert ComponentType.REDIS in failed

    @pytest.mark.asyncio
    async def test_wrap_response(self, manager: DegradationManager) -> None:
        """Should wrap response with degradation info."""
        await manager.mark_unhealthy(ComponentType.MARKET_DATA_API)

        response = manager.wrap_response({"price": 100.0}, cache_age=120)

        assert response.result == {"price": 100.0}
        assert response.degradation_level == DegradationLevel.CACHED_DATA
        assert len(response.warnings) >= 1
        assert response.cache_age_seconds == 120

    @pytest.mark.asyncio
    async def test_wrap_response_adds_cache_warning(
        self, manager: DegradationManager
    ) -> None:
        """Should add warning for old cache data."""
        response = manager.wrap_response("data", cache_age=300)

        assert any("cache" in w.lower() for w in response.warnings)

    @pytest.mark.asyncio
    async def test_alert_callback(self) -> None:
        """Should trigger alert on degradation."""
        alerts: list[tuple[DegradationLevel, list[str]]] = []

        async def alert_handler(
            level: DegradationLevel, warnings: list[str]
        ) -> None:
            alerts.append((level, warnings))

        manager = DegradationManager(alert_callback=alert_handler)
        await manager.mark_unhealthy(ComponentType.LLM)

        assert len(alerts) == 1
        assert alerts[0][0] == DegradationLevel.BASIC_ANALYSIS

    @pytest.mark.asyncio
    async def test_no_alert_on_recovery(self) -> None:
        """Should not alert when recovering."""
        alerts: list[tuple[DegradationLevel, list[str]]] = []

        async def alert_handler(
            level: DegradationLevel, warnings: list[str]
        ) -> None:
            alerts.append((level, warnings))

        manager = DegradationManager(alert_callback=alert_handler)
        await manager.mark_unhealthy(ComponentType.LLM)
        alerts.clear()

        await manager.mark_healthy(ComponentType.LLM)
        assert len(alerts) == 0  # No alert on recovery

    @pytest.mark.asyncio
    async def test_get_health_summary(self, manager: DegradationManager) -> None:
        """Should return health summary."""
        await manager.mark_unhealthy(ComponentType.NEWS_API, "Timeout")

        summary = manager.get_health_summary()
        assert summary["degradation_level"] == "cached_data"
        assert summary["is_degraded"] is True
        assert "news_api" in summary["failed_components"]
        assert "components" in summary
        assert "metrics" in summary

    @pytest.mark.asyncio
    async def test_get_all_component_statuses(
        self, manager: DegradationManager
    ) -> None:
        """Should return all component statuses."""
        statuses = manager.get_all_component_statuses()
        assert len(statuses) == len(ComponentType)
        assert all(isinstance(s, ComponentStatus) for s in statuses.values())

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, manager: DegradationManager) -> None:
        """Should track metrics."""
        await manager.mark_unhealthy(ComponentType.LLM)
        await manager.mark_unhealthy(ComponentType.LLM)

        metrics = manager.get_metrics()
        assert metrics["component_failures"]["llm"] == 2

    def test_reset_metrics(self, manager: DegradationManager) -> None:
        """Should reset metrics."""
        manager.metrics.total_degraded_requests = 100
        manager.reset_metrics()
        assert manager.metrics.total_degraded_requests == 0


class TestWithFallback:
    """Tests for with_fallback decorator."""

    @pytest.mark.asyncio
    async def test_returns_primary_on_success(self) -> None:
        """Should return primary result when successful."""

        @with_fallback(lambda x: "fallback")  # noqa: ARG005
        async def fetch_data(x: int) -> str:
            return f"data-{x}"

        result = await fetch_data(42)
        assert result == "data-42"

    @pytest.mark.asyncio
    async def test_returns_fallback_on_error(self) -> None:
        """Should return fallback on error."""

        @with_fallback(lambda x: f"fallback-{x}")
        async def fetch_data(x: int) -> str:  # noqa: ARG001
            raise ConnectionError("API down")

        result = await fetch_data(42)
        assert result == "fallback-42"

    @pytest.mark.asyncio
    async def test_calls_on_error_callback(self) -> None:
        """Should call on_error callback."""
        errors: list[Exception] = []

        @with_fallback(lambda: "fallback", on_error=lambda e: errors.append(e))
        async def fetch_data() -> str:
            raise ValueError("Bad value")

        await fetch_data()
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)

    @pytest.mark.asyncio
    async def test_async_fallback(self) -> None:
        """Should handle async fallback function."""

        async def async_fallback(x: int) -> str:
            await asyncio.sleep(0)
            return f"async-fallback-{x}"

        @with_fallback(async_fallback)
        async def fetch_data(x: int) -> str:  # noqa: ARG001
            raise RuntimeError("Error")

        result = await fetch_data(5)
        assert result == "async-fallback-5"


class TestFallbackChain:
    """Tests for FallbackChain."""

    @pytest.mark.asyncio
    async def test_returns_first_success(self) -> None:
        """Should return result from first successful source."""
        chain: FallbackChain[str] = FallbackChain()
        chain.add("primary", lambda: "primary-data")
        chain.add("secondary", lambda: "secondary-data")

        result, source = await chain.execute()
        assert result == "primary-data"
        assert source == "primary"

    @pytest.mark.asyncio
    async def test_falls_through_on_error(self) -> None:
        """Should try next source on error."""

        def failing_source() -> str:
            raise ConnectionError("Down")

        chain: FallbackChain[str] = FallbackChain()
        chain.add("primary", failing_source)
        chain.add("secondary", lambda: "secondary-data")

        result, source = await chain.execute()
        assert result == "secondary-data"
        assert source == "secondary"

    @pytest.mark.asyncio
    async def test_raises_when_all_fail(self) -> None:
        """Should raise when all sources fail."""

        def failing() -> str:
            raise RuntimeError("Fail")

        chain: FallbackChain[str] = FallbackChain()
        chain.add("source1", failing)
        chain.add("source2", failing)

        with pytest.raises(RuntimeError, match="All fallback sources failed"):
            await chain.execute()

    @pytest.mark.asyncio
    async def test_async_sources(self) -> None:
        """Should handle async sources."""

        async def async_source() -> str:
            await asyncio.sleep(0)
            return "async-data"

        chain: FallbackChain[str] = FallbackChain()
        chain.add("async", async_source)

        result, source = await chain.execute()
        assert result == "async-data"
        assert source == "async"

    @pytest.mark.asyncio
    async def test_chaining_api(self) -> None:
        """Should support method chaining."""
        chain = (
            FallbackChain[int]()
            .add("a", lambda: 1)
            .add("b", lambda: 2)
            .add("c", lambda: 3)
        )

        result, _ = await chain.execute()
        assert result == 1


class TestGlobalManager:
    """Tests for global manager functions."""

    def test_initial_state(self) -> None:
        """Should be None initially."""
        reset_degradation_manager()
        assert get_degradation_manager() is None

    def test_set_and_get(self) -> None:
        """Should set and get manager."""
        reset_degradation_manager()
        manager = DegradationManager()
        set_degradation_manager(manager)

        assert get_degradation_manager() is manager

    def test_reset(self) -> None:
        """Should reset manager."""
        manager = DegradationManager()
        set_degradation_manager(manager)
        reset_degradation_manager()

        assert get_degradation_manager() is None


class TestConcurrency:
    """Tests for thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_status_updates(self) -> None:
        """Should handle concurrent status updates."""
        manager = DegradationManager()

        async def update_component(comp: ComponentType) -> None:
            for _ in range(10):
                await manager.mark_unhealthy(comp)
                await manager.mark_healthy(comp)

        await asyncio.gather(
            *[update_component(comp) for comp in ComponentType]
        )

        # Should not raise, final state depends on timing

    @pytest.mark.asyncio
    async def test_concurrent_metrics_recording(self) -> None:
        """Should handle concurrent metrics updates."""
        metrics = DegradationMetrics()

        async def record_events() -> None:
            for _ in range(100):
                await metrics.record_degradation(DegradationLevel.CACHED_DATA)
                await metrics.record_component_failure(ComponentType.LLM)

        await asyncio.gather(*[record_events() for _ in range(10)])

        # 10 tasks * 100 iterations = 1000 events
        assert metrics.total_degraded_requests == 1000
        assert metrics.component_failures[ComponentType.LLM] == 1000


class TestIntegration:
    """Integration tests for graceful degradation."""

    @pytest.mark.asyncio
    async def test_full_degradation_flow(self) -> None:
        """Test complete degradation and recovery flow."""
        alerts: list[DegradationLevel] = []

        async def on_alert(
            level: DegradationLevel, warnings: list[str]  # noqa: ARG001
        ) -> None:
            alerts.append(level)

        manager = DegradationManager(alert_callback=on_alert)

        # Start healthy
        await manager.mark_healthy(ComponentType.MARKET_DATA_API)
        await manager.mark_healthy(ComponentType.LLM)
        assert manager.get_degradation_level() == DegradationLevel.FULL

        # Market data fails -> cached data mode
        await manager.mark_unhealthy(ComponentType.MARKET_DATA_API, "API timeout")
        assert manager.get_degradation_level() == DegradationLevel.CACHED_DATA
        response = manager.wrap_response({"symbol": "AAPL"})
        assert response.is_degraded()
        assert len(response.warnings) >= 1

        # LLM also fails -> basic analysis (worse)
        await manager.mark_unhealthy(ComponentType.LLM, "Rate limited")
        assert manager.get_degradation_level() == DegradationLevel.BASIC_ANALYSIS

        # Market data recovers, but LLM still down
        await manager.mark_healthy(ComponentType.MARKET_DATA_API)
        assert manager.get_degradation_level() == DegradationLevel.BASIC_ANALYSIS

        # LLM recovers -> full service
        await manager.mark_healthy(ComponentType.LLM)
        assert manager.get_degradation_level() == DegradationLevel.FULL

        # Check alerts were triggered
        assert DegradationLevel.CACHED_DATA in alerts
        assert DegradationLevel.BASIC_ANALYSIS in alerts

    @pytest.mark.asyncio
    async def test_fallback_chain_with_degradation(self) -> None:
        """Test fallback chain updates degradation manager."""
        manager = DegradationManager()

        # Simulate fetching market data with fallbacks
        chain: FallbackChain[dict[str, float]] = FallbackChain()

        def api_fetch() -> dict[str, float]:
            raise ConnectionError("API down")

        def cache_fetch() -> dict[str, float]:
            return {"price": 150.0, "cached": True}

        chain.add("api", api_fetch)
        chain.add("cache", cache_fetch)

        result, source = await chain.execute()

        # Update degradation based on source
        if source != "api":
            await manager.mark_unhealthy(ComponentType.MARKET_DATA_API)

        assert result["price"] == 150.0
        assert manager.get_degradation_level() == DegradationLevel.CACHED_DATA
