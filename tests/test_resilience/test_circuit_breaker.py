"""Tests for circuit breaker implementation."""

import asyncio

import pytest

from src.resilience.circuit_breaker import (
    API_CIRCUIT_CONFIG,
    LLM_CIRCUIT_CONFIG,
    MARKET_DATA_CIRCUIT_CONFIG,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    api_circuit_breaker,
    circuit_breaker,
    get_all_circuit_breakers,
    get_circuit_breaker,
    llm_circuit_breaker,
    market_data_circuit_breaker,
    reset_all_circuit_breakers,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_has_closed_state(self) -> None:
        """Test CLOSED state exists."""
        assert CircuitState.CLOSED.value == "closed"

    def test_has_open_state(self) -> None:
        """Test OPEN state exists."""
        assert CircuitState.OPEN.value == "open"

    def test_has_half_open_state(self) -> None:
        """Test HALF_OPEN state exists."""
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.excluded_exceptions == ()

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            half_open_max_calls=5,
            success_threshold=3,
            excluded_exceptions=(ValueError,),
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120.0
        assert config.half_open_max_calls == 5
        assert config.success_threshold == 3
        assert config.excluded_exceptions == (ValueError,)

    def test_llm_config(self) -> None:
        """Test predefined LLM circuit config."""
        assert LLM_CIRCUIT_CONFIG.failure_threshold == 5
        assert LLM_CIRCUIT_CONFIG.recovery_timeout == 60.0

    def test_api_config(self) -> None:
        """Test predefined API circuit config."""
        assert API_CIRCUIT_CONFIG.failure_threshold == 3
        assert API_CIRCUIT_CONFIG.recovery_timeout == 30.0

    def test_market_data_config(self) -> None:
        """Test predefined market data circuit config."""
        assert MARKET_DATA_CIRCUIT_CONFIG.failure_threshold == 5
        assert MARKET_DATA_CIRCUIT_CONFIG.recovery_timeout == 45.0


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_message(self) -> None:
        """Test error message formatting."""
        error = CircuitOpenError("test_circuit", 30.5)
        assert "test_circuit" in str(error)
        assert "30.5" in str(error)
        assert error.name == "test_circuit"
        assert error.recovery_time == 30.5


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create a circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short for testing
            half_open_max_calls=2,
            success_threshold=2,
        )
        return CircuitBreaker("test", config)

    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker: CircuitBreaker) -> None:
        """Test circuit breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_allows_requests_when_closed(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test requests are allowed when closed."""
        allowed = await breaker.allow_request()
        assert allowed is True

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit opens after failure threshold is reached."""
        # Record failures up to threshold
        for _ in range(3):
            await breaker.record_failure(Exception("test error"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_rejects_requests_when_open(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test requests are rejected when circuit is open."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure(Exception("test error"))

        with pytest.raises(CircuitOpenError) as exc_info:
            await breaker.allow_request()

        assert exc_info.value.name == "test"

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test success resets failure count when closed."""
        await breaker.record_failure(Exception("test error"))
        assert breaker.failure_count == 1

        await breaker.record_success()
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_recovery_timeout(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit transitions to half-open after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure(Exception("test error"))

        assert breaker._state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # State property should now return HALF_OPEN
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_allows_limited_requests(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test half-open state allows limited requests."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure(Exception("test error"))

        # Wait for recovery
        await asyncio.sleep(1.1)

        # Should allow up to half_open_max_calls requests
        await breaker.allow_request()
        await breaker.allow_request()

        # Third request should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.allow_request()

    @pytest.mark.asyncio
    async def test_closes_after_successes_in_half_open(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit closes after successful calls in half-open."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure(Exception("test error"))

        # Wait for recovery
        await asyncio.sleep(1.1)

        # Transition to half-open and record successes
        await breaker.allow_request()
        await breaker.record_success()
        await breaker.record_success()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test circuit reopens on failure in half-open state."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure(Exception("test error"))

        # Wait for recovery
        await asyncio.sleep(1.1)

        # Enter half-open
        await breaker.allow_request()

        # Fail in half-open
        await breaker.record_failure(Exception("test error"))

        assert breaker._state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions_not_counted(self) -> None:
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test", config)

        # ValueError should not count
        await breaker.record_failure(ValueError("excluded"))
        assert breaker.failure_count == 0

        # Other exceptions should count
        await breaker.record_failure(RuntimeError("included"))
        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_success(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test context manager records success."""
        async with breaker:
            pass  # Successful operation

        # Should have recorded success (failure count stays 0)
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_failure(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test context manager records failure."""
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("test error")

        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_get_status(self, breaker: CircuitBreaker) -> None:
        """Test get_status returns expected fields."""
        status = breaker.get_status()

        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 3
        assert "time_until_recovery" in status
        assert "recovery_timeout" in status

    @pytest.mark.asyncio
    async def test_time_until_recovery(
        self, breaker: CircuitBreaker
    ) -> None:
        """Test time_until_recovery calculation."""
        # Initially zero
        assert breaker.time_until_recovery == 0.0

        # After failure, should be close to recovery_timeout
        await breaker.record_failure(Exception("test"))
        assert breaker.time_until_recovery > 0.9
        assert breaker.time_until_recovery <= 1.0


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry functions."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_all_circuit_breakers()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_all_circuit_breakers()

    def test_get_circuit_breaker_creates_new(self) -> None:
        """Test get_circuit_breaker creates new circuit breaker."""
        breaker = get_circuit_breaker("new_circuit")
        assert breaker.name == "new_circuit"

    def test_get_circuit_breaker_returns_existing(self) -> None:
        """Test get_circuit_breaker returns existing circuit breaker."""
        breaker1 = get_circuit_breaker("test")
        breaker2 = get_circuit_breaker("test")
        assert breaker1 is breaker2

    def test_get_circuit_breaker_with_config(self) -> None:
        """Test get_circuit_breaker with custom config."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = get_circuit_breaker("custom", config)
        assert breaker.config.failure_threshold == 10

    def test_get_all_circuit_breakers(self) -> None:
        """Test get_all_circuit_breakers returns all registered."""
        get_circuit_breaker("circuit1")
        get_circuit_breaker("circuit2")

        all_breakers = get_all_circuit_breakers()
        assert "circuit1" in all_breakers
        assert "circuit2" in all_breakers
        assert len(all_breakers) == 2

    def test_reset_all_circuit_breakers(self) -> None:
        """Test reset_all_circuit_breakers clears registry."""
        get_circuit_breaker("circuit1")
        get_circuit_breaker("circuit2")

        reset_all_circuit_breakers()

        assert len(get_all_circuit_breakers()) == 0


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker decorator."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_all_circuit_breakers()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_decorator_wraps_async_function(self) -> None:
        """Test decorator wraps async function."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @circuit_breaker("test_async", config)
        async def async_func() -> str:
            return "success"

        result = await async_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_records_success(self) -> None:
        """Test decorator records successful calls."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @circuit_breaker("test_success", config)
        async def successful_func() -> str:
            return "success"

        await successful_func()

        breaker = get_circuit_breaker("test_success")
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_decorator_records_failure(self) -> None:
        """Test decorator records failed calls."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @circuit_breaker("test_failure", config)
        async def failing_func() -> str:
            raise RuntimeError("test error")

        with pytest.raises(RuntimeError):
            await failing_func()

        breaker = get_circuit_breaker("test_failure")
        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_decorator_opens_circuit(self) -> None:
        """Test decorator opens circuit after failures."""
        config = CircuitBreakerConfig(failure_threshold=2)

        @circuit_breaker("test_open", config)
        async def failing_func() -> str:
            raise RuntimeError("test error")

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await failing_func()

        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await failing_func()

    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self) -> None:
        """Test decorator uses fallback when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=2)

        async def fallback_func() -> str:
            return "fallback"

        @circuit_breaker("test_fallback", config, fallback=fallback_func)
        async def failing_func() -> str:
            raise RuntimeError("test error")

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await failing_func()

        # Next call should use fallback
        result = await failing_func()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_decorator_with_sync_fallback(self) -> None:
        """Test decorator works with sync fallback."""
        config = CircuitBreakerConfig(failure_threshold=2)

        def sync_fallback() -> str:
            return "sync_fallback"

        @circuit_breaker("test_sync_fallback", config, fallback=sync_fallback)
        async def failing_func() -> str:
            raise RuntimeError("test error")

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await failing_func()

        # Next call should use fallback
        result = await failing_func()
        assert result == "sync_fallback"


class TestConvenienceDecorators:
    """Tests for convenience decorators."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_all_circuit_breakers()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_llm_circuit_breaker(self) -> None:
        """Test llm_circuit_breaker decorator."""

        @llm_circuit_breaker()
        async def call_llm() -> str:
            return "llm response"

        result = await call_llm()
        assert result == "llm response"

        # Check correct circuit was created
        breaker = get_circuit_breaker("llm")
        assert breaker.config.failure_threshold == 5
        assert breaker.config.recovery_timeout == 60.0

    @pytest.mark.asyncio
    async def test_api_circuit_breaker(self) -> None:
        """Test api_circuit_breaker decorator."""

        @api_circuit_breaker("test_api")
        async def call_api() -> str:
            return "api response"

        result = await call_api()
        assert result == "api response"

        # Check correct circuit was created
        breaker = get_circuit_breaker("api_test_api")
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 30.0

    @pytest.mark.asyncio
    async def test_market_data_circuit_breaker(self) -> None:
        """Test market_data_circuit_breaker decorator."""

        @market_data_circuit_breaker()
        async def fetch_data() -> str:
            return "market data"

        result = await fetch_data()
        assert result == "market data"

        # Check correct circuit was created
        breaker = get_circuit_breaker("market_data")
        assert breaker.config.failure_threshold == 5
        assert breaker.config.recovery_timeout == 45.0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_all_circuit_breakers()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_all_circuit_breakers()

    @pytest.mark.asyncio
    async def test_full_cycle_with_recovery(self) -> None:
        """Test full circuit breaker cycle: closed -> open -> half-open -> closed."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            half_open_max_calls=2,
            success_threshold=2,
        )
        breaker = CircuitBreaker("integration_test", config)

        # Start closed
        assert breaker.state == CircuitState.CLOSED

        # Fail to open
        for _ in range(2):
            await breaker.record_failure(Exception("fail"))
        assert breaker._state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

        # Succeed to close
        await breaker.allow_request()
        await breaker.record_success()
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_circuits_independent(self) -> None:
        """Test multiple circuits operate independently."""
        breaker1 = get_circuit_breaker(
            "circuit1",
            CircuitBreakerConfig(failure_threshold=2),
        )
        breaker2 = get_circuit_breaker(
            "circuit2",
            CircuitBreakerConfig(failure_threshold=2),
        )

        # Open circuit1
        for _ in range(2):
            await breaker1.record_failure(Exception("fail"))

        # Circuit1 is open, circuit2 is still closed
        assert breaker1._state == CircuitState.OPEN
        assert breaker2.state == CircuitState.CLOSED

        # Can still use circuit2
        allowed = await breaker2.allow_request()
        assert allowed is True

    @pytest.mark.asyncio
    async def test_concurrent_access(self) -> None:
        """Test circuit breaker handles concurrent access."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("concurrent_test", config)

        async def record_failure() -> None:
            await breaker.record_failure(Exception("fail"))

        # Concurrent failures
        await asyncio.gather(*[record_failure() for _ in range(5)])

        assert breaker._state == CircuitState.OPEN
        assert breaker.failure_count >= 5
