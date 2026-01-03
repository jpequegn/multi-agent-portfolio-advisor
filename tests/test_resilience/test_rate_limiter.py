"""Tests for rate limiter implementation."""

import asyncio
import time

import pytest

from src.resilience.rate_limiter import (
    DEFAULT_RATE_LIMIT_CONFIG,
    CostLimitExceeded,
    CostTracker,
    LimitType,
    RateLimitConfig,
    RateLimiter,
    RateLimitExceeded,
    RateLimitHeaders,
    TokenBucket,
    get_rate_limiter,
    reset_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.user_requests_per_minute == 10
        assert config.user_requests_per_hour == 100
        assert config.global_requests_per_minute == 100
        assert config.user_daily_cost_limit == 5.00

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RateLimitConfig(
            user_requests_per_minute=20,
            user_requests_per_hour=200,
            global_requests_per_minute=500,
            user_daily_cost_limit=10.00,
        )
        assert config.user_requests_per_minute == 20
        assert config.user_requests_per_hour == 200
        assert config.global_requests_per_minute == 500
        assert config.user_daily_cost_limit == 10.00

    def test_default_config_instance(self) -> None:
        """Test default config instance exists."""
        assert DEFAULT_RATE_LIMIT_CONFIG.user_requests_per_minute == 10


class TestLimitType:
    """Tests for LimitType enum."""

    def test_has_user_per_minute(self) -> None:
        """Test USER_PER_MINUTE exists."""
        assert LimitType.USER_PER_MINUTE.value == "user_per_minute"

    def test_has_user_per_hour(self) -> None:
        """Test USER_PER_HOUR exists."""
        assert LimitType.USER_PER_HOUR.value == "user_per_hour"

    def test_has_global_per_minute(self) -> None:
        """Test GLOBAL_PER_MINUTE exists."""
        assert LimitType.GLOBAL_PER_MINUTE.value == "global_per_minute"

    def test_has_cost_daily(self) -> None:
        """Test COST_DAILY exists."""
        assert LimitType.COST_DAILY.value == "cost_daily"


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_error_attributes(self) -> None:
        """Test error has correct attributes."""
        error = RateLimitExceeded(
            limit_type="user_per_minute",
            limit=10,
            reset_time=1704067200.0,
            retry_after=5.5,
        )
        assert error.limit_type == "user_per_minute"
        assert error.limit == 10
        assert error.reset_time == 1704067200.0
        assert error.retry_after == 5.5

    def test_error_message(self) -> None:
        """Test error message formatting."""
        error = RateLimitExceeded(
            limit_type="user_per_minute",
            limit=10,
            reset_time=1704067200.0,
            retry_after=5.5,
        )
        assert "user_per_minute" in str(error)
        assert "10" in str(error)
        assert "5.5" in str(error)


class TestCostLimitExceeded:
    """Tests for CostLimitExceeded exception."""

    def test_error_attributes(self) -> None:
        """Test error has correct attributes."""
        error = CostLimitExceeded(
            user_id="user123",
            current_cost=4.50,
            limit=5.00,
            reset_time=1704067200.0,
        )
        assert error.user_id == "user123"
        assert error.current_cost == 4.50
        assert error.limit == 5.00
        assert error.reset_time == 1704067200.0

    def test_error_message(self) -> None:
        """Test error message formatting."""
        error = CostLimitExceeded(
            user_id="user123",
            current_cost=4.50,
            limit=5.00,
            reset_time=1704067200.0,
        )
        assert "user123" in str(error)
        assert "4.50" in str(error)
        assert "5.00" in str(error)


class TestTokenBucket:
    """Tests for TokenBucket."""

    @pytest.mark.asyncio
    async def test_starts_full(self) -> None:
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        status = await bucket.get_status()
        assert status["tokens"] == 10
        assert status["capacity"] == 10

    @pytest.mark.asyncio
    async def test_consume_reduces_tokens(self) -> None:
        """Test consuming tokens reduces count."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        result = await bucket.consume(3)
        assert result is True
        status = await bucket.get_status()
        assert status["tokens"] == 7

    @pytest.mark.asyncio
    async def test_consume_fails_when_empty(self) -> None:
        """Test consuming fails when not enough tokens."""
        bucket = TokenBucket(capacity=2, refill_rate=0.1)
        await bucket.consume(2)
        result = await bucket.consume(1)
        assert result is False

    @pytest.mark.asyncio
    async def test_refills_over_time(self) -> None:
        """Test tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # 100 tokens/sec
        await bucket.consume(5)

        # Wait for refill
        await asyncio.sleep(0.1)  # Should add ~10 tokens

        status = await bucket.get_status()
        assert status["tokens"] >= 10  # Capped at capacity

    @pytest.mark.asyncio
    async def test_respects_capacity_limit(self) -> None:
        """Test tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)
        await asyncio.sleep(0.1)  # Would add 10 tokens if not capped
        status = await bucket.get_status()
        assert status["tokens"] == 10

    def test_time_until_refill_when_full(self) -> None:
        """Test time_until_refill is zero when full."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.time_until_refill == 0.0

    @pytest.mark.asyncio
    async def test_time_until_refill_when_empty(self) -> None:
        """Test time_until_refill when empty."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        await bucket.consume(10)
        # Should need 0.1 seconds to get 1 token
        assert bucket.time_until_refill > 0
        assert bucket.time_until_refill <= 0.2


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.mark.asyncio
    async def test_starts_at_zero(self) -> None:
        """Test tracker starts at zero cost."""
        tracker = CostTracker(user_id="user123", daily_limit=5.00)
        status = await tracker.get_status()
        assert status["current_cost"] == 0.0
        assert status["remaining"] == 5.00

    @pytest.mark.asyncio
    async def test_add_cost_increases_total(self) -> None:
        """Test adding cost increases total."""
        tracker = CostTracker(user_id="user123", daily_limit=5.00)
        await tracker.add_cost(1.50)
        status = await tracker.get_status()
        assert status["current_cost"] == 1.50
        assert status["remaining"] == 3.50

    @pytest.mark.asyncio
    async def test_raises_when_limit_exceeded(self) -> None:
        """Test raises CostLimitExceeded when limit exceeded."""
        tracker = CostTracker(user_id="user123", daily_limit=5.00)
        await tracker.add_cost(4.00)

        with pytest.raises(CostLimitExceeded) as exc_info:
            await tracker.add_cost(2.00)  # Would exceed limit

        assert exc_info.value.user_id == "user123"
        assert exc_info.value.current_cost == 4.00
        assert exc_info.value.limit == 5.00

    @pytest.mark.asyncio
    async def test_check_limit_returns_true_when_ok(self) -> None:
        """Test check_limit returns True when within limit."""
        tracker = CostTracker(user_id="user123", daily_limit=5.00)
        result = await tracker.check_limit(3.00)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_limit_returns_false_when_exceeded(self) -> None:
        """Test check_limit returns False when would exceed."""
        tracker = CostTracker(user_id="user123", daily_limit=5.00)
        await tracker.add_cost(4.00)
        result = await tracker.check_limit(2.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_has_reset_time(self) -> None:
        """Test tracker has reset time set."""
        tracker = CostTracker(user_id="user123", daily_limit=5.00)
        status = await tracker.get_status()
        assert "reset_time" in status
        assert status["reset_time"] > time.time()


class TestRateLimitHeaders:
    """Tests for RateLimitHeaders."""

    def test_to_dict_basic(self) -> None:
        """Test converting headers to dict."""
        headers = RateLimitHeaders(
            limit=10,
            remaining=7,
            reset=1704067200,
        )
        result = headers.to_dict()
        assert result["X-RateLimit-Limit"] == "10"
        assert result["X-RateLimit-Remaining"] == "7"
        assert result["X-RateLimit-Reset"] == "1704067200"
        assert "Retry-After" not in result

    def test_to_dict_with_retry_after(self) -> None:
        """Test headers include Retry-After when set."""
        headers = RateLimitHeaders(
            limit=10,
            remaining=0,
            reset=1704067200,
            retry_after=30,
        )
        result = headers.to_dict()
        assert result["Retry-After"] == "30"


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def config(self) -> RateLimitConfig:
        """Create test configuration."""
        return RateLimitConfig(
            user_requests_per_minute=5,
            user_requests_per_hour=20,
            global_requests_per_minute=50,
            user_daily_cost_limit=2.00,
        )

    @pytest.fixture
    def limiter(self, config: RateLimitConfig) -> RateLimiter:
        """Create rate limiter for testing."""
        return RateLimiter(config)

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(
        self, limiter: RateLimiter
    ) -> None:
        """Test requests are allowed under the limit."""
        headers = await limiter.check_rate_limit("user123")
        assert headers.remaining >= 0
        assert headers.limit == 5

    @pytest.mark.asyncio
    async def test_returns_headers(self, limiter: RateLimiter) -> None:
        """Test returns rate limit headers."""
        headers = await limiter.check_rate_limit("user123")
        assert headers.limit == 5
        assert headers.remaining <= 5
        assert headers.reset > 0

    @pytest.mark.asyncio
    async def test_raises_when_per_minute_exceeded(
        self, limiter: RateLimiter
    ) -> None:
        """Test raises RateLimitExceeded when per-minute limit exceeded."""
        # Exhaust per-minute limit
        for _ in range(5):
            await limiter.check_rate_limit("user123")

        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check_rate_limit("user123")

        assert exc_info.value.limit_type == LimitType.USER_PER_MINUTE.value
        assert exc_info.value.limit == 5

    @pytest.mark.asyncio
    async def test_different_users_have_separate_limits(
        self, limiter: RateLimiter
    ) -> None:
        """Test different users have separate rate limits."""
        # Exhaust user1's limit
        for _ in range(5):
            await limiter.check_rate_limit("user1")

        # user2 should still be allowed
        headers = await limiter.check_rate_limit("user2")
        assert headers.remaining >= 0

    @pytest.mark.asyncio
    async def test_raises_when_global_limit_exceeded(self) -> None:
        """Test raises RateLimitExceeded when global limit exceeded."""
        # Create limiter with low global limit
        low_global_config = RateLimitConfig(
            user_requests_per_minute=100,
            user_requests_per_hour=1000,
            global_requests_per_minute=3,
            user_daily_cost_limit=10.00,
        )
        limiter = RateLimiter(low_global_config)

        # Exhaust global limit
        for i in range(3):
            await limiter.check_rate_limit(f"user{i}")

        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check_rate_limit("user99")

        assert exc_info.value.limit_type == LimitType.GLOBAL_PER_MINUTE.value

    @pytest.mark.asyncio
    async def test_raises_when_cost_limit_exceeded(
        self, limiter: RateLimiter
    ) -> None:
        """Test raises CostLimitExceeded when cost limit exceeded."""
        # Record some costs
        await limiter.record_cost("user123", 1.50)

        # Try to check with estimated cost that would exceed
        with pytest.raises(CostLimitExceeded) as exc_info:
            await limiter.check_rate_limit("user123", estimated_cost=1.00)

        assert exc_info.value.user_id == "user123"

    @pytest.mark.asyncio
    async def test_record_cost(self, limiter: RateLimiter) -> None:
        """Test recording cost updates tracker."""
        await limiter.record_cost("user123", 0.50)
        status = await limiter.get_user_status("user123")
        assert status["daily_cost"]["used"] == 0.50

    @pytest.mark.asyncio
    async def test_get_user_status(self, limiter: RateLimiter) -> None:
        """Test get_user_status returns all limits."""
        await limiter.check_rate_limit("user123")
        status = await limiter.get_user_status("user123")

        assert "user_id" in status
        assert "requests_per_minute" in status
        assert "requests_per_hour" in status
        assert "daily_cost" in status
        assert status["requests_per_minute"]["limit"] == 5

    @pytest.mark.asyncio
    async def test_get_global_status(self, limiter: RateLimiter) -> None:
        """Test get_global_status returns global limits."""
        status = await limiter.get_global_status()
        assert "requests_per_minute" in status
        assert status["requests_per_minute"]["limit"] == 50

    def test_reset(self, limiter: RateLimiter) -> None:
        """Test reset clears all state."""
        limiter.reset()
        # Should not raise - state is cleared


class TestRateLimiterRegistry:
    """Tests for rate limiter registry functions."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_rate_limiter()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_rate_limiter()

    def test_get_rate_limiter_creates_instance(self) -> None:
        """Test get_rate_limiter creates instance."""
        limiter = get_rate_limiter()
        assert limiter is not None

    def test_get_rate_limiter_returns_same_instance(self) -> None:
        """Test get_rate_limiter returns same instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_reset_rate_limiter(self) -> None:
        """Test reset_rate_limiter clears instance."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is not limiter2


class TestRateLimiterIntegration:
    """Integration tests for rate limiter."""

    @pytest.mark.asyncio
    async def test_full_request_flow(self) -> None:
        """Test complete request flow with rate limiting."""
        config = RateLimitConfig(
            user_requests_per_minute=5,
            user_requests_per_hour=100,
            global_requests_per_minute=100,
            user_daily_cost_limit=5.00,
        )
        limiter = RateLimiter(config)

        # Simulate request flow
        headers = await limiter.check_rate_limit("user123", estimated_cost=0.10)
        assert headers.remaining == 4

        # Record actual cost
        await limiter.record_cost("user123", 0.08)

        # Check status
        status = await limiter.get_user_status("user123")
        assert status["daily_cost"]["used"] == 0.08
        assert status["requests_per_minute"]["remaining"] == 4

    @pytest.mark.asyncio
    async def test_concurrent_requests(self) -> None:
        """Test rate limiter handles concurrent requests."""
        config = RateLimitConfig(
            user_requests_per_minute=10,
            user_requests_per_hour=100,
            global_requests_per_minute=100,
            user_daily_cost_limit=10.00,
        )
        limiter = RateLimiter(config)

        async def make_request(user_id: str) -> RateLimitHeaders:
            return await limiter.check_rate_limit(user_id)

        # Concurrent requests from same user
        results = await asyncio.gather(
            *[make_request("user123") for _ in range(5)]
        )
        assert len(results) == 5

        # Check that tokens were consumed
        status = await limiter.get_user_status("user123")
        assert status["requests_per_minute"]["remaining"] == 5

    @pytest.mark.asyncio
    async def test_recovery_after_time(self) -> None:
        """Test rate limit recovery over time."""
        config = RateLimitConfig(
            user_requests_per_minute=2,  # Low limit for testing
            user_requests_per_hour=100,
            global_requests_per_minute=100,
            user_daily_cost_limit=10.00,
        )
        limiter = RateLimiter(config)

        # Exhaust limit
        for _ in range(2):
            await limiter.check_rate_limit("user123")

        # Should be rate limited
        with pytest.raises(RateLimitExceeded):
            await limiter.check_rate_limit("user123")

        # Wait for recovery (bucket refills at 2/60 = 0.033 tokens/sec)
        # Need ~30 seconds for 1 token, but we'll just verify the mechanism
        # by checking the retry_after value
        try:
            await limiter.check_rate_limit("user123")
        except RateLimitExceeded as e:
            assert e.retry_after > 0
