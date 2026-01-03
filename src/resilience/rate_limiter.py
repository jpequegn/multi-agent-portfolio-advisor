"""Rate limiting implementation for API protection and cost management.

This module implements rate limiting using the token bucket algorithm
to protect the system and manage costs effectively.

Features:
- Per-user rate limits (requests per minute/hour)
- Global rate limits
- Cost-based limiting (daily spend caps)
- Rate limit header generation for HTTP responses
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded."""

    def __init__(
        self,
        limit_type: str,
        limit: int,
        reset_time: float,
        retry_after: float,
    ) -> None:
        self.limit_type = limit_type
        self.limit = limit
        self.reset_time = reset_time
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded: {limit_type}. "
            f"Limit: {limit}, retry after {retry_after:.1f}s"
        )


class CostLimitExceeded(Exception):
    """Raised when a cost limit is exceeded."""

    def __init__(
        self,
        user_id: str,
        current_cost: float,
        limit: float,
        reset_time: float,
    ) -> None:
        self.user_id = user_id
        self.current_cost = current_cost
        self.limit = limit
        self.reset_time = reset_time
        super().__init__(
            f"Cost limit exceeded for user {user_id}. "
            f"Current: ${current_cost:.2f}, Limit: ${limit:.2f}"
        )


class LimitType(Enum):
    """Types of rate limits."""

    USER_PER_MINUTE = "user_per_minute"
    USER_PER_HOUR = "user_per_hour"
    GLOBAL_PER_MINUTE = "global_per_minute"
    COST_DAILY = "cost_daily"


@dataclass
class RateLimitConfig:
    """Configuration for rate limits.

    Attributes:
        user_requests_per_minute: Max requests per user per minute.
        user_requests_per_hour: Max requests per user per hour.
        global_requests_per_minute: Max global requests per minute.
        user_daily_cost_limit: Max daily cost per user in dollars.
    """

    user_requests_per_minute: int = 10
    user_requests_per_hour: int = 100
    global_requests_per_minute: int = 100
    user_daily_cost_limit: float = 5.00


# Default configuration
DEFAULT_RATE_LIMIT_CONFIG = RateLimitConfig()


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Implements the token bucket algorithm where tokens are added
    at a constant rate and consumed by requests.

    Attributes:
        capacity: Maximum number of tokens in the bucket.
        refill_rate: Tokens added per second.
        tokens: Current number of tokens.
        last_refill: Timestamp of last refill.
    """

    capacity: int
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize tokens to capacity."""
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if insufficient tokens.
        """
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def get_status(self) -> dict[str, Any]:
        """Get current bucket status."""
        async with self._lock:
            self._refill()
            return {
                "tokens": int(self.tokens),
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
            }

    @property
    def time_until_refill(self) -> float:
        """Get seconds until at least one token is available."""
        if self.tokens >= 1:
            return 0.0
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.refill_rate


@dataclass
class CostTracker:
    """Track costs per user for cost-based rate limiting.

    Attributes:
        user_id: The user being tracked.
        daily_limit: Maximum daily cost in dollars.
        current_cost: Current accumulated cost.
        reset_time: Timestamp when the cost resets.
    """

    user_id: str
    daily_limit: float
    current_cost: float = 0.0
    reset_time: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        """Initialize reset time to end of current day."""
        self.reset_time = self._next_reset_time()

    @staticmethod
    def _next_reset_time() -> float:
        """Calculate the next reset time (midnight UTC)."""
        now = time.time()
        # Calculate seconds until midnight UTC
        seconds_per_day = 86400
        current_day_start = (now // seconds_per_day) * seconds_per_day
        return current_day_start + seconds_per_day

    def _check_reset(self) -> None:
        """Reset cost if past reset time."""
        now = time.time()
        if now >= self.reset_time:
            self.current_cost = 0.0
            self.reset_time = self._next_reset_time()
            logger.info(
                "cost_tracker_reset",
                user_id=self.user_id,
                new_reset_time=self.reset_time,
            )

    async def add_cost(self, cost: float) -> None:
        """Add cost to the tracker.

        Args:
            cost: Cost in dollars to add.

        Raises:
            CostLimitExceeded: If adding cost would exceed daily limit.
        """
        async with self._lock:
            self._check_reset()

            if self.current_cost + cost > self.daily_limit:
                raise CostLimitExceeded(
                    user_id=self.user_id,
                    current_cost=self.current_cost,
                    limit=self.daily_limit,
                    reset_time=self.reset_time,
                )

            self.current_cost += cost
            logger.debug(
                "cost_added",
                user_id=self.user_id,
                cost=cost,
                total_cost=self.current_cost,
                remaining=self.daily_limit - self.current_cost,
            )

    async def check_limit(self, estimated_cost: float = 0.0) -> bool:
        """Check if adding estimated cost would exceed limit.

        Args:
            estimated_cost: Estimated cost of the operation.

        Returns:
            True if within limit, False if would exceed.
        """
        async with self._lock:
            self._check_reset()
            return self.current_cost + estimated_cost <= self.daily_limit

    async def get_status(self) -> dict[str, Any]:
        """Get current cost tracker status."""
        async with self._lock:
            self._check_reset()
            return {
                "user_id": self.user_id,
                "current_cost": self.current_cost,
                "daily_limit": self.daily_limit,
                "remaining": self.daily_limit - self.current_cost,
                "reset_time": self.reset_time,
            }


@dataclass
class RateLimitHeaders:
    """Rate limit headers for HTTP responses.

    Standard headers following RFC 6585 / draft-ietf-httpapi-ratelimit-headers.

    Attributes:
        limit: The rate limit ceiling.
        remaining: Number of requests remaining.
        reset: Unix timestamp when the limit resets.
        retry_after: Seconds until retry is allowed (when limited).
    """

    limit: int
    remaining: int
    reset: int
    retry_after: int | None = None

    def to_dict(self) -> dict[str, str]:
        """Convert to HTTP headers dict."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return headers


class RateLimiter:
    """Rate limiter with per-user, global, and cost-based limits.

    Example:
        limiter = RateLimiter()

        # Check rate limit before processing request
        try:
            await limiter.check_rate_limit(user_id="user123")
            # Process request
            await limiter.record_cost(user_id="user123", cost=0.05)
        except RateLimitExceeded as e:
            return Response(status=429, headers=e.headers)
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration. Uses defaults if not provided.
        """
        self.config = config or DEFAULT_RATE_LIMIT_CONFIG

        # Per-user buckets: user_id -> {limit_type -> TokenBucket}
        self._user_buckets: dict[str, dict[str, TokenBucket]] = {}

        # Global bucket
        self._global_bucket = TokenBucket(
            capacity=self.config.global_requests_per_minute,
            refill_rate=self.config.global_requests_per_minute / 60.0,
        )

        # Cost trackers: user_id -> CostTracker
        self._cost_trackers: dict[str, CostTracker] = {}

        self._lock = asyncio.Lock()

    def _get_user_buckets(self, user_id: str) -> dict[str, TokenBucket]:
        """Get or create token buckets for a user."""
        if user_id not in self._user_buckets:
            self._user_buckets[user_id] = {
                LimitType.USER_PER_MINUTE.value: TokenBucket(
                    capacity=self.config.user_requests_per_minute,
                    refill_rate=self.config.user_requests_per_minute / 60.0,
                ),
                LimitType.USER_PER_HOUR.value: TokenBucket(
                    capacity=self.config.user_requests_per_hour,
                    refill_rate=self.config.user_requests_per_hour / 3600.0,
                ),
            }
        return self._user_buckets[user_id]

    def _get_cost_tracker(self, user_id: str) -> CostTracker:
        """Get or create cost tracker for a user."""
        if user_id not in self._cost_trackers:
            self._cost_trackers[user_id] = CostTracker(
                user_id=user_id,
                daily_limit=self.config.user_daily_cost_limit,
            )
        return self._cost_trackers[user_id]

    async def check_rate_limit(
        self,
        user_id: str,
        estimated_cost: float = 0.0,
    ) -> RateLimitHeaders:
        """Check if request is allowed under rate limits.

        Args:
            user_id: The user making the request.
            estimated_cost: Estimated cost of the operation for cost limiting.

        Returns:
            Rate limit headers to include in response.

        Raises:
            RateLimitExceeded: If any rate limit is exceeded.
            CostLimitExceeded: If cost limit would be exceeded.
        """
        async with self._lock:
            user_buckets = self._get_user_buckets(user_id)

        # Check global limit first
        if not await self._global_bucket.consume():
            reset_time = time.time() + 60
            raise RateLimitExceeded(
                limit_type=LimitType.GLOBAL_PER_MINUTE.value,
                limit=self.config.global_requests_per_minute,
                reset_time=reset_time,
                retry_after=self._global_bucket.time_until_refill,
            )

        # Check per-user per-minute limit
        minute_bucket = user_buckets[LimitType.USER_PER_MINUTE.value]
        if not await minute_bucket.consume():
            reset_time = time.time() + 60
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                limit_type=LimitType.USER_PER_MINUTE.value,
            )
            raise RateLimitExceeded(
                limit_type=LimitType.USER_PER_MINUTE.value,
                limit=self.config.user_requests_per_minute,
                reset_time=reset_time,
                retry_after=minute_bucket.time_until_refill,
            )

        # Check per-user per-hour limit
        hour_bucket = user_buckets[LimitType.USER_PER_HOUR.value]
        if not await hour_bucket.consume():
            reset_time = time.time() + 3600
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                limit_type=LimitType.USER_PER_HOUR.value,
            )
            raise RateLimitExceeded(
                limit_type=LimitType.USER_PER_HOUR.value,
                limit=self.config.user_requests_per_hour,
                reset_time=reset_time,
                retry_after=hour_bucket.time_until_refill,
            )

        # Check cost limit if estimated cost provided
        if estimated_cost > 0:
            cost_tracker = self._get_cost_tracker(user_id)
            if not await cost_tracker.check_limit(estimated_cost):
                status = await cost_tracker.get_status()
                raise CostLimitExceeded(
                    user_id=user_id,
                    current_cost=status["current_cost"],
                    limit=status["daily_limit"],
                    reset_time=status["reset_time"],
                )

        # Generate headers based on the most restrictive limit
        minute_status = await minute_bucket.get_status()
        return RateLimitHeaders(
            limit=self.config.user_requests_per_minute,
            remaining=minute_status["tokens"],
            reset=int(time.time() + 60),
        )

    async def record_cost(self, user_id: str, cost: float) -> None:
        """Record cost after a request completes.

        Args:
            user_id: The user who incurred the cost.
            cost: The cost in dollars.

        Raises:
            CostLimitExceeded: If cost limit is exceeded.
        """
        cost_tracker = self._get_cost_tracker(user_id)
        await cost_tracker.add_cost(cost)

    async def get_user_status(self, user_id: str) -> dict[str, Any]:
        """Get rate limit status for a user.

        Args:
            user_id: The user to get status for.

        Returns:
            Dict with all rate limit statuses.
        """
        async with self._lock:
            user_buckets = self._get_user_buckets(user_id)

        minute_status = await user_buckets[LimitType.USER_PER_MINUTE.value].get_status()
        hour_status = await user_buckets[LimitType.USER_PER_HOUR.value].get_status()
        cost_tracker = self._get_cost_tracker(user_id)
        cost_status = await cost_tracker.get_status()

        return {
            "user_id": user_id,
            "requests_per_minute": {
                "remaining": minute_status["tokens"],
                "limit": self.config.user_requests_per_minute,
            },
            "requests_per_hour": {
                "remaining": hour_status["tokens"],
                "limit": self.config.user_requests_per_hour,
            },
            "daily_cost": {
                "used": cost_status["current_cost"],
                "limit": cost_status["daily_limit"],
                "remaining": cost_status["remaining"],
            },
        }

    async def get_global_status(self) -> dict[str, Any]:
        """Get global rate limit status."""
        status = await self._global_bucket.get_status()
        return {
            "requests_per_minute": {
                "remaining": status["tokens"],
                "limit": self.config.global_requests_per_minute,
            },
        }

    def reset(self) -> None:
        """Reset all rate limiters (for testing)."""
        self._user_buckets.clear()
        self._cost_trackers.clear()
        self._global_bucket = TokenBucket(
            capacity=self.config.global_requests_per_minute,
            refill_rate=self.config.global_requests_per_minute / 60.0,
        )


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(config: RateLimitConfig | None = None) -> RateLimiter:
    """Get or create the global rate limiter.

    Args:
        config: Optional configuration. Only used when creating new instance.

    Returns:
        The global rate limiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(config)
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (for testing)."""
    global _rate_limiter
    _rate_limiter = None
