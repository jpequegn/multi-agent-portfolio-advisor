"""Cache manager for intelligent caching with Redis.

This module provides a caching layer to reduce costs and latency
by caching market data, news, and analysis results.

Features:
- Redis-based caching with configurable TTLs
- get_or_compute pattern for transparent caching
- Cache metrics tracking (hits, misses, latency)
- Cache invalidation support
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, cast

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CacheType(Enum):
    """Types of cached data with different TTLs."""

    MARKET_DATA = "market_data"
    NEWS = "news"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    # New types for Polygon.io integration
    QUOTE = "quote"
    DAILY_BARS = "daily_bars"
    COMPANY_INFO = "company_info"


@dataclass
class CacheConfig:
    """Configuration for cache TTLs.

    Attributes:
        market_data_ttl: TTL for market data in seconds (default: 5 min).
        news_ttl: TTL for news data in seconds (default: 15 min for Polygon).
        analysis_ttl: TTL for analysis results in seconds (default: 24 hours).
        recommendation_ttl: TTL for recommendations in seconds (default: 1 hour).
        quote_ttl: TTL for real-time quotes in seconds (default: 1 min).
        daily_bars_ttl: TTL for daily bars in seconds (default: 1 hour).
        company_info_ttl: TTL for company info in seconds (default: 24 hours).
        default_ttl: Default TTL for unspecified types (default: 5 min).
    """

    market_data_ttl: int = 300  # 5 minutes
    news_ttl: int = 900  # 15 minutes (optimized for Polygon rate limits)
    analysis_ttl: int = 86400  # 24 hours
    recommendation_ttl: int = 3600  # 1 hour
    quote_ttl: int = 60  # 1 minute (real-time quotes)
    daily_bars_ttl: int = 3600  # 1 hour
    company_info_ttl: int = 86400  # 24 hours
    default_ttl: int = 300  # 5 minutes

    def get_ttl(self, cache_type: CacheType) -> int:
        """Get TTL for a cache type."""
        ttl_map = {
            CacheType.MARKET_DATA: self.market_data_ttl,
            CacheType.NEWS: self.news_ttl,
            CacheType.ANALYSIS: self.analysis_ttl,
            CacheType.RECOMMENDATION: self.recommendation_ttl,
            CacheType.QUOTE: self.quote_ttl,
            CacheType.DAILY_BARS: self.daily_bars_ttl,
            CacheType.COMPANY_INFO: self.company_info_ttl,
        }
        return ttl_map.get(cache_type, self.default_ttl)


# Default cache configuration
DEFAULT_CACHE_CONFIG = CacheConfig()


@dataclass
class CacheMetrics:
    """Metrics for cache performance.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        errors: Number of cache errors.
        total_hit_latency_ms: Total latency for hits in milliseconds.
        total_miss_latency_ms: Total latency for misses in milliseconds.
    """

    hits: int = 0
    misses: int = 0
    errors: int = 0
    total_hit_latency_ms: float = 0.0
    total_miss_latency_ms: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    @property
    def avg_hit_latency_ms(self) -> float:
        """Average latency for cache hits."""
        if self.hits == 0:
            return 0.0
        return self.total_hit_latency_ms / self.hits

    @property
    def avg_miss_latency_ms(self) -> float:
        """Average latency for cache misses."""
        if self.misses == 0:
            return 0.0
        return self.total_miss_latency_ms / self.misses

    async def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit."""
        async with self._lock:
            self.hits += 1
            self.total_hit_latency_ms += latency_ms

    async def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss."""
        async with self._lock:
            self.misses += 1
            self.total_miss_latency_ms += latency_ms

    async def record_error(self) -> None:
        """Record a cache error."""
        async with self._lock:
            self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 2),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 2),
            "avg_miss_latency_ms": round(self.avg_miss_latency_ms, 2),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.total_hit_latency_ms = 0.0
        self.total_miss_latency_ms = 0.0


class CacheKeyBuilder:
    """Builder for consistent cache key generation."""

    PREFIX = "portfolio_advisor"

    @classmethod
    def market_data(cls, symbol: str, data_type: str = "quote") -> str:
        """Build cache key for market data.

        Args:
            symbol: Stock symbol (e.g., "AAPL").
            data_type: Type of data (e.g., "quote", "history").

        Returns:
            Cache key string.
        """
        return f"{cls.PREFIX}:market:{symbol.upper()}:{data_type}"

    @classmethod
    def news(cls, symbols: list[str], timestamp_bucket: int | None = None) -> str:
        """Build cache key for news data.

        Args:
            symbols: List of stock symbols.
            timestamp_bucket: Optional timestamp bucket for time-based keys.

        Returns:
            Cache key string.
        """
        symbols_hash = cls._hash_list(sorted(symbols))
        if timestamp_bucket:
            return f"{cls.PREFIX}:news:{symbols_hash}:{timestamp_bucket}"
        return f"{cls.PREFIX}:news:{symbols_hash}"

    @classmethod
    def analysis(cls, portfolio_id: str, version: str = "v1") -> str:
        """Build cache key for analysis results.

        Args:
            portfolio_id: Unique portfolio identifier.
            version: Analysis version for invalidation.

        Returns:
            Cache key string.
        """
        return f"{cls.PREFIX}:analysis:{portfolio_id}:{version}"

    @classmethod
    def recommendation(cls, portfolio_id: str, request_hash: str) -> str:
        """Build cache key for recommendations.

        Args:
            portfolio_id: Unique portfolio identifier.
            request_hash: Hash of the request parameters.

        Returns:
            Cache key string.
        """
        return f"{cls.PREFIX}:recommendation:{portfolio_id}:{request_hash}"

    @classmethod
    def custom(cls, namespace: str, *parts: str) -> str:
        """Build a custom cache key.

        Args:
            namespace: Namespace for the key.
            parts: Additional parts to include in the key.

        Returns:
            Cache key string.
        """
        return f"{cls.PREFIX}:{namespace}:{':'.join(parts)}"

    @staticmethod
    def _hash_list(items: list[str]) -> str:
        """Create a hash of a list of items."""
        content = ",".join(items)
        return hashlib.md5(content.encode()).hexdigest()[:12]


def serialize(value: Any) -> str:
    """Serialize a value for caching.

    Args:
        value: Value to serialize.

    Returns:
        JSON string representation.
    """
    return json.dumps(value, default=str)


def deserialize(data: str | bytes) -> Any:
    """Deserialize a cached value.

    Args:
        data: Serialized data from cache.

    Returns:
        Deserialized value.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


class CacheManager:
    """Cache manager with Redis backend.

    Provides intelligent caching with:
    - get_or_compute pattern for transparent caching
    - Configurable TTLs per data type
    - Metrics tracking
    - Cache invalidation

    Example:
        from redis.asyncio import Redis

        redis = Redis.from_url("redis://localhost:6379")
        cache = CacheManager(redis)

        # Get or compute with caching
        data = await cache.get_or_compute(
            key=CacheKeyBuilder.market_data("AAPL"),
            compute_fn=lambda: fetch_market_data("AAPL"),
            ttl=300,
        )

        # Check metrics
        print(cache.metrics.hit_rate)
    """

    def __init__(
        self,
        redis: Any,  # redis.asyncio.Redis
        config: CacheConfig | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize cache manager.

        Args:
            redis: Redis client instance.
            config: Cache configuration.
            enabled: Whether caching is enabled.
        """
        self.redis = redis
        self.config = config or DEFAULT_CACHE_CONFIG
        self.enabled = enabled
        self.metrics = CacheMetrics()

    async def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if not self.enabled:
            return None

        try:
            start = time.monotonic()
            data = await self.redis.get(key)
            latency_ms = (time.monotonic() - start) * 1000

            if data is not None:
                await self.metrics.record_hit(latency_ms)
                logger.debug("cache_hit", key=key, latency_ms=round(latency_ms, 2))
                return deserialize(data)

            await self.metrics.record_miss(latency_ms)
            logger.debug("cache_miss", key=key)
            return None

        except Exception as e:
            await self.metrics.record_error()
            logger.error("cache_get_error", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        cache_type: CacheType | None = None,
    ) -> bool:
        """Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: TTL in seconds. If not provided, uses cache_type or default.
            cache_type: Type of cache for TTL lookup.

        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            if ttl is None:
                ttl = self.config.get_ttl(cache_type) if cache_type else self.config.default_ttl

            serialized = serialize(value)
            await self.redis.setex(key, ttl, serialized)
            logger.debug("cache_set", key=key, ttl=ttl)
            return True

        except Exception as e:
            await self.metrics.record_error()
            logger.error("cache_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value from cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            result: int = await self.redis.delete(key)
            logger.debug("cache_delete", key=key, deleted=result > 0)
            return bool(result > 0)

        except Exception as e:
            await self.metrics.record_error()
            logger.error("cache_delete_error", key=key, error=str(e))
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "portfolio_advisor:analysis:*").

        Returns:
            Number of keys deleted.
        """
        if not self.enabled:
            return 0

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted: int = await self.redis.delete(*keys)
                logger.info(
                    "cache_delete_pattern",
                    pattern=pattern,
                    deleted_count=deleted,
                )
                return int(deleted)

            return 0

        except Exception as e:
            await self.metrics.record_error()
            logger.error("cache_delete_pattern_error", pattern=pattern, error=str(e))
            return 0

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Awaitable[T]],
        ttl: int | None = None,
        cache_type: CacheType | None = None,
    ) -> T:
        """Get value from cache or compute and cache it.

        This is the primary caching pattern - transparently handle
        cache hits and misses.

        Args:
            key: Cache key.
            compute_fn: Async function to compute value if not cached.
            ttl: TTL in seconds.
            cache_type: Type of cache for TTL lookup.

        Returns:
            Cached or computed value.
        """
        # Try to get from cache
        cached = await self.get(key)
        if cached is not None:
            return cast(T, cached)

        # Compute the value
        start = time.monotonic()
        result = await compute_fn()
        compute_time_ms = (time.monotonic() - start) * 1000

        # Cache the result
        await self.set(key, result, ttl=ttl, cache_type=cache_type)

        logger.debug(
            "cache_computed",
            key=key,
            compute_time_ms=round(compute_time_ms, 2),
        )

        return result

    async def invalidate_portfolio(self, portfolio_id: str) -> int:
        """Invalidate all cache entries for a portfolio.

        Call this when portfolio data changes.

        Args:
            portfolio_id: Portfolio identifier.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"{CacheKeyBuilder.PREFIX}:*:{portfolio_id}:*"
        deleted = await self.delete_pattern(pattern)
        logger.info(
            "cache_portfolio_invalidated",
            portfolio_id=portfolio_id,
            deleted_count=deleted,
        )
        return deleted

    async def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol.

        Call this when market data updates.

        Args:
            symbol: Stock symbol.

        Returns:
            Number of keys invalidated.
        """
        pattern = f"{CacheKeyBuilder.PREFIX}:market:{symbol.upper()}:*"
        deleted = await self.delete_pattern(pattern)
        logger.info(
            "cache_symbol_invalidated",
            symbol=symbol,
            deleted_count=deleted,
        )
        return deleted

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists.
        """
        if not self.enabled:
            return False

        try:
            result: int = await self.redis.exists(key)
            return bool(result > 0)
        except Exception as e:
            await self.metrics.record_error()
            logger.error("cache_exists_error", key=key, error=str(e))
            return False

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key.

        Args:
            key: Cache key.

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist.
        """
        if not self.enabled:
            return -2

        try:
            result: int = await self.redis.ttl(key)
            return int(result)
        except Exception as e:
            await self.metrics.record_error()
            logger.error("cache_ttl_error", key=key, error=str(e))
            return -2

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics.

        Returns:
            Dictionary of cache metrics.
        """
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.metrics.reset()


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager | None:
    """Get the global cache manager instance.

    Returns:
        Cache manager or None if not initialized.
    """
    return _cache_manager


def set_cache_manager(manager: CacheManager) -> None:
    """Set the global cache manager instance.

    Args:
        manager: Cache manager to set as global.
    """
    global _cache_manager
    _cache_manager = manager


def reset_cache_manager() -> None:
    """Reset the global cache manager (for testing)."""
    global _cache_manager
    _cache_manager = None
