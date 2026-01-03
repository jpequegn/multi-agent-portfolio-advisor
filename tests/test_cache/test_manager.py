"""Tests for cache manager implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.cache.manager import (
    DEFAULT_CACHE_CONFIG,
    CacheConfig,
    CacheKeyBuilder,
    CacheManager,
    CacheMetrics,
    CacheType,
    deserialize,
    get_cache_manager,
    reset_cache_manager,
    serialize,
    set_cache_manager,
)


class TestCacheType:
    """Tests for CacheType enum."""

    def test_has_market_data(self) -> None:
        """Test MARKET_DATA type exists."""
        assert CacheType.MARKET_DATA.value == "market_data"

    def test_has_news(self) -> None:
        """Test NEWS type exists."""
        assert CacheType.NEWS.value == "news"

    def test_has_analysis(self) -> None:
        """Test ANALYSIS type exists."""
        assert CacheType.ANALYSIS.value == "analysis"

    def test_has_recommendation(self) -> None:
        """Test RECOMMENDATION type exists."""
        assert CacheType.RECOMMENDATION.value == "recommendation"


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.market_data_ttl == 300  # 5 minutes
        assert config.news_ttl == 3600  # 1 hour
        assert config.analysis_ttl == 86400  # 24 hours
        assert config.recommendation_ttl == 3600  # 1 hour
        assert config.default_ttl == 300  # 5 minutes

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CacheConfig(
            market_data_ttl=60,
            news_ttl=1800,
            analysis_ttl=7200,
            recommendation_ttl=900,
            default_ttl=120,
        )
        assert config.market_data_ttl == 60
        assert config.news_ttl == 1800
        assert config.analysis_ttl == 7200
        assert config.recommendation_ttl == 900
        assert config.default_ttl == 120

    def test_get_ttl_for_market_data(self) -> None:
        """Test get_ttl returns correct TTL for market data."""
        config = CacheConfig(market_data_ttl=120)
        assert config.get_ttl(CacheType.MARKET_DATA) == 120

    def test_get_ttl_for_news(self) -> None:
        """Test get_ttl returns correct TTL for news."""
        config = CacheConfig(news_ttl=1800)
        assert config.get_ttl(CacheType.NEWS) == 1800

    def test_get_ttl_for_analysis(self) -> None:
        """Test get_ttl returns correct TTL for analysis."""
        config = CacheConfig(analysis_ttl=7200)
        assert config.get_ttl(CacheType.ANALYSIS) == 7200

    def test_default_config_instance(self) -> None:
        """Test default config instance exists."""
        assert DEFAULT_CACHE_CONFIG.market_data_ttl == 300


class TestCacheMetrics:
    """Tests for CacheMetrics."""

    @pytest.fixture
    def metrics(self) -> CacheMetrics:
        """Create metrics instance for testing."""
        return CacheMetrics()

    def test_starts_at_zero(self, metrics: CacheMetrics) -> None:
        """Test metrics start at zero."""
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.errors == 0

    @pytest.mark.asyncio
    async def test_record_hit(self, metrics: CacheMetrics) -> None:
        """Test recording a hit."""
        await metrics.record_hit(5.0)
        assert metrics.hits == 1
        assert metrics.total_hit_latency_ms == 5.0

    @pytest.mark.asyncio
    async def test_record_miss(self, metrics: CacheMetrics) -> None:
        """Test recording a miss."""
        await metrics.record_miss(10.0)
        assert metrics.misses == 1
        assert metrics.total_miss_latency_ms == 10.0

    @pytest.mark.asyncio
    async def test_record_error(self, metrics: CacheMetrics) -> None:
        """Test recording an error."""
        await metrics.record_error()
        assert metrics.errors == 1

    @pytest.mark.asyncio
    async def test_total_requests(self, metrics: CacheMetrics) -> None:
        """Test total requests calculation."""
        await metrics.record_hit(1.0)
        await metrics.record_hit(1.0)
        await metrics.record_miss(1.0)
        assert metrics.total_requests == 3

    @pytest.mark.asyncio
    async def test_hit_rate(self, metrics: CacheMetrics) -> None:
        """Test hit rate calculation."""
        await metrics.record_hit(1.0)
        await metrics.record_hit(1.0)
        await metrics.record_miss(1.0)
        await metrics.record_miss(1.0)
        assert metrics.hit_rate == 50.0

    def test_hit_rate_zero_requests(self, metrics: CacheMetrics) -> None:
        """Test hit rate with zero requests."""
        assert metrics.hit_rate == 0.0

    @pytest.mark.asyncio
    async def test_avg_hit_latency(self, metrics: CacheMetrics) -> None:
        """Test average hit latency calculation."""
        await metrics.record_hit(10.0)
        await metrics.record_hit(20.0)
        assert metrics.avg_hit_latency_ms == 15.0

    def test_avg_hit_latency_zero_hits(self, metrics: CacheMetrics) -> None:
        """Test average hit latency with zero hits."""
        assert metrics.avg_hit_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_to_dict(self, metrics: CacheMetrics) -> None:
        """Test converting metrics to dictionary."""
        await metrics.record_hit(10.0)
        await metrics.record_miss(20.0)

        result = metrics.to_dict()
        assert result["hits"] == 1
        assert result["misses"] == 1
        assert result["total_requests"] == 2
        assert result["hit_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_reset(self, metrics: CacheMetrics) -> None:
        """Test resetting metrics."""
        await metrics.record_hit(10.0)
        await metrics.record_miss(20.0)
        await metrics.record_error()

        metrics.reset()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.errors == 0


class TestCacheKeyBuilder:
    """Tests for CacheKeyBuilder."""

    def test_market_data_key(self) -> None:
        """Test building market data key."""
        key = CacheKeyBuilder.market_data("AAPL", "quote")
        assert key == "portfolio_advisor:market:AAPL:quote"

    def test_market_data_uppercase(self) -> None:
        """Test market data key uppercases symbol."""
        key = CacheKeyBuilder.market_data("aapl", "history")
        assert key == "portfolio_advisor:market:AAPL:history"

    def test_news_key(self) -> None:
        """Test building news key."""
        key = CacheKeyBuilder.news(["AAPL", "GOOGL"])
        assert key.startswith("portfolio_advisor:news:")

    def test_news_key_with_timestamp(self) -> None:
        """Test building news key with timestamp bucket."""
        key = CacheKeyBuilder.news(["AAPL"], timestamp_bucket=1704067200)
        assert "1704067200" in key

    def test_news_key_consistent_order(self) -> None:
        """Test news key is consistent regardless of symbol order."""
        key1 = CacheKeyBuilder.news(["AAPL", "GOOGL"])
        key2 = CacheKeyBuilder.news(["GOOGL", "AAPL"])
        assert key1 == key2  # Sorted before hashing

    def test_analysis_key(self) -> None:
        """Test building analysis key."""
        key = CacheKeyBuilder.analysis("portfolio123", "v2")
        assert key == "portfolio_advisor:analysis:portfolio123:v2"

    def test_recommendation_key(self) -> None:
        """Test building recommendation key."""
        key = CacheKeyBuilder.recommendation("portfolio123", "abc123")
        assert key == "portfolio_advisor:recommendation:portfolio123:abc123"

    def test_custom_key(self) -> None:
        """Test building custom key."""
        key = CacheKeyBuilder.custom("test", "part1", "part2")
        assert key == "portfolio_advisor:test:part1:part2"


class TestSerializeDeserialize:
    """Tests for serialize and deserialize functions."""

    def test_serialize_dict(self) -> None:
        """Test serializing a dictionary."""
        data = {"key": "value", "number": 42}
        result = serialize(data)
        assert isinstance(result, str)
        assert "key" in result

    def test_deserialize_string(self) -> None:
        """Test deserializing a string."""
        data = '{"key": "value"}'
        result = deserialize(data)
        assert result == {"key": "value"}

    def test_deserialize_bytes(self) -> None:
        """Test deserializing bytes."""
        data = b'{"key": "value"}'
        result = deserialize(data)
        assert result == {"key": "value"}

    def test_round_trip(self) -> None:
        """Test serialize then deserialize returns original."""
        original = {"nested": {"data": [1, 2, 3]}}
        serialized = serialize(original)
        result = deserialize(serialized)
        assert result == original


class TestCacheManager:
    """Tests for CacheManager."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock(return_value=True)
        redis.delete = AsyncMock(return_value=1)
        redis.exists = AsyncMock(return_value=1)
        redis.ttl = AsyncMock(return_value=300)
        return redis

    @pytest.fixture
    def cache(self, mock_redis: AsyncMock) -> CacheManager:
        """Create cache manager for testing."""
        return CacheManager(mock_redis)

    @pytest.mark.asyncio
    async def test_get_returns_none_when_not_found(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get returns None when key not found."""
        mock_redis.get.return_value = None
        result = await cache.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_deserialized_value(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get returns deserialized value."""
        mock_redis.get.return_value = b'{"data": "test"}'
        result = await cache.get("test_key")
        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_get_records_hit(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get records hit in metrics."""
        mock_redis.get.return_value = b'{"data": "test"}'
        await cache.get("test_key")
        assert cache.metrics.hits == 1

    @pytest.mark.asyncio
    async def test_get_records_miss(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get records miss in metrics."""
        mock_redis.get.return_value = None
        await cache.get("test_key")
        assert cache.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_get_handles_error(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get handles Redis errors gracefully."""
        mock_redis.get.side_effect = Exception("Redis error")
        result = await cache.get("test_key")
        assert result is None
        assert cache.metrics.errors == 1

    @pytest.mark.asyncio
    async def test_set_stores_value(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test set stores serialized value."""
        result = await cache.set("test_key", {"data": "test"}, ttl=300)
        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_uses_cache_type_ttl(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test set uses TTL from cache type."""
        await cache.set(
            "test_key",
            {"data": "test"},
            cache_type=CacheType.MARKET_DATA,
        )
        # Should use market data TTL (300)
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # TTL argument

    @pytest.mark.asyncio
    async def test_set_handles_error(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test set handles Redis errors gracefully."""
        mock_redis.setex.side_effect = Exception("Redis error")
        result = await cache.set("test_key", {"data": "test"})
        assert result is False
        assert cache.metrics.errors == 1

    @pytest.mark.asyncio
    async def test_delete_removes_key(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test delete removes key."""
        result = await cache.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_not_found(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test delete returns False when key not found."""
        mock_redis.delete.return_value = 0
        result = await cache.delete("missing_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_or_compute_returns_cached(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get_or_compute returns cached value."""
        mock_redis.get.return_value = b'{"cached": true}'
        compute_fn = AsyncMock(return_value={"computed": True})

        result = await cache.get_or_compute("test_key", compute_fn, ttl=300)

        assert result == {"cached": True}
        compute_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_compute_computes_when_not_cached(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get_or_compute computes when not cached."""
        mock_redis.get.return_value = None
        compute_fn = AsyncMock(return_value={"computed": True})

        result = await cache.get_or_compute("test_key", compute_fn, ttl=300)

        assert result == {"computed": True}
        compute_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_compute_caches_result(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test get_or_compute caches computed result."""
        mock_redis.get.return_value = None
        compute_fn = AsyncMock(return_value={"computed": True})

        await cache.get_or_compute("test_key", compute_fn, ttl=300)

        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_exists(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test exists checks key existence."""
        mock_redis.exists.return_value = 1
        result = await cache.exists("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_ttl(
        self, cache: CacheManager, mock_redis: AsyncMock
    ) -> None:
        """Test ttl returns remaining TTL."""
        mock_redis.ttl.return_value = 250
        result = await cache.ttl("test_key")
        assert result == 250

    def test_get_metrics(self, cache: CacheManager) -> None:
        """Test get_metrics returns metrics dict."""
        metrics = cache.get_metrics()
        assert "hits" in metrics
        assert "misses" in metrics
        assert "hit_rate" in metrics

    @pytest.mark.asyncio
    async def test_disabled_cache_get_returns_none(
        self, mock_redis: AsyncMock
    ) -> None:
        """Test disabled cache always returns None for get."""
        cache = CacheManager(mock_redis, enabled=False)
        result = await cache.get("test_key")
        assert result is None
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_cache_set_returns_false(
        self, mock_redis: AsyncMock
    ) -> None:
        """Test disabled cache returns False for set."""
        cache = CacheManager(mock_redis, enabled=False)
        result = await cache.set("test_key", "value")
        assert result is False
        mock_redis.setex.assert_not_called()


class TestCacheManagerInvalidation:
    """Tests for cache invalidation."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create mock Redis client with scan_iter."""
        redis = AsyncMock()
        redis.delete = AsyncMock(return_value=3)

        async def mock_scan_iter(match: str):  # noqa: ARG001
            yield b"key1"
            yield b"key2"
            yield b"key3"

        redis.scan_iter = mock_scan_iter
        return redis

    @pytest.fixture
    def cache(self, mock_redis: AsyncMock) -> CacheManager:
        """Create cache manager for testing."""
        return CacheManager(mock_redis)

    @pytest.mark.asyncio
    async def test_delete_pattern(self, cache: CacheManager) -> None:
        """Test delete_pattern removes matching keys."""
        deleted = await cache.delete_pattern("portfolio_advisor:*")
        assert deleted == 3

    @pytest.mark.asyncio
    async def test_invalidate_portfolio(self, cache: CacheManager) -> None:
        """Test invalidating portfolio cache."""
        deleted = await cache.invalidate_portfolio("portfolio123")
        assert deleted == 3

    @pytest.mark.asyncio
    async def test_invalidate_symbol(self, cache: CacheManager) -> None:
        """Test invalidating symbol cache."""
        deleted = await cache.invalidate_symbol("AAPL")
        assert deleted == 3


class TestCacheManagerRegistry:
    """Tests for cache manager registry functions."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        reset_cache_manager()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        reset_cache_manager()

    def test_get_returns_none_initially(self) -> None:
        """Test get_cache_manager returns None when not set."""
        assert get_cache_manager() is None

    def test_set_and_get(self) -> None:
        """Test setting and getting cache manager."""
        mock_redis = MagicMock()
        manager = CacheManager(mock_redis)
        set_cache_manager(manager)
        assert get_cache_manager() is manager

    def test_reset_clears_manager(self) -> None:
        """Test reset_cache_manager clears the global instance."""
        mock_redis = MagicMock()
        manager = CacheManager(mock_redis)
        set_cache_manager(manager)
        reset_cache_manager()
        assert get_cache_manager() is None


class TestCacheManagerIntegration:
    """Integration tests for cache manager."""

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create mock Redis client with realistic behavior."""
        redis = AsyncMock()
        cache_store: dict[str, bytes] = {}

        async def mock_get(key: str) -> bytes | None:
            return cache_store.get(key)

        async def mock_setex(key: str, ttl: int, value: str) -> bool:  # noqa: ARG001
            cache_store[key] = value.encode()
            return True

        async def mock_delete(key: str) -> int:
            if key in cache_store:
                del cache_store[key]
                return 1
            return 0

        redis.get = mock_get
        redis.setex = mock_setex
        redis.delete = mock_delete

        return redis

    @pytest.fixture
    def cache(self, mock_redis: AsyncMock) -> CacheManager:
        """Create cache manager for testing."""
        return CacheManager(mock_redis)

    @pytest.mark.asyncio
    async def test_full_cache_cycle(self, cache: CacheManager) -> None:
        """Test full cache cycle: miss -> compute -> hit."""
        call_count = 0

        async def compute_fn() -> dict:
            nonlocal call_count
            call_count += 1
            return {"data": "computed", "count": call_count}

        key = CacheKeyBuilder.market_data("AAPL")

        # First call - cache miss, computes
        result1 = await cache.get_or_compute(key, compute_fn, ttl=300)
        assert result1 == {"data": "computed", "count": 1}
        assert call_count == 1

        # Second call - cache hit, no compute
        result2 = await cache.get_or_compute(key, compute_fn, ttl=300)
        assert result2 == {"data": "computed", "count": 1}
        assert call_count == 1  # Not incremented

        # Check metrics
        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 1
        assert cache.metrics.hit_rate == 50.0

    @pytest.mark.asyncio
    async def test_cache_invalidation_forces_recompute(
        self, cache: CacheManager
    ) -> None:
        """Test cache invalidation forces recomputation."""
        call_count = 0

        async def compute_fn() -> dict:
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        key = CacheKeyBuilder.market_data("AAPL")

        # First call
        await cache.get_or_compute(key, compute_fn, ttl=300)
        assert call_count == 1

        # Delete and recompute
        await cache.delete(key)
        result = await cache.get_or_compute(key, compute_fn, ttl=300)
        assert result == {"count": 2}
        assert call_count == 2
