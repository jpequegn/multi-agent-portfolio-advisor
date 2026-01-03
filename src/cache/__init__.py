"""Intelligent caching layer for cost and latency reduction.

This module contains:
- CacheManager for Redis-based caching
- CacheKeyBuilder for consistent key generation
- TTL configurations for different data types
- Cache metrics tracking
"""

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

__all__ = [
    # Core classes
    "CacheConfig",
    "CacheKeyBuilder",
    "CacheManager",
    "CacheMetrics",
    "CacheType",
    # Configuration
    "DEFAULT_CACHE_CONFIG",
    # Utilities
    "deserialize",
    "serialize",
    # Global instance functions
    "get_cache_manager",
    "reset_cache_manager",
    "set_cache_manager",
]
