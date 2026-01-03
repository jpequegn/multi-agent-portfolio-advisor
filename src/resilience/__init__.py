"""Resilience patterns for fault tolerance.

This module contains:
- Circuit breaker pattern for preventing cascade failures
- Rate limiting for API protection and cost management
- Configuration for different service types (LLM, API, market data)
"""

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
from src.resilience.rate_limiter import (
    DEFAULT_RATE_LIMIT_CONFIG,
    CostLimitExceeded,
    CostTracker,
    LimitType,
    RateLimitConfig,
    RateLimitExceeded,
    RateLimitHeaders,
    RateLimiter,
    TokenBucket,
    get_rate_limiter,
    reset_rate_limiter,
)

__all__ = [
    # Circuit breaker classes
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    # Circuit breaker configurations
    "API_CIRCUIT_CONFIG",
    "LLM_CIRCUIT_CONFIG",
    "MARKET_DATA_CIRCUIT_CONFIG",
    # Circuit breaker decorators
    "api_circuit_breaker",
    "circuit_breaker",
    "llm_circuit_breaker",
    "market_data_circuit_breaker",
    # Circuit breaker registry
    "get_all_circuit_breakers",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
    # Rate limiter classes
    "CostLimitExceeded",
    "CostTracker",
    "LimitType",
    "RateLimitConfig",
    "RateLimitExceeded",
    "RateLimitHeaders",
    "RateLimiter",
    "TokenBucket",
    # Rate limiter configuration
    "DEFAULT_RATE_LIMIT_CONFIG",
    # Rate limiter functions
    "get_rate_limiter",
    "reset_rate_limiter",
]
