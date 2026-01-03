"""Resilience patterns for fault tolerance.

This module contains:
- Circuit breaker pattern for preventing cascade failures
- Rate limiting for API protection and cost management
- Graceful degradation for service continuity
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
    # Degradation classes
    "ComponentHealth",
    "ComponentStatus",
    "ComponentType",
    "DegradationLevel",
    "DegradationManager",
    "DegradationMetrics",
    "DegradationStrategy",
    "DegradedResponse",
    "FallbackChain",
    # Degradation utilities
    "with_fallback",
    # Degradation global instance
    "get_degradation_manager",
    "reset_degradation_manager",
    "set_degradation_manager",
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
