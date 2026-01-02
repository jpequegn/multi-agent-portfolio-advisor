"""Resilience patterns for fault tolerance.

This module contains:
- Circuit breaker pattern for preventing cascade failures
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

__all__ = [
    # Core classes
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    # Configurations
    "API_CIRCUIT_CONFIG",
    "LLM_CIRCUIT_CONFIG",
    "MARKET_DATA_CIRCUIT_CONFIG",
    # Decorators
    "api_circuit_breaker",
    "circuit_breaker",
    "llm_circuit_breaker",
    "market_data_circuit_breaker",
    # Registry functions
    "get_all_circuit_breakers",
    "get_circuit_breaker",
    "reset_all_circuit_breakers",
]
