"""Prompt caching module for Claude API optimization.

This module provides utilities for implementing Claude's prompt caching
to reduce costs and latency for repeated system prompts and context.

Key features:
- CachedPromptBuilder: Build prompts with optimal cache breakpoints
- CacheMetrics: Track cache hits/misses and cost savings
- CacheObservabilityReporter: Report metrics to Langfuse
"""

from src.caching.metrics import CacheMetrics, CacheMetricsCollector
from src.caching.observability import (
    CacheObservabilityConfig,
    CacheObservabilityReporter,
    create_cache_reporter,
)
from src.caching.prompt_cache import (
    CacheableContent,
    CacheControl,
    CachedPromptBuilder,
    build_cached_system_prompt,
    build_cached_tool_definitions,
    estimate_cache_tokens,
    is_cacheable,
)

__all__ = [
    # Prompt building
    "CacheControl",
    "CacheableContent",
    "CachedPromptBuilder",
    "build_cached_system_prompt",
    "build_cached_tool_definitions",
    "estimate_cache_tokens",
    "is_cacheable",
    # Metrics
    "CacheMetrics",
    "CacheMetricsCollector",
    # Observability
    "CacheObservabilityConfig",
    "CacheObservabilityReporter",
    "create_cache_reporter",
]
