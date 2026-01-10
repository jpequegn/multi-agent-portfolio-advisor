"""Application configuration settings.

This module provides centralized configuration management using environment
variables with sensible defaults.
"""

import os
from dataclasses import dataclass


def _get_bool_env(name: str, default: bool = False) -> bool:
    """Get a boolean value from environment variable.

    Args:
        name: Environment variable name.
        default: Default value if not set.

    Returns:
        Boolean value from environment.
    """
    value = os.getenv(name, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


@dataclass
class Settings:
    """Application settings loaded from environment variables.

    Attributes:
        USE_PROMPT_CACHING: Enable Claude prompt caching for cost reduction.
        LANGFUSE_HOST: Langfuse server host URL.
        LANGFUSE_PUBLIC_KEY: Langfuse public API key.
        LANGFUSE_SECRET_KEY: Langfuse secret API key.
        POLYGON_API_KEY: Polygon.io API key for market data.
        REDIS_URL: Redis connection URL for caching.
        DEFAULT_MODEL: Default Claude model to use.
        LOG_LEVEL: Logging level.
    """

    # Prompt caching
    USE_PROMPT_CACHING: bool = True  # Default to enabled for cost savings

    # Observability
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None

    # Data sources
    POLYGON_API_KEY: str | None = None
    REDIS_URL: str = "redis://localhost:6379"

    # Model settings
    DEFAULT_MODEL: str = "claude-sonnet-4-20250514"

    # Logging
    LOG_LEVEL: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            Settings instance populated from environment.
        """
        return cls(
            USE_PROMPT_CACHING=_get_bool_env("USE_PROMPT_CACHING", default=True),
            LANGFUSE_HOST=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY"),
            LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY"),
            POLYGON_API_KEY=os.getenv("POLYGON_API_KEY"),
            REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379"),
            DEFAULT_MODEL=os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514"),
            LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
        )


# Global settings instance
settings = Settings.from_env()
