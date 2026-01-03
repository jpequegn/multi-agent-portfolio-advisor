"""FastAPI routes for the Portfolio Advisor API.

This module contains:
- Portfolio analysis endpoints
- Health check endpoints
- Request/response models
"""

from src.api.health import (
    DEFAULT_HEALTH_CONFIG,
    ComponentCheck,
    ExternalAPIHealthChecker,
    HealthCheckConfig,
    HealthChecker,
    HealthCheckResult,
    HealthService,
    HealthStatus,
    LangfuseHealthChecker,
    LLMHealthChecker,
    PostgreSQLHealthChecker,
    RedisHealthChecker,
    ServiceStatus,
    create_health_service,
    get_health_service,
    reset_health_service,
    set_health_service,
)
from src.api.routes import (
    AnalysisOutput,
    AnalysisResponse,
    AnalysisSummary,
    ErrorResponse,
    PortfolioRequest,
    PositionRequest,
    RecommendationOutput,
    ResearchOutput,
    app,
    create_app,
)

__all__ = [
    # Health check classes
    "ComponentCheck",
    "ExternalAPIHealthChecker",
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthChecker",
    "HealthService",
    "LangfuseHealthChecker",
    "LLMHealthChecker",
    "PostgreSQLHealthChecker",
    "RedisHealthChecker",
    # Health check enums
    "HealthStatus",
    "ServiceStatus",
    # Configuration
    "DEFAULT_HEALTH_CONFIG",
    # Health service factory and global instance
    "create_health_service",
    "get_health_service",
    "reset_health_service",
    "set_health_service",
    # Request models
    "PortfolioRequest",
    "PositionRequest",
    # Response models
    "AnalysisOutput",
    "AnalysisResponse",
    "AnalysisSummary",
    "ErrorResponse",
    "RecommendationOutput",
    "ResearchOutput",
    # App factory and instance
    "app",
    "create_app",
]
