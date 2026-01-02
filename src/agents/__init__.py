"""Agent implementations for portfolio analysis.

This module contains specialized agents:
- ResearchAgent: Gathers market data and news
- AnalysisAgent: Analyzes portfolio risk and performance
- RecommendationAgent: Generates trade recommendations
"""

from src.agents.analysis import (
    AnalysisAgent,
    AnalysisOutput,
    AttributionResult,
    BenchmarkComparison,
    CorrelationResult,
    RiskMetrics,
)
from src.agents.base import AgentState, BaseAgent
from src.agents.recommendation import (
    ComplianceResult,
    ExecutionCosts,
    RecommendationAgent,
    RecommendationOutput,
    TaxImpact,
    Trade,
)
from src.agents.research import (
    NewsItem,
    ResearchAgent,
    ResearchOutput,
    SymbolData,
)

__all__ = [
    "AgentState",
    "AnalysisAgent",
    "AnalysisOutput",
    "AttributionResult",
    "BaseAgent",
    "BenchmarkComparison",
    "ComplianceResult",
    "CorrelationResult",
    "ExecutionCosts",
    "NewsItem",
    "RecommendationAgent",
    "RecommendationOutput",
    "ResearchAgent",
    "ResearchOutput",
    "RiskMetrics",
    "SymbolData",
    "TaxImpact",
    "Trade",
]
