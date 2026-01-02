"""Multi-agent orchestration using LangGraph.

This module contains:
- Workflow definitions
- State management
- Error handling and recovery
"""

from src.orchestration.state import (
    AgentName,
    AnalysisOutputDict,
    Portfolio,
    PortfolioState,
    Position,
    RecommendationOutputDict,
    ResearchOutputDict,
    StatePersistence,
    StateValidationError,
    WorkflowStatus,
    create_initial_state,
    get_state_summary,
    mark_state_completed,
    mark_state_failed,
    state_from_json,
    state_to_json,
    update_state_for_agent,
    update_state_with_result,
    validate_state,
    validate_state_or_raise,
)
from src.orchestration.workflow import (
    analysis_node,
    create_workflow,
    default_workflow,
    error_handler_node,
    finalize_node,
    recommendation_node,
    research_node,
    route_after_analysis,
    route_after_recommendation,
    route_after_research,
    run_workflow,
)

__all__ = [
    # State exports
    "AgentName",
    "AnalysisOutputDict",
    "Portfolio",
    "PortfolioState",
    "Position",
    "RecommendationOutputDict",
    "ResearchOutputDict",
    "StatePersistence",
    "StateValidationError",
    "WorkflowStatus",
    "create_initial_state",
    "get_state_summary",
    "mark_state_completed",
    "mark_state_failed",
    "state_from_json",
    "state_to_json",
    "update_state_for_agent",
    "update_state_with_result",
    "validate_state",
    "validate_state_or_raise",
    # Workflow exports
    "analysis_node",
    "create_workflow",
    "default_workflow",
    "error_handler_node",
    "finalize_node",
    "recommendation_node",
    "research_node",
    "route_after_analysis",
    "route_after_recommendation",
    "route_after_research",
    "run_workflow",
]
