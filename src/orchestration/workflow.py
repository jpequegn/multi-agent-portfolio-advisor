"""LangGraph workflow for multi-agent portfolio analysis.

This module defines the main workflow that orchestrates all agents
to analyze portfolios and generate recommendations.
"""

from typing import Any, Literal, cast

import structlog
from langgraph.graph import END, StateGraph

from src.agents import AnalysisAgent, RecommendationAgent, ResearchAgent
from src.agents.base import AgentState
from src.orchestration.state import (
    AgentName,
    PortfolioState,
    WorkflowStatus,
    mark_state_completed,
    mark_state_failed,
    update_state_for_agent,
    update_state_with_result,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Agent Node Functions
# ============================================================================


async def research_node(state: PortfolioState) -> PortfolioState:
    """Execute the research agent.

    Gathers market data and news for the portfolio symbols.

    Args:
        state: Current workflow state.

    Returns:
        Updated state with research results.
    """
    logger.info("research_node_started", workflow_id=state.get("workflow_id"))

    # Mark state as running research
    state = update_state_for_agent(state, AgentName.RESEARCH)

    try:
        # Create agent and prepare input state
        agent = ResearchAgent()
        agent_state = AgentState(
            messages=[],
            context={
                "symbols": state.get("symbols", []),
                "portfolio": state.get("portfolio", {}),
                "user_request": state.get("user_request", ""),
            },
            errors=[],
        )

        # Execute agent
        result_state = await agent(agent_state)

        # Extract research output from agent result
        research_output = result_state.context.get("research_output", {})

        # If no research output, build from context
        if not research_output:
            research_output = {
                "market_data": result_state.context.get("market_data", {}),
                "news": result_state.context.get("news_items", []),
                "summary": result_state.context.get("summary", ""),
                "symbols_researched": state.get("symbols", []),
                "errors": result_state.errors,
            }

        # Update workflow state with results
        state = update_state_with_result(state, AgentName.RESEARCH, research_output)

        logger.info(
            "research_node_completed",
            workflow_id=state.get("workflow_id"),
            symbols_count=len(state.get("symbols", [])),
        )

    except Exception as e:
        logger.error(
            "research_node_failed",
            workflow_id=state.get("workflow_id"),
            error=str(e),
        )
        state["errors"].append(f"Research failed: {e!s}")

    return state


async def analysis_node(state: PortfolioState) -> PortfolioState:
    """Execute the analysis agent.

    Analyzes portfolio risk and performance using research data.

    Args:
        state: Current workflow state with research results.

    Returns:
        Updated state with analysis results.
    """
    logger.info("analysis_node_started", workflow_id=state.get("workflow_id"))

    # Mark state as running analysis
    state = update_state_for_agent(state, AgentName.ANALYSIS)

    try:
        # Create agent and prepare input state
        agent = AnalysisAgent()
        agent_state = AgentState(
            messages=[],
            context={
                "symbols": state.get("symbols", []),
                "portfolio": state.get("portfolio", {}),
                "research": state.get("research", {}),
                "user_request": state.get("user_request", ""),
            },
            errors=[],
        )

        # Execute agent
        result_state = await agent(agent_state)

        # Extract analysis output from agent result
        analysis_output = result_state.context.get("analysis_output", {})

        # If no analysis output, build from context
        if not analysis_output:
            analysis_output = {
                "risk_metrics": result_state.context.get("risk_metrics", {}),
                "correlations": result_state.context.get("correlations", {}),
                "benchmark_comparison": result_state.context.get(
                    "benchmark_comparison", {}
                ),
                "attribution": result_state.context.get("attribution", {}),
                "recommendations": result_state.context.get("recommendations", []),
                "summary": result_state.context.get("summary", ""),
                "errors": result_state.errors,
            }

        # Update workflow state with results
        state = update_state_with_result(state, AgentName.ANALYSIS, analysis_output)

        logger.info(
            "analysis_node_completed",
            workflow_id=state.get("workflow_id"),
        )

    except Exception as e:
        logger.error(
            "analysis_node_failed",
            workflow_id=state.get("workflow_id"),
            error=str(e),
        )
        state["errors"].append(f"Analysis failed: {e!s}")

    return state


async def recommendation_node(state: PortfolioState) -> PortfolioState:
    """Execute the recommendation agent.

    Generates trade recommendations based on research and analysis.

    Args:
        state: Current workflow state with research and analysis results.

    Returns:
        Updated state with recommendations.
    """
    logger.info("recommendation_node_started", workflow_id=state.get("workflow_id"))

    # Mark state as running recommendation
    state = update_state_for_agent(state, AgentName.RECOMMENDATION)

    try:
        # Create agent and prepare input state
        agent = RecommendationAgent()
        agent_state = AgentState(
            messages=[],
            context={
                "symbols": state.get("symbols", []),
                "portfolio": state.get("portfolio", {}),
                "research": state.get("research", {}),
                "analysis": state.get("analysis", {}),
                "user_request": state.get("user_request", ""),
            },
            errors=[],
        )

        # Execute agent
        result_state = await agent(agent_state)

        # Extract recommendation output from agent result
        recommendation_output = result_state.context.get("recommendation_output", {})

        # If no recommendation output, build from context
        if not recommendation_output:
            trades = result_state.context.get("trades", [])
            recommendation_output = {
                "trades": trades,
                "tax_impact": result_state.context.get("tax_impact", {}),
                "execution_costs": result_state.context.get("execution_costs", {}),
                "compliance": result_state.context.get("compliance", {}),
                "summary": result_state.context.get("summary", ""),
                "total_trades": len(trades),
                "buy_count": sum(1 for t in trades if t.get("action") == "buy"),
                "sell_count": sum(1 for t in trades if t.get("action") == "sell"),
                "hold_count": sum(1 for t in trades if t.get("action") == "hold"),
                "errors": result_state.errors,
            }

        # Update workflow state with results
        state = update_state_with_result(
            state, AgentName.RECOMMENDATION, recommendation_output
        )

        logger.info(
            "recommendation_node_completed",
            workflow_id=state.get("workflow_id"),
            trades_count=recommendation_output.get("total_trades", 0),
        )

    except Exception as e:
        logger.error(
            "recommendation_node_failed",
            workflow_id=state.get("workflow_id"),
            error=str(e),
        )
        state["errors"].append(f"Recommendation failed: {e!s}")

    return state


async def error_handler_node(state: PortfolioState) -> PortfolioState:
    """Handle workflow errors.

    Logs errors and marks the workflow as failed.

    Args:
        state: Current workflow state with errors.

    Returns:
        Updated state marked as failed.
    """
    errors = state.get("errors", [])
    workflow_id = state.get("workflow_id")

    logger.error(
        "workflow_error_handler",
        workflow_id=workflow_id,
        error_count=len(errors),
        errors=errors,
    )

    # Get the most recent error as the primary failure reason
    error_message = errors[-1] if errors else "Unknown error"
    state = mark_state_failed(state, error_message)

    return state


async def finalize_node(state: PortfolioState) -> PortfolioState:
    """Finalize the workflow.

    Marks the workflow as completed successfully.

    Args:
        state: Current workflow state with all results.

    Returns:
        Updated state marked as completed.
    """
    workflow_id = state.get("workflow_id")

    logger.info(
        "workflow_finalizing",
        workflow_id=workflow_id,
    )

    state = mark_state_completed(state)

    logger.info(
        "workflow_completed",
        workflow_id=workflow_id,
        has_research=state.get("research") is not None,
        has_analysis=state.get("analysis") is not None,
        has_recommendation=state.get("recommendation") is not None,
    )

    return state


# ============================================================================
# Routing Functions
# ============================================================================


def route_after_research(
    state: PortfolioState,
) -> Literal["analysis", "error_handler"]:
    """Route after research agent based on errors.

    Args:
        state: Current workflow state.

    Returns:
        Next node to execute.
    """
    errors = state.get("errors", [])
    research = state.get("research")

    # Check for critical errors
    if errors and any("critical" in e.lower() for e in errors):
        logger.warning("routing_to_error_handler", reason="critical_error_in_research")
        return "error_handler"

    # Check if research produced output
    if not research:
        logger.warning("routing_to_error_handler", reason="no_research_output")
        return "error_handler"

    logger.debug("routing_to_analysis")
    return "analysis"


def route_after_analysis(
    state: PortfolioState,
) -> Literal["recommendation", "error_handler"]:
    """Route after analysis agent based on errors.

    Args:
        state: Current workflow state.

    Returns:
        Next node to execute.
    """
    errors = state.get("errors", [])
    analysis = state.get("analysis")

    # Check for critical errors
    if errors and any("critical" in e.lower() for e in errors):
        logger.warning("routing_to_error_handler", reason="critical_error_in_analysis")
        return "error_handler"

    # Check if analysis produced output
    if not analysis:
        logger.warning("routing_to_error_handler", reason="no_analysis_output")
        return "error_handler"

    logger.debug("routing_to_recommendation")
    return "recommendation"


def route_after_recommendation(
    state: PortfolioState,
) -> Literal["finalize", "error_handler"]:
    """Route after recommendation agent based on errors.

    Args:
        state: Current workflow state.

    Returns:
        Next node to execute.
    """
    errors = state.get("errors", [])

    # Check for critical errors
    if errors and any("critical" in e.lower() for e in errors):
        logger.warning(
            "routing_to_error_handler", reason="critical_error_in_recommendation"
        )
        return "error_handler"

    # Even without recommendations, we can finalize (might just be holds)
    logger.debug("routing_to_finalize")
    return "finalize"


# ============================================================================
# Workflow Factory
# ============================================================================


def create_workflow() -> Any:
    """Create and compile the portfolio advisor workflow.

    The workflow executes agents in sequence:
    Research → Analysis → Recommendation → Finalize

    With error handling at each step that can route to an error handler.

    Returns:
        Compiled LangGraph workflow ready for execution.
    """
    logger.info("creating_workflow")

    # Create the state graph
    workflow = StateGraph(PortfolioState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("error_handler", error_handler_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("research")

    # Add conditional edges after each agent
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "analysis": "analysis",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "recommendation": "recommendation",
            "error_handler": "error_handler",
        },
    )

    workflow.add_conditional_edges(
        "recommendation",
        route_after_recommendation,
        {
            "finalize": "finalize",
            "error_handler": "error_handler",
        },
    )

    # Terminal edges
    workflow.add_edge("finalize", END)
    workflow.add_edge("error_handler", END)

    # Compile and return
    compiled = workflow.compile()
    logger.info("workflow_compiled")

    return compiled


# ============================================================================
# Workflow Executor
# ============================================================================


async def run_workflow(
    state: PortfolioState,
    workflow: Any | None = None,
) -> PortfolioState:
    """Execute the portfolio advisor workflow.

    Args:
        state: Initial workflow state with portfolio and request.
        workflow: Optional pre-compiled workflow. If not provided,
            creates a new one.

    Returns:
        Final workflow state with all results.
    """
    if workflow is None:
        workflow = create_workflow()

    workflow_id = state.get("workflow_id")
    logger.info(
        "workflow_execution_started",
        workflow_id=workflow_id,
        symbols=state.get("symbols", []),
    )

    # Update status to running
    state["status"] = WorkflowStatus.RUNNING.value

    try:
        # Execute the workflow
        result = cast(PortfolioState, await workflow.ainvoke(state))

        logger.info(
            "workflow_execution_finished",
            workflow_id=workflow_id,
            status=result.get("status"),
        )

        return result

    except Exception as e:
        logger.error(
            "workflow_execution_error",
            workflow_id=workflow_id,
            error=str(e),
        )
        state = mark_state_failed(state, f"Workflow execution failed: {e!s}")
        return state


# Create a default workflow instance for convenience
default_workflow = create_workflow()
