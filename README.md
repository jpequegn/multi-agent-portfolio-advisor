# Multi-Agent Portfolio Advisor

Multi-agent system for portfolio analysis with full observability. A learning project demonstrating AI agent orchestration and monitoring patterns.

## Overview

**Difficulty**: Hard | **Time**: 6-8 weeks | **Primary Learning**: Multi-agent orchestration & observability

This project builds a multi-agent system where specialized agents collaborate to analyze portfolios: Research Agent → Analysis Agent → Recommendation Agent. The same patterns apply to fitness optimization (nutrition + training + recovery agents).

### What You'll Learn
- Multi-agent orchestration patterns with LangGraph
- Production observability setup with Langfuse
- Cost attribution and optimization
- Debugging agent failures and reasoning chains

### Why This Matters
Multi-agent systems are the future of complex AI applications. Understanding how to observe and debug them is rare and valuable in the market.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Claude 3.5 Sonnet | Agent reasoning |
| Orchestration | LangGraph | Multi-agent workflow |
| Observability | Langfuse | Tracing, cost tracking |
| State | PostgreSQL + Redis | Agent memory & cache |
| API | FastAPI | External interface |
| Language | Python 3.11+ | Implementation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER REQUEST                                │
│  "Analyze my portfolio for risk and suggest rebalancing"        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (LangGraph)                      │
├─────────────────────────────────────────────────────────────────┤
│  - Request routing                                               │
│  - Agent coordination                                            │
│  - State management                                              │
│  - Error handling & recovery                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   RESEARCH    │ │   ANALYSIS    │ │ RECOMMENDATION│
│    AGENT      │ │    AGENT      │ │    AGENT      │
├───────────────┤ ├───────────────┤ ├───────────────┤
│ - Market data │ │ - Risk calc   │ │ - Rebalancing │
│ - News fetch  │ │ - Correlation │ │ - Trade ideas │
│ - SEC filings │ │ - Attribution │ │ - Tax impact  │
│ - Analyst est │ │ - Benchmarks  │ │ - Execution   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        │    ┌────────────┼────────────┐    │
        │    │            │            │    │
        ▼    ▼            ▼            ▼    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SHARED TOOL LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Market Data API │ Calculator │ Document Store │ News Search   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OBSERVABILITY (Langfuse)                       │
├─────────────────────────────────────────────────────────────────┤
│  - Full trace per request                                        │
│  - Agent-level spans                                             │
│  - Tool call logging                                             │
│  - Token usage & cost attribution                                │
│  - Latency breakdown                                             │
│  - Error tracking & debugging                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Definitions

### Research Agent
**Role**: Gather relevant market data and information

**Tools**:
- `get_market_data(symbols)` - Fetch current prices, volume
- `search_news(query)` - Search financial news
- `get_sec_filings(symbol)` - Fetch recent SEC filings
- `get_analyst_estimates(symbol)` - Fetch consensus estimates

**Output**: Structured research document with sources

### Analysis Agent
**Role**: Analyze portfolio risk and performance

**Tools**:
- `calculate_risk_metrics(portfolio)` - VaR, Sharpe, Beta
- `correlation_analysis(portfolio)` - Cross-asset correlations
- `benchmark_comparison(portfolio, benchmark)` - Relative performance
- `attribution_analysis(portfolio)` - Return attribution

**Output**: Risk report with quantitative metrics

### Recommendation Agent
**Role**: Generate actionable recommendations

**Tools**:
- `generate_rebalancing(portfolio, targets)` - Rebalancing trades
- `estimate_tax_impact(trades)` - Tax lot analysis
- `estimate_execution_cost(trades)` - Transaction cost estimate
- `validate_compliance(trades)` - Check against constraints

**Output**: Actionable recommendation with trade list

---

## Implementation Plan

### Phase 1: Single Agent Foundation (Week 1-2)

#### Week 1: Infrastructure Setup
- [ ] Set up Python project with LangGraph
- [ ] Configure Langfuse for observability
- [ ] Set up PostgreSQL for state persistence
- [ ] Create base agent class with tracing
- [ ] Implement tool interface pattern

**Deliverable**: Working infrastructure with single traced agent

#### Week 2: Research Agent
- [ ] Implement Research Agent
- [ ] Build market data tool (mock or real API)
- [ ] Build news search tool
- [ ] Add comprehensive Langfuse tracing
- [ ] Write tests for agent behavior

**Deliverable**: Research Agent with full observability

### Phase 2: Multi-Agent Orchestration (Week 3-4)

#### Week 3: Additional Agents
- [ ] Implement Analysis Agent
- [ ] Implement Recommendation Agent
- [ ] Define agent communication protocol
- [ ] Implement state passing between agents

**Deliverable**: Three working agents (not yet orchestrated)

#### Week 4: Orchestration
- [ ] Build LangGraph workflow
  - Define state schema
  - Create agent nodes
  - Define edges and conditions
- [ ] Implement error handling
  - Retry logic
  - Fallback strategies
  - Human escalation
- [ ] Add end-to-end tracing

**Deliverable**: Full multi-agent workflow

### Phase 3: Observability Deep Dive (Week 5-6)

#### Week 5: Comprehensive Tracing
- [ ] Implement nested span hierarchy
  - Session → Request → Agent → Tool
- [ ] Add custom attributes
  - Agent reasoning traces
  - Tool input/output
  - Decision points
- [ ] Build cost attribution
  - Per-agent token usage
  - Per-request total cost
- [ ] Create observability dashboard

**Deliverable**: Production-grade observability

#### Week 6: Debugging & Analysis
- [ ] Implement failure analysis
  - Capture failure patterns
  - Root cause identification
- [ ] Build replay capability
  - Replay failed requests
  - Debug with full context
- [ ] Add performance monitoring
  - Latency percentiles
  - Bottleneck identification
- [ ] Document debugging workflows

**Deliverable**: Debugging toolkit and documentation

### Phase 4: Production Hardening (Week 7-8)

#### Week 7: Reliability
- [ ] Implement circuit breakers
- [ ] Add rate limiting
- [ ] Build caching layer
- [ ] Implement graceful degradation
- [ ] Add health checks

**Deliverable**: Production-ready reliability

#### Week 8: Documentation & Polish
- [ ] API documentation
- [ ] Architecture documentation
- [ ] Runbook for operations
- [ ] Performance optimization
- [ ] Final testing

**Deliverable**: Complete, documented system

---

## Key Files Structure

```
multi-agent-portfolio-advisor/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Base agent class
│   │   ├── research.py          # Research Agent
│   │   ├── analysis.py          # Analysis Agent
│   │   └── recommendation.py    # Recommendation Agent
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── market_data.py       # Market data fetching
│   │   ├── news_search.py       # News search
│   │   ├── risk_metrics.py      # Risk calculations
│   │   └── trade_generator.py   # Trade generation
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── workflow.py          # LangGraph workflow
│   │   ├── state.py             # State management
│   │   └── errors.py            # Error handling
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── tracing.py           # Langfuse setup
│   │   ├── metrics.py           # Custom metrics
│   │   └── dashboards.py        # Dashboard configs
│   └── api/
│       ├── __init__.py
│       └── routes.py            # FastAPI routes
├── tests/
│   ├── test_agents/
│   ├── test_tools/
│   └── test_orchestration/
└── docs/
    ├── ARCHITECTURE.md
    ├── DEBUGGING.md
    └── RUNBOOK.md
```

---

## Sample Code Snippets

### LangGraph Workflow
```python
from langgraph.graph import StateGraph, END
from langfuse.decorators import observe

class PortfolioState(TypedDict):
    portfolio: dict
    research: Optional[dict]
    analysis: Optional[dict]
    recommendation: Optional[dict]
    error: Optional[str]

def create_workflow():
    workflow = StateGraph(PortfolioState)

    # Add agent nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("recommendation", recommendation_agent)
    workflow.add_node("error_handler", handle_error)

    # Define edges
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "recommendation")
    workflow.add_edge("recommendation", END)

    # Conditional edges for error handling
    workflow.add_conditional_edges(
        "research",
        lambda s: "error_handler" if s.get("error") else "analysis"
    )

    workflow.set_entry_point("research")
    return workflow.compile()

@observe(name="research_agent")
def research_agent(state: PortfolioState) -> PortfolioState:
    """Gather market research for portfolio."""
    # Implementation...
    return {**state, "research": research_results}
```

### Observability Setup
```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe(name="portfolio_analysis")
def analyze_portfolio(portfolio: dict) -> dict:
    # Add custom attributes
    langfuse_context.update_current_observation(
        metadata={"portfolio_size": len(portfolio["holdings"])}
    )

    # Track cost
    langfuse_context.update_current_trace(
        tags=["portfolio-analysis"],
        user_id=portfolio.get("user_id")
    )

    # Run workflow
    result = workflow.invoke({"portfolio": portfolio})

    # Log metrics
    langfuse_context.score_current_trace(
        name="completion_success",
        value=1.0 if not result.get("error") else 0.0
    )

    return result
```

### Cost Attribution
```python
def calculate_request_cost(trace_id: str) -> dict:
    """Calculate cost breakdown for a request."""
    trace = langfuse.get_trace(trace_id)

    costs = {"total": 0, "by_agent": {}}

    for span in trace.observations:
        if span.type == "generation":
            cost = calculate_token_cost(
                span.usage.input_tokens,
                span.usage.output_tokens,
                span.model
            )
            costs["total"] += cost
            agent = span.metadata.get("agent", "unknown")
            costs["by_agent"][agent] = costs["by_agent"].get(agent, 0) + cost

    return costs
```

---

## Success Criteria

### Functional
- [ ] Multi-agent workflow completes for 95%+ of requests
- [ ] Each agent produces valid, structured output
- [ ] Error recovery handles common failure modes

### Observability
- [ ] Full trace visibility for all requests
- [ ] Cost attribution accurate to agent level
- [ ] Latency breakdown available per component
- [ ] Failed requests have debugging context

### Performance
- [ ] End-to-end latency < 30 seconds
- [ ] Cost per analysis < $0.50
- [ ] 99% reliability (with retries)

---

## Fitness Variant

The same architecture applies to a **Training Plan Optimizer**:

| Finance Agent | Fitness Equivalent |
|---------------|-------------------|
| Research Agent | Training History Agent (past workouts, performance) |
| Analysis Agent | Fitness Analysis Agent (training load, recovery) |
| Recommendation Agent | Plan Recommendation Agent (next week's schedule) |

Tools would include:
- `get_training_history(athlete_id)` - Past workouts
- `calculate_training_load(workouts)` - TSS, ATL, CTL
- `analyze_recovery(athlete_id)` - HRV, sleep, readiness
- `generate_training_plan(targets)` - Weekly schedule

---

## Learning Outcomes

After completing this project, you will be able to:

1. **Design** multi-agent systems with clear responsibilities
2. **Implement** LangGraph workflows for complex orchestration
3. **Instrument** comprehensive observability for AI systems
4. **Debug** agent failures using trace analysis
5. **Optimize** costs in multi-agent systems

---

## Resources

- [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Langfuse Tracing Guide](https://langfuse.com/docs/tracing)
- [Building Production AI Agents](https://www.anthropic.com/research/building-effective-agents)
- [OpenTelemetry for AI](https://opentelemetry.io/blog/2025/ai-agent-observability/)

---

## License

MIT License - See LICENSE file for details.
