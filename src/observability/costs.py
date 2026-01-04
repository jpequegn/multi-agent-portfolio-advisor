"""Cost attribution system for observability.

This module provides comprehensive cost tracking with per-agent attribution,
enabling cost monitoring, alerts, and reporting for LLM usage.

Features:
- Token usage tracking per LLM call
- Cost attribution to agents
- Per-request cost calculation
- Cost alerts and thresholds
- Cost reports and aggregation
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Model Pricing
# ============================================================================


class ModelProvider(Enum):
    """LLM model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a specific model.

    Attributes:
        model_id: Model identifier.
        provider: Model provider.
        input_cost_per_million: Cost per million input tokens.
        output_cost_per_million: Cost per million output tokens.
        context_window: Maximum context window size.
    """

    model_id: str
    provider: ModelProvider
    input_cost_per_million: float
    output_cost_per_million: float
    context_window: int = 200_000

    @property
    def input_cost_per_token(self) -> float:
        """Cost per single input token."""
        return self.input_cost_per_million / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        """Cost per single output token."""
        return self.output_cost_per_million / 1_000_000

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Total cost in dollars.
        """
        input_cost = input_tokens * self.input_cost_per_token
        output_cost = output_tokens * self.output_cost_per_token
        return input_cost + output_cost


# Standard model pricing (as of 2024)
MODEL_PRICING: dict[str, ModelPricing] = {
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_id="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        model_id="claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        model_id="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
    ),
    "claude-3-opus-20240229": ModelPricing(
        model_id="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    # OpenAI GPT models
    "gpt-4-turbo": ModelPricing(
        model_id="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
        context_window=128_000,
    ),
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        provider=ModelProvider.OPENAI,
        input_cost_per_million=5.00,
        output_cost_per_million=15.00,
        context_window=128_000,
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        context_window=128_000,
    ),
}

# Default pricing for unknown models
DEFAULT_PRICING = ModelPricing(
    model_id="unknown",
    provider=ModelProvider.ANTHROPIC,
    input_cost_per_million=3.00,
    output_cost_per_million=15.00,
)


def get_model_pricing(model_id: str) -> ModelPricing:
    """Get pricing for a model.

    Args:
        model_id: Model identifier.

    Returns:
        ModelPricing for the model, or default if not found.
    """
    # Try exact match first
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]

    # Try partial match (for versioned model names)
    for key, pricing in MODEL_PRICING.items():
        if key in model_id or model_id in key:
            return pricing

    logger.warning("unknown_model_pricing", model_id=model_id)
    return DEFAULT_PRICING


# ============================================================================
# Token Usage
# ============================================================================


@dataclass
class TokenUsage:
    """Token usage for an LLM call.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: Model used for the call.
        agent_name: Agent that made the call.
        timestamp: When the call was made.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    agent_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def calculate_cost(self) -> float:
        """Calculate cost for this usage.

        Returns:
            Cost in dollars.
        """
        pricing = get_model_pricing(self.model)
        return pricing.calculate_cost(self.input_tokens, self.output_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "agent_name": self.agent_name,
            "cost": self.calculate_cost(),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Cost Breakdown
# ============================================================================


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a request.

    Attributes:
        trace_id: Trace identifier.
        workflow_id: Workflow identifier.
        total_cost: Total cost in dollars.
        by_agent: Cost breakdown by agent.
        by_model: Cost breakdown by model.
        input_tokens: Total input tokens.
        output_tokens: Total output tokens.
        usages: Individual token usages.
        started_at: When the request started.
        ended_at: When the request ended.
    """

    trace_id: str = ""
    workflow_id: str = ""
    total_cost: float = 0.0
    by_agent: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    usages: list[TokenUsage] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def duration_seconds(self) -> float | None:
        """Duration of the request in seconds."""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    @property
    def cost_per_token(self) -> float:
        """Average cost per token."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_cost / self.total_tokens

    def add_usage(self, usage: TokenUsage) -> None:
        """Add token usage to the breakdown.

        Args:
            usage: Token usage to add.
        """
        self.usages.append(usage)
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

        cost = usage.calculate_cost()
        self.total_cost += cost

        # Attribute to agent
        if usage.agent_name:
            self.by_agent[usage.agent_name] = (
                self.by_agent.get(usage.agent_name, 0.0) + cost
            )

        # Attribute to model
        if usage.model:
            self.by_model[usage.model] = self.by_model.get(usage.model, 0.0) + cost

    def add_agent_cost(self, agent_name: str, cost: float) -> None:
        """Add cost for an agent.

        Args:
            agent_name: Agent name.
            cost: Cost to add.
        """
        self.by_agent[agent_name] = self.by_agent.get(agent_name, 0.0) + cost
        self.total_cost += cost

    def finalize(self) -> None:
        """Mark the breakdown as complete."""
        self.ended_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "total_cost": self.total_cost,
            "total_cost_formatted": f"${self.total_cost:.4f}",
            "by_agent": self.by_agent,
            "by_model": self.by_model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_per_token": self.cost_per_token,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "usage_count": len(self.usages),
        }


# ============================================================================
# Cost Alerts
# ============================================================================


class AlertSeverity(Enum):
    """Severity levels for cost and performance alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CostAlert:
    """A cost alert.

    Attributes:
        severity: Alert severity level.
        message: Alert message.
        threshold: Threshold that was exceeded.
        actual_value: Actual value that triggered the alert.
        context: Additional context.
        timestamp: When the alert was triggered.
    """

    severity: AlertSeverity
    message: str
    threshold: float
    actual_value: float
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CostThresholds:
    """Cost thresholds for alerts.

    Attributes:
        per_request_warning: Warning threshold per request.
        per_request_critical: Critical threshold per request.
        per_agent_warning: Warning threshold per agent.
        per_agent_critical: Critical threshold per agent.
        daily_warning: Warning threshold for daily spend.
        daily_critical: Critical threshold for daily spend.
        tokens_per_request_warning: Token count warning threshold.
    """

    per_request_warning: float = 0.10  # $0.10
    per_request_critical: float = 0.50  # $0.50
    per_agent_warning: float = 0.05  # $0.05 per agent
    per_agent_critical: float = 0.25  # $0.25 per agent
    daily_warning: float = 10.00  # $10/day
    daily_critical: float = 50.00  # $50/day
    tokens_per_request_warning: int = 50_000  # 50k tokens


# Default thresholds
DEFAULT_THRESHOLDS = CostThresholds()


class CostAlertManager:
    """Manages cost alerts and thresholds.

    Provides functionality to check costs against thresholds
    and generate alerts when exceeded.
    """

    def __init__(
        self,
        thresholds: CostThresholds | None = None,
        on_alert: Callable[[CostAlert], None] | None = None,
    ) -> None:
        """Initialize the alert manager.

        Args:
            thresholds: Cost thresholds to use.
            on_alert: Callback function for alerts.
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.on_alert = on_alert
        self._alerts: list[CostAlert] = []
        self._daily_costs: dict[str, float] = {}  # date -> cost

    @property
    def alerts(self) -> list[CostAlert]:
        """Get all triggered alerts."""
        return self._alerts.copy()

    def check_request_cost(self, breakdown: CostBreakdown) -> list[CostAlert]:
        """Check request cost against thresholds.

        Args:
            breakdown: Cost breakdown to check.

        Returns:
            List of triggered alerts.
        """
        alerts: list[CostAlert] = []

        # Check total request cost
        if breakdown.total_cost >= self.thresholds.per_request_critical:
            alert = CostAlert(
                severity=AlertSeverity.CRITICAL,
                message=f"Request cost ${breakdown.total_cost:.4f} exceeds critical threshold",
                threshold=self.thresholds.per_request_critical,
                actual_value=breakdown.total_cost,
                context={"trace_id": breakdown.trace_id},
            )
            alerts.append(alert)
        elif breakdown.total_cost >= self.thresholds.per_request_warning:
            alert = CostAlert(
                severity=AlertSeverity.WARNING,
                message=f"Request cost ${breakdown.total_cost:.4f} exceeds warning threshold",
                threshold=self.thresholds.per_request_warning,
                actual_value=breakdown.total_cost,
                context={"trace_id": breakdown.trace_id},
            )
            alerts.append(alert)

        # Check per-agent costs
        for agent_name, agent_cost in breakdown.by_agent.items():
            if agent_cost >= self.thresholds.per_agent_critical:
                alert = CostAlert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"Agent '{agent_name}' cost ${agent_cost:.4f} exceeds critical threshold",
                    threshold=self.thresholds.per_agent_critical,
                    actual_value=agent_cost,
                    context={"agent_name": agent_name, "trace_id": breakdown.trace_id},
                )
                alerts.append(alert)
            elif agent_cost >= self.thresholds.per_agent_warning:
                alert = CostAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"Agent '{agent_name}' cost ${agent_cost:.4f} exceeds warning threshold",
                    threshold=self.thresholds.per_agent_warning,
                    actual_value=agent_cost,
                    context={"agent_name": agent_name, "trace_id": breakdown.trace_id},
                )
                alerts.append(alert)

        # Check token usage
        if breakdown.total_tokens >= self.thresholds.tokens_per_request_warning:
            alert = CostAlert(
                severity=AlertSeverity.WARNING,
                message=f"Token usage {breakdown.total_tokens:,} exceeds warning threshold",
                threshold=float(self.thresholds.tokens_per_request_warning),
                actual_value=float(breakdown.total_tokens),
                context={"trace_id": breakdown.trace_id},
            )
            alerts.append(alert)

        # Store and notify
        for alert in alerts:
            self._alerts.append(alert)
            if self.on_alert:
                self.on_alert(alert)
            logger.warning(
                "cost_alert_triggered",
                severity=alert.severity.value,
                message=alert.message,
                threshold=alert.threshold,
                actual_value=alert.actual_value,
            )

        return alerts

    def record_daily_cost(self, cost: float, date: datetime | None = None) -> None:
        """Record cost for daily tracking.

        Args:
            cost: Cost to record.
            date: Date to record for (defaults to today).
        """
        date = date or datetime.now(UTC)
        date_key = date.strftime("%Y-%m-%d")
        self._daily_costs[date_key] = self._daily_costs.get(date_key, 0.0) + cost

    def check_daily_cost(self, date: datetime | None = None) -> list[CostAlert]:
        """Check daily cost against thresholds.

        Args:
            date: Date to check (defaults to today).

        Returns:
            List of triggered alerts.
        """
        date = date or datetime.now(UTC)
        date_key = date.strftime("%Y-%m-%d")
        daily_total = self._daily_costs.get(date_key, 0.0)

        alerts: list[CostAlert] = []

        if daily_total >= self.thresholds.daily_critical:
            alert = CostAlert(
                severity=AlertSeverity.CRITICAL,
                message=f"Daily cost ${daily_total:.2f} exceeds critical threshold",
                threshold=self.thresholds.daily_critical,
                actual_value=daily_total,
                context={"date": date_key},
            )
            alerts.append(alert)
            self._alerts.append(alert)
        elif daily_total >= self.thresholds.daily_warning:
            alert = CostAlert(
                severity=AlertSeverity.WARNING,
                message=f"Daily cost ${daily_total:.2f} exceeds warning threshold",
                threshold=self.thresholds.daily_warning,
                actual_value=daily_total,
                context={"date": date_key},
            )
            alerts.append(alert)
            self._alerts.append(alert)

        return alerts

    def get_daily_cost(self, date: datetime | None = None) -> float:
        """Get daily cost.

        Args:
            date: Date to get cost for (defaults to today).

        Returns:
            Total cost for the day.
        """
        date = date or datetime.now(UTC)
        date_key = date.strftime("%Y-%m-%d")
        return self._daily_costs.get(date_key, 0.0)

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()


# ============================================================================
# Cost Tracker
# ============================================================================


class CostTracker:
    """Tracks costs across requests and agents.

    Provides methods to record usage, calculate costs,
    and generate reports.
    """

    def __init__(
        self,
        alert_manager: CostAlertManager | None = None,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            alert_manager: Optional alert manager for cost alerts.
        """
        self.alert_manager = alert_manager or CostAlertManager()
        self._current_breakdown: CostBreakdown | None = None
        self._completed_breakdowns: list[CostBreakdown] = []

    def start_request(
        self,
        trace_id: str = "",
        workflow_id: str = "",
    ) -> CostBreakdown:
        """Start tracking a new request.

        Args:
            trace_id: Trace identifier.
            workflow_id: Workflow identifier.

        Returns:
            New CostBreakdown for the request.
        """
        self._current_breakdown = CostBreakdown(
            trace_id=trace_id,
            workflow_id=workflow_id,
        )
        logger.debug("cost_tracking_started", trace_id=trace_id)
        return self._current_breakdown

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        agent_name: str = "",
    ) -> TokenUsage:
        """Record token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: Model used.
            agent_name: Agent that made the call.

        Returns:
            TokenUsage record.
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            agent_name=agent_name,
        )

        if self._current_breakdown:
            self._current_breakdown.add_usage(usage)

        logger.debug(
            "token_usage_recorded",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            agent_name=agent_name,
            cost=usage.calculate_cost(),
        )

        return usage

    def end_request(self) -> CostBreakdown | None:
        """End tracking the current request.

        Returns:
            Completed CostBreakdown, or None if no request was active.
        """
        if not self._current_breakdown:
            return None

        self._current_breakdown.finalize()
        breakdown = self._current_breakdown

        # Check for alerts
        self.alert_manager.check_request_cost(breakdown)
        self.alert_manager.record_daily_cost(breakdown.total_cost)
        self.alert_manager.check_daily_cost()

        # Store completed breakdown
        self._completed_breakdowns.append(breakdown)
        self._current_breakdown = None

        logger.info(
            "cost_tracking_ended",
            trace_id=breakdown.trace_id,
            total_cost=breakdown.total_cost,
            total_tokens=breakdown.total_tokens,
        )

        return breakdown

    @property
    def current_breakdown(self) -> CostBreakdown | None:
        """Get the current active breakdown."""
        return self._current_breakdown

    @property
    def completed_breakdowns(self) -> list[CostBreakdown]:
        """Get all completed breakdowns."""
        return self._completed_breakdowns.copy()


# ============================================================================
# Cost Reports
# ============================================================================


@dataclass
class CostReport:
    """Aggregated cost report.

    Attributes:
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        total_cost: Total cost in the period.
        total_requests: Number of requests.
        total_tokens: Total tokens used.
        by_agent: Cost breakdown by agent.
        by_model: Cost breakdown by model.
        by_day: Cost breakdown by day.
        average_cost_per_request: Average cost per request.
        alerts_count: Number of alerts triggered.
    """

    period_start: datetime
    period_end: datetime
    total_cost: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    by_agent: dict[str, float] = field(default_factory=dict)
    by_model: dict[str, float] = field(default_factory=dict)
    by_day: dict[str, float] = field(default_factory=dict)
    average_cost_per_request: float = 0.0
    alerts_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost": self.total_cost,
            "total_cost_formatted": f"${self.total_cost:.2f}",
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "by_agent": self.by_agent,
            "by_model": self.by_model,
            "by_day": self.by_day,
            "average_cost_per_request": self.average_cost_per_request,
            "alerts_count": self.alerts_count,
        }


def generate_cost_report(
    breakdowns: list[CostBreakdown],
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    alerts: list[CostAlert] | None = None,
) -> CostReport:
    """Generate a cost report from breakdowns.

    Args:
        breakdowns: List of cost breakdowns to aggregate.
        period_start: Start of reporting period (defaults to earliest breakdown).
        period_end: End of reporting period (defaults to latest breakdown).
        alerts: List of alerts to include in count.

    Returns:
        Aggregated cost report.
    """
    if not breakdowns:
        now = datetime.now(UTC)
        return CostReport(
            period_start=period_start or now,
            period_end=period_end or now,
            alerts_count=len(alerts) if alerts else 0,
        )

    # Determine period
    if period_start is None:
        period_start = min(b.started_at for b in breakdowns)
    if period_end is None:
        period_end = max(b.ended_at or b.started_at for b in breakdowns)

    # Aggregate data
    total_cost = 0.0
    total_tokens = 0
    by_agent: dict[str, float] = {}
    by_model: dict[str, float] = {}
    by_day: dict[str, float] = {}

    for breakdown in breakdowns:
        total_cost += breakdown.total_cost
        total_tokens += breakdown.total_tokens

        # Aggregate by agent
        for agent, cost in breakdown.by_agent.items():
            by_agent[agent] = by_agent.get(agent, 0.0) + cost

        # Aggregate by model
        for model, cost in breakdown.by_model.items():
            by_model[model] = by_model.get(model, 0.0) + cost

        # Aggregate by day
        day_key = breakdown.started_at.strftime("%Y-%m-%d")
        by_day[day_key] = by_day.get(day_key, 0.0) + breakdown.total_cost

    # Calculate averages
    total_requests = len(breakdowns)
    average_cost = total_cost / total_requests if total_requests > 0 else 0.0

    return CostReport(
        period_start=period_start,
        period_end=period_end,
        total_cost=total_cost,
        total_requests=total_requests,
        total_tokens=total_tokens,
        by_agent=by_agent,
        by_model=by_model,
        by_day=by_day,
        average_cost_per_request=average_cost,
        alerts_count=len(alerts) if alerts else 0,
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def calculate_llm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost for an LLM call.

    Args:
        model: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Cost in dollars.
    """
    pricing = get_model_pricing(model)
    return pricing.calculate_cost(input_tokens, output_tokens)


def format_cost(cost: float) -> str:
    """Format cost as a currency string.

    Args:
        cost: Cost in dollars.

    Returns:
        Formatted string (e.g., "$0.0123").
    """
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def estimate_request_cost(
    symbols_count: int,
    model: str = "claude-sonnet-4-20250514",
    agents_count: int = 3,
) -> float:
    """Estimate cost for a portfolio analysis request.

    Args:
        symbols_count: Number of symbols to analyze.
        model: Model to use.
        agents_count: Number of agents in the workflow.

    Returns:
        Estimated cost in dollars.
    """
    # Rough estimates based on typical usage
    # ~500 input tokens per symbol, ~200 output tokens per symbol per agent
    input_tokens_per_symbol = 500
    output_tokens_per_symbol = 200

    total_input = symbols_count * input_tokens_per_symbol * agents_count
    total_output = symbols_count * output_tokens_per_symbol * agents_count

    return calculate_llm_cost(model, total_input, total_output)


# Global tracker instance
_global_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Returns:
        Global CostTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _global_tracker
    _global_tracker = None
