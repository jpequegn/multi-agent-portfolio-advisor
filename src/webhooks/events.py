"""Webhook event types and payload models.

This module defines the structure of webhook events that can be sent
to external systems (Slack, email, monitoring tools, etc.).
"""

import os
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WebhookEventType(str, Enum):
    """Supported webhook event types.

    Events are organized by category:
    - analysis.*: Portfolio analysis lifecycle events
    - approval.*: Human approval workflow events
    - alert.*: System alert events
    - system.*: Infrastructure events
    """

    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"

    # Approval events
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    APPROVAL_TIMEOUT = "approval.timeout"

    # Alert events
    COST_THRESHOLD_EXCEEDED = "alert.cost_threshold"
    LATENCY_THRESHOLD_EXCEEDED = "alert.latency_threshold"
    ERROR_RATE_HIGH = "alert.error_rate"

    # System events
    CIRCUIT_BREAKER_OPENED = "system.circuit_open"
    CIRCUIT_BREAKER_CLOSED = "system.circuit_closed"


class WebhookEventMetadata(BaseModel):
    """Metadata included with every webhook event."""

    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development"),
        description="Deployment environment (development, staging, production)",
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
    )
    source: str = Field(
        default="portfolio-advisor",
        description="System source identifier",
    )


class WebhookPayload(BaseModel):
    """Standard payload structure for webhook events.

    All webhook deliveries use this consistent format to make
    integration easier for external systems.
    """

    id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}",
        description="Unique event identifier",
    )
    type: WebhookEventType = Field(
        ..., description="Event type"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the event occurred",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data",
    )
    metadata: WebhookEventMetadata = Field(
        default_factory=WebhookEventMetadata,
        description="Event metadata",
    )

    def to_json_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary with ISO-formatted timestamp.
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "metadata": {
                "environment": self.metadata.environment,
                "version": self.metadata.version,
                "source": self.metadata.source,
            },
        }


def create_webhook_event(
    event_type: WebhookEventType,
    data: dict[str, Any],
    *,
    event_id: str | None = None,
    timestamp: datetime | None = None,
) -> WebhookPayload:
    """Create a webhook event payload.

    Args:
        event_type: Type of event.
        data: Event-specific data.
        event_id: Optional custom event ID.
        timestamp: Optional custom timestamp.

    Returns:
        WebhookPayload ready for delivery.
    """
    payload = WebhookPayload(
        type=event_type,
        data=data,
    )

    if event_id:
        payload.id = event_id

    if timestamp:
        payload.timestamp = timestamp

    return payload


# Event data builders for common events


def build_analysis_started_event(
    trace_id: str,
    workflow_id: str,
    *,
    user_id: str | None = None,
    portfolio_id: str | None = None,
    symbol_count: int = 0,
) -> WebhookPayload:
    """Build an analysis.started event.

    Args:
        trace_id: Trace ID for observability.
        workflow_id: Workflow identifier.
        user_id: Optional user identifier.
        portfolio_id: Optional portfolio identifier.
        symbol_count: Number of symbols being analyzed.

    Returns:
        Webhook payload for analysis started.
    """
    return create_webhook_event(
        WebhookEventType.ANALYSIS_STARTED,
        {
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "symbol_count": symbol_count,
        },
    )


def build_analysis_completed_event(
    trace_id: str,
    workflow_id: str,
    *,
    user_id: str | None = None,
    portfolio_id: str | None = None,
    duration_ms: float = 0,
    cost_usd: float = 0,
    recommendation_count: int = 0,
    has_errors: bool = False,
) -> WebhookPayload:
    """Build an analysis.completed event.

    Args:
        trace_id: Trace ID for observability.
        workflow_id: Workflow identifier.
        user_id: Optional user identifier.
        portfolio_id: Optional portfolio identifier.
        duration_ms: Total duration in milliseconds.
        cost_usd: Estimated cost in USD.
        recommendation_count: Number of recommendations generated.
        has_errors: Whether errors occurred.

    Returns:
        Webhook payload for analysis completed.
    """
    return create_webhook_event(
        WebhookEventType.ANALYSIS_COMPLETED,
        {
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "duration_ms": duration_ms,
            "cost_usd": cost_usd,
            "recommendation_count": recommendation_count,
            "has_errors": has_errors,
        },
    )


def build_analysis_failed_event(
    trace_id: str,
    workflow_id: str,
    error_message: str,
    *,
    user_id: str | None = None,
    portfolio_id: str | None = None,
    duration_ms: float = 0,
) -> WebhookPayload:
    """Build an analysis.failed event.

    Args:
        trace_id: Trace ID for observability.
        workflow_id: Workflow identifier.
        error_message: Error description.
        user_id: Optional user identifier.
        portfolio_id: Optional portfolio identifier.
        duration_ms: Duration before failure.

    Returns:
        Webhook payload for analysis failed.
    """
    return create_webhook_event(
        WebhookEventType.ANALYSIS_FAILED,
        {
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "portfolio_id": portfolio_id,
            "error_message": error_message,
            "duration_ms": duration_ms,
        },
    )


def build_approval_required_event(
    approval_id: str,
    trace_id: str,
    workflow_id: str,
    *,
    user_id: str | None = None,
    risk_level: str = "medium",
    total_value: float = 0,
    trade_count: int = 0,
    triggers: list[str] | None = None,
    expires_at: datetime | None = None,
) -> WebhookPayload:
    """Build an approval.required event.

    Args:
        approval_id: Approval request identifier.
        trace_id: Trace ID for observability.
        workflow_id: Workflow identifier.
        user_id: Optional user identifier.
        risk_level: Risk classification.
        total_value: Total value of trades requiring approval.
        trade_count: Number of trades.
        triggers: Approval triggers.
        expires_at: When the approval expires.

    Returns:
        Webhook payload for approval required.
    """
    return create_webhook_event(
        WebhookEventType.APPROVAL_REQUIRED,
        {
            "approval_id": approval_id,
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "risk_level": risk_level,
            "total_value": total_value,
            "trade_count": trade_count,
            "triggers": triggers or [],
            "expires_at": expires_at.isoformat() if expires_at else None,
        },
    )


def build_approval_decision_event(
    event_type: WebhookEventType,
    approval_id: str,
    trace_id: str,
    workflow_id: str,
    *,
    user_id: str | None = None,
    reviewer_id: str | None = None,
    reason: str | None = None,
) -> WebhookPayload:
    """Build an approval decision event (granted/rejected/timeout).

    Args:
        event_type: APPROVAL_GRANTED, APPROVAL_REJECTED, or APPROVAL_TIMEOUT.
        approval_id: Approval request identifier.
        trace_id: Trace ID for observability.
        workflow_id: Workflow identifier.
        user_id: Optional user identifier.
        reviewer_id: ID of reviewer who made decision.
        reason: Reason for decision.

    Returns:
        Webhook payload for approval decision.
    """
    return create_webhook_event(
        event_type,
        {
            "approval_id": approval_id,
            "trace_id": trace_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "reviewer_id": reviewer_id,
            "reason": reason,
        },
    )


def build_alert_event(
    event_type: WebhookEventType,
    alert_name: str,
    current_value: float,
    threshold: float,
    *,
    trace_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> WebhookPayload:
    """Build an alert event.

    Args:
        event_type: Type of alert.
        alert_name: Name of the alert.
        current_value: Current metric value.
        threshold: Threshold that was exceeded.
        trace_id: Optional trace ID if related to specific request.
        details: Optional additional details.

    Returns:
        Webhook payload for alert.
    """
    return create_webhook_event(
        event_type,
        {
            "alert_name": alert_name,
            "current_value": current_value,
            "threshold": threshold,
            "trace_id": trace_id,
            "details": details or {},
        },
    )


def build_circuit_breaker_event(
    event_type: WebhookEventType,
    service_name: str,
    *,
    failure_count: int = 0,
    recovery_time_seconds: int | None = None,
) -> WebhookPayload:
    """Build a circuit breaker event.

    Args:
        event_type: CIRCUIT_BREAKER_OPENED or CIRCUIT_BREAKER_CLOSED.
        service_name: Name of the affected service.
        failure_count: Number of failures that triggered the circuit.
        recovery_time_seconds: Time until recovery attempt.

    Returns:
        Webhook payload for circuit breaker event.
    """
    return create_webhook_event(
        event_type,
        {
            "service_name": service_name,
            "failure_count": failure_count,
            "recovery_time_seconds": recovery_time_seconds,
        },
    )
