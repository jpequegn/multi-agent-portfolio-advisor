"""Approval request models and types.

This module defines the data models for the human approval workflow.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    MODIFIED = "modified"


class RiskLevel(str, Enum):
    """Risk level classification for recommendations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalTrigger(str, Enum):
    """Reasons that triggered the approval requirement."""

    HIGH_VALUE_TRADE = "high_value_trade"
    HIGH_CONCENTRATION = "high_concentration"
    HIGH_RISK_ASSET = "high_risk_asset"
    FIRST_TIME_SYMBOL = "first_time_symbol"
    LOW_CONFIDENCE = "low_confidence"
    COMPLIANCE_WARNING = "compliance_warning"
    MANUAL_REVIEW = "manual_review"


class ApprovalDecision(BaseModel):
    """A human reviewer's decision on an approval request."""

    status: ApprovalStatus = Field(
        ..., description="Decision: approved, rejected, or modified"
    )
    reviewer_id: str = Field(..., description="Identifier of the reviewer")
    reason: str | None = Field(
        default=None, description="Reason for rejection or modification"
    )
    modified_trades: list[dict[str, Any]] | None = Field(
        default=None, description="Modified trades (if status is modified)"
    )
    decided_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the decision was made",
    )


class ApprovalRequest(BaseModel):
    """A request for human approval of high-risk recommendations.

    This model tracks the full lifecycle of an approval request from
    creation through final decision.
    """

    # Identifiers
    approval_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this approval request",
    )
    workflow_id: str = Field(..., description="Associated workflow ID")
    trace_id: str = Field(..., description="Trace ID for observability")
    user_id: str | None = Field(default=None, description="User who initiated analysis")

    # Status tracking
    status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING,
        description="Current status of the approval request",
    )

    # Risk assessment
    risk_level: RiskLevel = Field(
        default=RiskLevel.MEDIUM, description="Overall risk level"
    )
    triggers: list[ApprovalTrigger] = Field(
        default_factory=list,
        description="Reasons that triggered approval requirement",
    )

    # Recommendation data
    trades: list[dict[str, Any]] = Field(
        default_factory=list, description="Proposed trades requiring approval"
    )
    total_value: float = Field(
        default=0.0, description="Total value of proposed trades"
    )
    portfolio_value: float = Field(
        default=0.0, description="Total portfolio value for context"
    )
    compliance: dict[str, Any] = Field(
        default_factory=dict, description="Compliance check results"
    )

    # Context
    summary: str = Field(
        default="", description="Summary of the recommendation"
    )
    analysis_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context from analysis (risk metrics, etc.)",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the approval request was created",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="When the approval request expires (if not acted upon)",
    )

    # Decision
    decision: ApprovalDecision | None = Field(
        default=None, description="Human reviewer's decision"
    )

    def is_pending(self) -> bool:
        """Check if the request is still pending."""
        return self.status == ApprovalStatus.PENDING

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    def approve(self, reviewer_id: str, reason: str | None = None) -> None:
        """Approve the request."""
        self.status = ApprovalStatus.APPROVED
        self.decision = ApprovalDecision(
            status=ApprovalStatus.APPROVED,
            reviewer_id=reviewer_id,
            reason=reason,
        )

    def reject(self, reviewer_id: str, reason: str) -> None:
        """Reject the request."""
        self.status = ApprovalStatus.REJECTED
        self.decision = ApprovalDecision(
            status=ApprovalStatus.REJECTED,
            reviewer_id=reviewer_id,
            reason=reason,
        )

    def modify(
        self,
        reviewer_id: str,
        modified_trades: list[dict[str, Any]],
        reason: str | None = None,
    ) -> None:
        """Approve with modifications."""
        self.status = ApprovalStatus.MODIFIED
        self.decision = ApprovalDecision(
            status=ApprovalStatus.MODIFIED,
            reviewer_id=reviewer_id,
            reason=reason,
            modified_trades=modified_trades,
        )

    def mark_expired(self) -> None:
        """Mark the request as expired."""
        self.status = ApprovalStatus.EXPIRED


class ApprovalConfig(BaseModel):
    """Configuration for approval thresholds and rules."""

    # Value thresholds
    high_value_threshold: float = Field(
        default=10000.0,
        description="Trade value threshold that triggers approval ($)",
    )
    critical_value_threshold: float = Field(
        default=50000.0,
        description="Trade value threshold for critical risk level ($)",
    )

    # Concentration thresholds
    concentration_threshold: float = Field(
        default=0.20,
        description="Portfolio concentration threshold (20%)",
    )
    critical_concentration_threshold: float = Field(
        default=0.30,
        description="Critical concentration threshold (30%)",
    )

    # Confidence thresholds
    low_confidence_threshold: float = Field(
        default=0.7,
        description="Confidence score below which approval is required",
    )

    # High-risk asset classes
    high_risk_asset_types: list[str] = Field(
        default_factory=lambda: ["options", "leveraged_etf", "inverse_etf", "crypto"],
        description="Asset types that always require approval",
    )

    # Timeout configuration
    default_timeout_hours: int = Field(
        default=24,
        description="Default timeout for pending approvals (hours)",
    )
    critical_timeout_hours: int = Field(
        default=4,
        description="Timeout for critical risk approvals (hours)",
    )

    # Auto-approval settings
    auto_approve_low_risk: bool = Field(
        default=True,
        description="Automatically approve low-risk trades",
    )
