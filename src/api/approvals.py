"""API routes for human approval workflows.

This module provides endpoints for managing approval requests:
- List pending approvals
- Get approval details
- Approve/reject/modify recommendations
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.approval import (
    ApprovalRequest,
    ApprovalStatus,
    get_approval_manager,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/approvals", tags=["Approvals"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ApproveRequest(BaseModel):
    """Request to approve a recommendation."""

    reviewer_id: str = Field(..., description="ID of the reviewer approving")
    reason: str | None = Field(
        default=None, description="Optional reason for approval"
    )


class RejectRequest(BaseModel):
    """Request to reject a recommendation."""

    reviewer_id: str = Field(..., description="ID of the reviewer rejecting")
    reason: str = Field(..., description="Reason for rejection")


class ModifyRequest(BaseModel):
    """Request to approve with modifications."""

    reviewer_id: str = Field(..., description="ID of the reviewer")
    modified_trades: list[dict[str, Any]] = Field(
        ..., description="Modified trade list"
    )
    reason: str | None = Field(
        default=None, description="Reason for modification"
    )


class ApprovalSummary(BaseModel):
    """Summary of an approval request for listing."""

    approval_id: str
    workflow_id: str
    trace_id: str
    status: str
    risk_level: str
    total_value: float
    trade_count: int
    created_at: str
    expires_at: str | None
    summary: str


class ApprovalDetail(BaseModel):
    """Full detail of an approval request."""

    approval_id: str
    workflow_id: str
    trace_id: str
    user_id: str | None
    status: str
    risk_level: str
    triggers: list[str]
    trades: list[dict[str, Any]]
    total_value: float
    portfolio_value: float
    compliance: dict[str, Any]
    summary: str
    analysis_context: dict[str, Any]
    created_at: str
    expires_at: str | None
    decision: dict[str, Any] | None


class ApprovalResponse(BaseModel):
    """Response after an approval action."""

    approval_id: str
    status: str
    message: str


# ============================================================================
# Helper Functions
# ============================================================================


def _approval_to_summary(approval: ApprovalRequest) -> ApprovalSummary:
    """Convert an approval request to a summary."""
    return ApprovalSummary(
        approval_id=approval.approval_id,
        workflow_id=approval.workflow_id,
        trace_id=approval.trace_id,
        status=approval.status.value,
        risk_level=approval.risk_level.value,
        total_value=approval.total_value,
        trade_count=len(approval.trades),
        created_at=approval.created_at.isoformat(),
        expires_at=approval.expires_at.isoformat() if approval.expires_at else None,
        summary=approval.summary,
    )


def _approval_to_detail(approval: ApprovalRequest) -> ApprovalDetail:
    """Convert an approval request to full detail."""
    return ApprovalDetail(
        approval_id=approval.approval_id,
        workflow_id=approval.workflow_id,
        trace_id=approval.trace_id,
        user_id=approval.user_id,
        status=approval.status.value,
        risk_level=approval.risk_level.value,
        triggers=[t.value for t in approval.triggers],
        trades=approval.trades,
        total_value=approval.total_value,
        portfolio_value=approval.portfolio_value,
        compliance=approval.compliance,
        summary=approval.summary,
        analysis_context=approval.analysis_context,
        created_at=approval.created_at.isoformat(),
        expires_at=approval.expires_at.isoformat() if approval.expires_at else None,
        decision=approval.decision.model_dump(mode="json")
        if approval.decision
        else None,
    )


# ============================================================================
# Routes
# ============================================================================


@router.get(
    "",
    response_model=list[ApprovalSummary],
    summary="List pending approvals",
    description="Get a list of pending approval requests.",
)
async def list_approvals(
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ApprovalSummary]:
    """List pending approval requests.

    Args:
        user_id: Optional filter by user ID.
        limit: Maximum number of results (default 50).
        offset: Pagination offset.

    Returns:
        List of pending approval summaries.
    """
    manager = get_approval_manager()
    approvals = await manager.list_pending(
        user_id=user_id, limit=limit, offset=offset
    )

    logger.info(
        "approvals_listed",
        count=len(approvals),
        user_id=user_id,
    )

    return [_approval_to_summary(a) for a in approvals]


@router.get(
    "/{approval_id}",
    response_model=ApprovalDetail,
    summary="Get approval details",
    description="Get full details of an approval request.",
    responses={
        404: {"description": "Approval not found"},
    },
)
async def get_approval(approval_id: str) -> ApprovalDetail:
    """Get approval request details.

    Args:
        approval_id: The approval request ID.

    Returns:
        Full approval request details.

    Raises:
        HTTPException: If approval not found.
    """
    manager = get_approval_manager()
    approval = await manager.get_approval(approval_id)

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"Approval request '{approval_id}' not found",
        )

    logger.info(
        "approval_retrieved",
        approval_id=approval_id,
        status=approval.status.value,
    )

    return _approval_to_detail(approval)


@router.post(
    "/{approval_id}/approve",
    response_model=ApprovalResponse,
    summary="Approve a recommendation",
    description="Approve a pending recommendation for execution.",
    responses={
        404: {"description": "Approval not found"},
        400: {"description": "Approval is not in pending status"},
    },
)
async def approve_recommendation(
    approval_id: str,
    request: ApproveRequest,
) -> ApprovalResponse:
    """Approve a pending recommendation.

    Args:
        approval_id: The approval request ID.
        request: Approval request with reviewer info.

    Returns:
        Response confirming the approval.

    Raises:
        HTTPException: If approval not found or not pending.
    """
    manager = get_approval_manager()

    # Check current state before attempting approval
    current_approval = await manager.get_approval(approval_id)
    if current_approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"Approval request '{approval_id}' not found",
        )

    if not current_approval.is_pending():
        if current_approval.status == ApprovalStatus.EXPIRED:
            raise HTTPException(
                status_code=400,
                detail="Approval request has expired",
            )
        raise HTTPException(
            status_code=400,
            detail=f"Approval is not in pending status (current: {current_approval.status.value})",
        )

    approval = await manager.approve(
        approval_id=approval_id,
        reviewer_id=request.reviewer_id,
        reason=request.reason,
    )

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"Approval request '{approval_id}' not found",
        )

    if approval.status == ApprovalStatus.EXPIRED:
        raise HTTPException(
            status_code=400,
            detail="Approval request has expired",
        )

    if approval.status != ApprovalStatus.APPROVED:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to approve (current: {approval.status.value})",
        )

    return ApprovalResponse(
        approval_id=approval.approval_id,
        status=approval.status.value,
        message="Recommendation approved successfully",
    )


@router.post(
    "/{approval_id}/reject",
    response_model=ApprovalResponse,
    summary="Reject a recommendation",
    description="Reject a pending recommendation with a reason.",
    responses={
        404: {"description": "Approval not found"},
        400: {"description": "Approval is not in pending status"},
    },
)
async def reject_recommendation(
    approval_id: str,
    request: RejectRequest,
) -> ApprovalResponse:
    """Reject a pending recommendation.

    Args:
        approval_id: The approval request ID.
        request: Rejection request with reviewer info and reason.

    Returns:
        Response confirming the rejection.

    Raises:
        HTTPException: If approval not found or not pending.
    """
    manager = get_approval_manager()
    approval = await manager.reject(
        approval_id=approval_id,
        reviewer_id=request.reviewer_id,
        reason=request.reason,
    )

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"Approval request '{approval_id}' not found",
        )

    if approval.status == ApprovalStatus.EXPIRED:
        raise HTTPException(
            status_code=400,
            detail="Approval request has expired",
        )

    if approval.status != ApprovalStatus.REJECTED:
        raise HTTPException(
            status_code=400,
            detail=f"Approval is not in pending status (current: {approval.status.value})",
        )

    return ApprovalResponse(
        approval_id=approval.approval_id,
        status=approval.status.value,
        message="Recommendation rejected",
    )


@router.post(
    "/{approval_id}/modify",
    response_model=ApprovalResponse,
    summary="Approve with modifications",
    description="Approve a recommendation with modified trades.",
    responses={
        404: {"description": "Approval not found"},
        400: {"description": "Approval is not in pending status"},
    },
)
async def modify_recommendation(
    approval_id: str,
    request: ModifyRequest,
) -> ApprovalResponse:
    """Approve a recommendation with modifications.

    Args:
        approval_id: The approval request ID.
        request: Modification request with new trades.

    Returns:
        Response confirming the modification.

    Raises:
        HTTPException: If approval not found or not pending.
    """
    manager = get_approval_manager()
    approval = await manager.modify(
        approval_id=approval_id,
        reviewer_id=request.reviewer_id,
        modified_trades=request.modified_trades,
        reason=request.reason,
    )

    if approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"Approval request '{approval_id}' not found",
        )

    if approval.status == ApprovalStatus.EXPIRED:
        raise HTTPException(
            status_code=400,
            detail="Approval request has expired",
        )

    if approval.status != ApprovalStatus.MODIFIED:
        raise HTTPException(
            status_code=400,
            detail=f"Approval is not in pending status (current: {approval.status.value})",
        )

    return ApprovalResponse(
        approval_id=approval.approval_id,
        status=approval.status.value,
        message="Recommendation approved with modifications",
    )
