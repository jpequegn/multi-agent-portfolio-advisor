"""Human-in-the-loop approval workflow module.

This module provides approval workflows for high-risk recommendations,
enabling human oversight before trade execution.
"""

from src.approval.manager import ApprovalManager, get_approval_manager
from src.approval.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalTrigger,
    RiskLevel,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalManager",
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalTrigger",
    "RiskLevel",
    "get_approval_manager",
]
