"""Tests for approval API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.approvals import router
from src.api.health import reset_health_service
from src.api.routes import create_app
from src.approval.manager import ApprovalManager, InMemoryApprovalPersistence
from src.approval.models import (
    ApprovalConfig,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalTrigger,
    RiskLevel,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def test_manager():
    """Create test approval manager."""
    config = ApprovalConfig(
        high_value_threshold=5000.0,
        concentration_threshold=0.15,
    )
    return ApprovalManager(config=config)


@pytest.fixture
def app(test_manager):
    """Create test FastAPI app with approval routes."""
    reset_health_service()
    application = create_app(title="Test API", version="0.1.0")

    # Patch the approval manager
    with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
        yield application


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_approval():
    """Create sample approval request."""
    return ApprovalRequest(
        workflow_id="wf-123",
        trace_id="trace-456",
        user_id="user-789",
        risk_level=RiskLevel.HIGH,
        triggers=[ApprovalTrigger.HIGH_VALUE_TRADE],
        trades=[
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 100,
                "estimated_value": 15000.0,
            }
        ],
        total_value=15000.0,
        portfolio_value=100000.0,
        compliance={"is_compliant": True, "warnings": []},
        summary="Buy AAPL",
    )


# ============================================================================
# List Approvals Tests
# ============================================================================


class TestListApprovals:
    """Tests for GET /approvals endpoint."""

    def test_list_empty(self, client, test_manager):
        """Test listing when no approvals exist."""
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.get("/approvals")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_with_approvals(self, client, test_manager, sample_approval):
        """Test listing with existing approvals."""
        # Add approval to persistence
        await test_manager.persistence.save(sample_approval)

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.get("/approvals")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["approval_id"] == sample_approval.approval_id

    @pytest.mark.asyncio
    async def test_list_filter_by_user(self, client, test_manager):
        """Test filtering by user_id."""
        # Create approvals for different users
        approval_a = ApprovalRequest(
            workflow_id="wf-1",
            trace_id="trace-1",
            user_id="user-A",
            trades=[],
            compliance={},
        )
        approval_b = ApprovalRequest(
            workflow_id="wf-2",
            trace_id="trace-2",
            user_id="user-B",
            trades=[],
            compliance={},
        )

        await test_manager.persistence.save(approval_a)
        await test_manager.persistence.save(approval_b)

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.get("/approvals", params={"user_id": "user-A"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["workflow_id"] == "wf-1"


# ============================================================================
# Get Approval Tests
# ============================================================================


class TestGetApproval:
    """Tests for GET /approvals/{approval_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_approval_success(self, client, test_manager, sample_approval):
        """Test getting an existing approval."""
        await test_manager.persistence.save(sample_approval)

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.get(f"/approvals/{sample_approval.approval_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["approval_id"] == sample_approval.approval_id
        assert data["workflow_id"] == "wf-123"
        assert data["status"] == "pending"
        assert data["risk_level"] == "high"
        assert len(data["trades"]) == 1

    def test_get_approval_not_found(self, client, test_manager):
        """Test getting a non-existent approval."""
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.get("/approvals/non-existent-id")

        assert response.status_code == 404
        data = response.json()
        # Error message is in 'error' field (from ErrorResponse model)
        error_message = data.get("detail") or data.get("error", "")
        assert "not found" in error_message


# ============================================================================
# Approve Tests
# ============================================================================


class TestApproveEndpoint:
    """Tests for POST /approvals/{approval_id}/approve endpoint."""

    @pytest.mark.asyncio
    async def test_approve_success(self, client, test_manager, sample_approval):
        """Test approving an approval request."""
        await test_manager.persistence.save(sample_approval)

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                f"/approvals/{sample_approval.approval_id}/approve",
                json={"reviewer_id": "reviewer-123", "reason": "Approved"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["message"] == "Recommendation approved successfully"

    def test_approve_not_found(self, client, test_manager):
        """Test approving a non-existent approval."""
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                "/approvals/non-existent-id/approve",
                json={"reviewer_id": "reviewer-123"},
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_already_approved(self, client, test_manager, sample_approval):
        """Test approving an already approved request - returns 400."""
        sample_approval.approve(reviewer_id="previous-reviewer")
        await test_manager.persistence.save(sample_approval)

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                f"/approvals/{sample_approval.approval_id}/approve",
                json={"reviewer_id": "reviewer-123"},
            )

        # API returns 400 because it's not pending
        assert response.status_code == 400
        data = response.json()
        # Error message is in 'error' field (from ErrorResponse model)
        error_message = data.get("detail") or data.get("error", "")
        assert "not in pending status" in error_message


# ============================================================================
# Reject Tests
# ============================================================================


class TestRejectEndpoint:
    """Tests for POST /approvals/{approval_id}/reject endpoint."""

    @pytest.mark.asyncio
    async def test_reject_success(self, client, test_manager, sample_approval):
        """Test rejecting an approval request."""
        await test_manager.persistence.save(sample_approval)

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                f"/approvals/{sample_approval.approval_id}/reject",
                json={"reviewer_id": "reviewer-456", "reason": "Too risky"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["message"] == "Recommendation rejected"

    def test_reject_not_found(self, client, test_manager):
        """Test rejecting a non-existent approval."""
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                "/approvals/non-existent-id/reject",
                json={"reviewer_id": "reviewer-456", "reason": "Not found"},
            )

        assert response.status_code == 404

    def test_reject_requires_reason(self, client, test_manager):
        """Test that reject requires a reason."""
        # FastAPI validation should fail without reason
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                "/approvals/some-id/reject",
                json={"reviewer_id": "reviewer-456"},  # Missing reason
            )

        assert response.status_code == 422  # Validation error


# ============================================================================
# Modify Tests
# ============================================================================


class TestModifyEndpoint:
    """Tests for POST /approvals/{approval_id}/modify endpoint."""

    @pytest.mark.asyncio
    async def test_modify_success(self, client, test_manager, sample_approval):
        """Test modifying an approval request."""
        await test_manager.persistence.save(sample_approval)

        modified_trades = [{"symbol": "AAPL", "action": "buy", "quantity": 50}]

        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                f"/approvals/{sample_approval.approval_id}/modify",
                json={
                    "reviewer_id": "reviewer-789",
                    "modified_trades": modified_trades,
                    "reason": "Reduced quantity",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "modified"
        assert data["message"] == "Recommendation approved with modifications"

    def test_modify_not_found(self, client, test_manager):
        """Test modifying a non-existent approval."""
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                "/approvals/non-existent-id/modify",
                json={
                    "reviewer_id": "reviewer-789",
                    "modified_trades": [],
                },
            )

        assert response.status_code == 404

    def test_modify_requires_trades(self, client, test_manager):
        """Test that modify requires modified_trades."""
        # FastAPI validation should fail without modified_trades
        with patch("src.api.approvals.get_approval_manager", return_value=test_manager):
            response = client.post(
                "/approvals/some-id/modify",
                json={"reviewer_id": "reviewer-789"},  # Missing modified_trades
            )

        assert response.status_code == 422  # Validation error
