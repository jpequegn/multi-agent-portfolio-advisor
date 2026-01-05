"""Tests for approval manager."""

from datetime import UTC, datetime, timedelta

import pytest

from src.approval.manager import (
    ApprovalConfig,
    ApprovalEvaluation,
    ApprovalManager,
    InMemoryApprovalPersistence,
)
from src.approval.models import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalTrigger,
    RiskLevel,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config():
    """Create test config with lower thresholds."""
    return ApprovalConfig(
        high_value_threshold=5000.0,
        critical_value_threshold=25000.0,
        concentration_threshold=0.15,
        critical_concentration_threshold=0.25,
        low_confidence_threshold=0.7,
        default_timeout_hours=24,
        critical_timeout_hours=4,
        auto_approve_low_risk=True,
    )


@pytest.fixture
def manager(config):
    """Create test approval manager."""
    return ApprovalManager(config=config)


@pytest.fixture
def sample_low_value_trades():
    """Sample low value trades that don't require approval."""
    return [
        {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "estimated_value": 1000.0,
            "target_weight": 0.05,
            "current_weight": 0.04,
        },
    ]


@pytest.fixture
def sample_high_value_trades():
    """Sample high value trades that require approval."""
    return [
        {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "estimated_value": 15000.0,
            "target_weight": 0.18,
            "current_weight": 0.10,
        },
    ]


@pytest.fixture
def sample_high_concentration_trades():
    """Sample trades with high concentration."""
    return [
        {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 50,
            "estimated_value": 3000.0,
            "target_weight": 0.30,
            "current_weight": 0.10,
        },
    ]


@pytest.fixture
def sample_compliance_clean():
    """Sample clean compliance result."""
    return {
        "is_compliant": True,
        "violations": [],
        "warnings": [],
        "requires_approval": False,
    }


@pytest.fixture
def sample_compliance_with_warnings():
    """Sample compliance with warnings."""
    return {
        "is_compliant": True,
        "violations": [],
        "warnings": ["High sector concentration"],
        "requires_approval": True,
    }


# ============================================================================
# ApprovalEvaluation Tests
# ============================================================================


class TestApprovalEvaluation:
    """Tests for ApprovalEvaluation."""

    def test_no_approval_for_low_value(
        self, manager, sample_low_value_trades, sample_compliance_clean
    ):
        """Test that low value trades don't require approval."""
        evaluation = manager.evaluate_recommendation(
            trades=sample_low_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
            known_symbols={"AAPL"},  # Mark AAPL as known to avoid first-time trigger
        )

        assert evaluation.requires_approval is False
        assert evaluation.risk_level == RiskLevel.LOW
        assert len(evaluation.triggers) == 0

    def test_approval_required_for_high_value(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test that high value trades require approval."""
        evaluation = manager.evaluate_recommendation(
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        assert evaluation.requires_approval is True
        assert ApprovalTrigger.HIGH_VALUE_TRADE in evaluation.triggers
        assert evaluation.details["total_trade_value"] == 15000.0

    def test_approval_required_for_high_concentration(
        self, manager, sample_high_concentration_trades, sample_compliance_clean
    ):
        """Test that high concentration triggers approval."""
        evaluation = manager.evaluate_recommendation(
            trades=sample_high_concentration_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        assert evaluation.requires_approval is True
        assert ApprovalTrigger.HIGH_CONCENTRATION in evaluation.triggers

    def test_approval_required_for_compliance_warnings(
        self, manager, sample_low_value_trades, sample_compliance_with_warnings
    ):
        """Test that compliance warnings trigger approval."""
        evaluation = manager.evaluate_recommendation(
            trades=sample_low_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_with_warnings,
        )

        assert evaluation.requires_approval is True
        assert ApprovalTrigger.COMPLIANCE_WARNING in evaluation.triggers

    def test_approval_for_low_confidence(
        self, manager, sample_low_value_trades, sample_compliance_clean
    ):
        """Test that low confidence triggers approval."""
        evaluation = manager.evaluate_recommendation(
            trades=sample_low_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
            confidence_score=0.5,  # Below threshold of 0.7
        )

        assert evaluation.requires_approval is True
        assert ApprovalTrigger.LOW_CONFIDENCE in evaluation.triggers

    def test_approval_for_first_time_symbol(
        self, manager, sample_low_value_trades, sample_compliance_clean
    ):
        """Test that first-time symbols trigger approval."""
        evaluation = manager.evaluate_recommendation(
            trades=sample_low_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
            known_symbols={"GOOGL", "MSFT"},  # AAPL is not known
        )

        assert evaluation.requires_approval is True
        assert ApprovalTrigger.FIRST_TIME_SYMBOL in evaluation.triggers
        assert "AAPL" in evaluation.details["new_symbols"]

    def test_high_risk_assets_detected(self, manager, sample_compliance_clean):
        """Test that high-risk assets are detected."""
        high_risk_trades = [
            {
                "symbol": "TQQQ",
                "action": "buy",
                "quantity": 100,
                "estimated_value": 3000.0,
            },
        ]

        evaluation = manager.evaluate_recommendation(
            trades=high_risk_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        assert evaluation.requires_approval is True
        assert ApprovalTrigger.HIGH_RISK_ASSET in evaluation.triggers
        assert "TQQQ" in evaluation.details["high_risk_assets"]

    def test_critical_risk_level(self, manager, sample_compliance_clean):
        """Test that critical value triggers critical risk level."""
        critical_trades = [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 500,
                "estimated_value": 30000.0,  # Above critical threshold of 25000
            },
        ]

        evaluation = manager.evaluate_recommendation(
            trades=critical_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        assert evaluation.risk_level == RiskLevel.CRITICAL

    def test_multiple_triggers_high_risk(self, manager, sample_compliance_with_warnings):
        """Test that multiple triggers result in high risk."""
        multi_issue_trades = [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 100,
                "estimated_value": 10000.0,
                "target_weight": 0.20,
            },
        ]

        evaluation = manager.evaluate_recommendation(
            trades=multi_issue_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_with_warnings,
            confidence_score=0.5,  # Low confidence
        )

        # Should have at least 3 triggers: high_value, high_concentration, compliance, low_confidence
        assert len(evaluation.triggers) >= 3
        assert evaluation.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]


# ============================================================================
# ApprovalManager Create/Get Tests
# ============================================================================


class TestApprovalManagerCreateGet:
    """Tests for creating and retrieving approvals."""

    @pytest.mark.asyncio
    async def test_create_approval_request(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test creating an approval request."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
            summary="Test approval",
            user_id="user-789",
        )

        assert approval.approval_id is not None
        assert approval.workflow_id == "wf-123"
        assert approval.trace_id == "trace-456"
        assert approval.user_id == "user-789"
        assert approval.status == ApprovalStatus.PENDING
        assert approval.expires_at is not None
        assert len(approval.trades) == 1

    @pytest.mark.asyncio
    async def test_get_approval(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test retrieving an approval by ID."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        retrieved = await manager.get_approval(approval.approval_id)

        assert retrieved is not None
        assert retrieved.approval_id == approval.approval_id

    @pytest.mark.asyncio
    async def test_get_approval_not_found(self, manager):
        """Test retrieving a non-existent approval."""
        retrieved = await manager.get_approval("non-existent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_approval_by_workflow(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test retrieving an approval by workflow ID."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        retrieved = await manager.get_approval_by_workflow("wf-123")

        assert retrieved is not None
        assert retrieved.workflow_id == "wf-123"

    @pytest.mark.asyncio
    async def test_list_pending(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test listing pending approvals."""
        # Create multiple approvals
        await manager.create_approval_request(
            workflow_id="wf-1",
            trace_id="trace-1",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
            user_id="user-A",
        )
        await manager.create_approval_request(
            workflow_id="wf-2",
            trace_id="trace-2",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
            user_id="user-B",
        )

        # List all pending
        pending = await manager.list_pending()
        assert len(pending) == 2

        # List by user
        pending_user_a = await manager.list_pending(user_id="user-A")
        assert len(pending_user_a) == 1
        assert pending_user_a[0].user_id == "user-A"


# ============================================================================
# ApprovalManager Decision Tests
# ============================================================================


class TestApprovalManagerDecisions:
    """Tests for approval decisions."""

    @pytest.mark.asyncio
    async def test_approve(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test approving a request."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        result = await manager.approve(
            approval_id=approval.approval_id,
            reviewer_id="reviewer-123",
            reason="Looks good",
        )

        assert result is not None
        assert result.status == ApprovalStatus.APPROVED
        assert result.decision is not None
        assert result.decision.reviewer_id == "reviewer-123"

    @pytest.mark.asyncio
    async def test_reject(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test rejecting a request."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        result = await manager.reject(
            approval_id=approval.approval_id,
            reviewer_id="reviewer-456",
            reason="Too risky",
        )

        assert result is not None
        assert result.status == ApprovalStatus.REJECTED
        assert result.decision.reason == "Too risky"

    @pytest.mark.asyncio
    async def test_modify(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test modifying a request."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        modified_trades = [{"symbol": "AAPL", "action": "buy", "quantity": 50}]
        result = await manager.modify(
            approval_id=approval.approval_id,
            reviewer_id="reviewer-789",
            modified_trades=modified_trades,
            reason="Reduced quantity",
        )

        assert result is not None
        assert result.status == ApprovalStatus.MODIFIED
        assert result.decision.modified_trades == modified_trades

    @pytest.mark.asyncio
    async def test_approve_not_found(self, manager):
        """Test approving a non-existent request."""
        result = await manager.approve(
            approval_id="non-existent",
            reviewer_id="reviewer-123",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cannot_approve_already_approved(
        self, manager, sample_high_value_trades, sample_compliance_clean
    ):
        """Test that already approved requests can't be approved again."""
        approval = await manager.create_approval_request(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_high_value_trades,
            portfolio_value=100000.0,
            compliance=sample_compliance_clean,
        )

        # First approval
        await manager.approve(
            approval_id=approval.approval_id,
            reviewer_id="reviewer-1",
        )

        # Second approval attempt
        result = await manager.approve(
            approval_id=approval.approval_id,
            reviewer_id="reviewer-2",
        )

        # Should return the approval but not change status
        assert result is not None
        assert result.status == ApprovalStatus.APPROVED
        assert result.decision.reviewer_id == "reviewer-1"


# ============================================================================
# InMemoryApprovalPersistence Tests
# ============================================================================


class TestInMemoryApprovalPersistence:
    """Tests for in-memory persistence."""

    @pytest.fixture
    def persistence(self):
        """Create test persistence."""
        return InMemoryApprovalPersistence()

    @pytest.fixture
    def sample_approval(self):
        """Create sample approval."""
        return ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            user_id="user-789",
            trades=[{"symbol": "AAPL", "action": "buy"}],
            compliance={},
        )

    @pytest.mark.asyncio
    async def test_save_and_get(self, persistence, sample_approval):
        """Test saving and retrieving."""
        await persistence.save(sample_approval)
        retrieved = await persistence.get(sample_approval.approval_id)

        assert retrieved is not None
        assert retrieved.approval_id == sample_approval.approval_id

    @pytest.mark.asyncio
    async def test_get_by_workflow(self, persistence, sample_approval):
        """Test retrieving by workflow ID."""
        await persistence.save(sample_approval)
        retrieved = await persistence.get_by_workflow(sample_approval.workflow_id)

        assert retrieved is not None
        assert retrieved.workflow_id == sample_approval.workflow_id

    @pytest.mark.asyncio
    async def test_list_pending(self, persistence):
        """Test listing pending approvals."""
        # Create approvals with different statuses
        pending1 = ApprovalRequest(
            workflow_id="wf-1",
            trace_id="trace-1",
            trades=[],
            compliance={},
            status=ApprovalStatus.PENDING,
        )
        pending2 = ApprovalRequest(
            workflow_id="wf-2",
            trace_id="trace-2",
            trades=[],
            compliance={},
            status=ApprovalStatus.PENDING,
        )
        approved = ApprovalRequest(
            workflow_id="wf-3",
            trace_id="trace-3",
            trades=[],
            compliance={},
            status=ApprovalStatus.APPROVED,
        )

        await persistence.save(pending1)
        await persistence.save(pending2)
        await persistence.save(approved)

        pending = await persistence.list_pending()

        assert len(pending) == 2
        assert all(p.status == ApprovalStatus.PENDING for p in pending)

    @pytest.mark.asyncio
    async def test_expire_stale(self, persistence):
        """Test expiring stale approvals."""
        # Create an expired approval
        expired = ApprovalRequest(
            workflow_id="wf-expired",
            trace_id="trace-expired",
            trades=[],
            compliance={},
            status=ApprovalStatus.PENDING,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
        # Create a valid approval
        valid = ApprovalRequest(
            workflow_id="wf-valid",
            trace_id="trace-valid",
            trades=[],
            compliance={},
            status=ApprovalStatus.PENDING,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )

        await persistence.save(expired)
        await persistence.save(valid)

        # Expire stale
        expired_count = await persistence.expire_stale()

        assert expired_count == 1

        # Check that expired one is now expired status
        retrieved_expired = await persistence.get(expired.approval_id)
        assert retrieved_expired.status == ApprovalStatus.EXPIRED

        # Check that valid one is still pending
        retrieved_valid = await persistence.get(valid.approval_id)
        assert retrieved_valid.status == ApprovalStatus.PENDING
