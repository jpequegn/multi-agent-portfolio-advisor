"""Tests for approval models."""

from datetime import UTC, datetime, timedelta

import pytest

from src.approval.models import (
    ApprovalConfig,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalTrigger,
    RiskLevel,
)


# ============================================================================
# ApprovalStatus Tests
# ============================================================================


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.EXPIRED.value == "expired"
        assert ApprovalStatus.MODIFIED.value == "modified"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_risk_levels_exist(self):
        """Test that all expected risk levels exist."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestApprovalTrigger:
    """Tests for ApprovalTrigger enum."""

    def test_all_triggers_exist(self):
        """Test that all expected triggers exist."""
        assert ApprovalTrigger.HIGH_VALUE_TRADE.value == "high_value_trade"
        assert ApprovalTrigger.HIGH_CONCENTRATION.value == "high_concentration"
        assert ApprovalTrigger.HIGH_RISK_ASSET.value == "high_risk_asset"
        assert ApprovalTrigger.FIRST_TIME_SYMBOL.value == "first_time_symbol"
        assert ApprovalTrigger.LOW_CONFIDENCE.value == "low_confidence"
        assert ApprovalTrigger.COMPLIANCE_WARNING.value == "compliance_warning"


# ============================================================================
# ApprovalDecision Tests
# ============================================================================


class TestApprovalDecision:
    """Tests for ApprovalDecision model."""

    def test_create_approval_decision(self):
        """Test creating an approval decision."""
        decision = ApprovalDecision(
            status=ApprovalStatus.APPROVED,
            reviewer_id="reviewer-123",
            reason="Looks good",
        )
        assert decision.status == ApprovalStatus.APPROVED
        assert decision.reviewer_id == "reviewer-123"
        assert decision.reason == "Looks good"
        assert decision.modified_trades is None
        assert decision.decided_at is not None

    def test_create_rejection_decision(self):
        """Test creating a rejection decision."""
        decision = ApprovalDecision(
            status=ApprovalStatus.REJECTED,
            reviewer_id="reviewer-456",
            reason="Trade value too high",
        )
        assert decision.status == ApprovalStatus.REJECTED
        assert decision.reason == "Trade value too high"

    def test_create_modified_decision(self):
        """Test creating a modified decision."""
        modified_trades = [{"symbol": "AAPL", "action": "buy", "quantity": 50}]
        decision = ApprovalDecision(
            status=ApprovalStatus.MODIFIED,
            reviewer_id="reviewer-789",
            reason="Reduced quantity",
            modified_trades=modified_trades,
        )
        assert decision.status == ApprovalStatus.MODIFIED
        assert decision.modified_trades == modified_trades


# ============================================================================
# ApprovalRequest Tests
# ============================================================================


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    @pytest.fixture
    def sample_trades(self):
        """Sample trades for testing."""
        return [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 100,
                "estimated_value": 15000.0,
            },
            {
                "symbol": "GOOGL",
                "action": "sell",
                "quantity": 50,
                "estimated_value": 10000.0,
            },
        ]

    @pytest.fixture
    def sample_compliance(self):
        """Sample compliance data for testing."""
        return {
            "is_compliant": True,
            "violations": [],
            "warnings": ["High concentration in tech sector"],
            "requires_approval": True,
        }

    def test_create_approval_request(self, sample_trades, sample_compliance):
        """Test creating an approval request."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            user_id="user-789",
            risk_level=RiskLevel.HIGH,
            triggers=[ApprovalTrigger.HIGH_VALUE_TRADE],
            trades=sample_trades,
            total_value=25000.0,
            portfolio_value=100000.0,
            compliance=sample_compliance,
            summary="Buy AAPL, sell GOOGL",
        )

        assert request.approval_id is not None
        assert request.workflow_id == "wf-123"
        assert request.trace_id == "trace-456"
        assert request.user_id == "user-789"
        assert request.status == ApprovalStatus.PENDING
        assert request.risk_level == RiskLevel.HIGH
        assert len(request.triggers) == 1
        assert len(request.trades) == 2
        assert request.total_value == 25000.0
        assert request.decision is None

    def test_is_pending(self, sample_trades, sample_compliance):
        """Test is_pending method."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
        )
        assert request.is_pending() is True

        request.status = ApprovalStatus.APPROVED
        assert request.is_pending() is False

    def test_is_expired_without_expiry(self, sample_trades, sample_compliance):
        """Test is_expired when no expiry is set."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
        )
        assert request.is_expired() is False

    def test_is_expired_with_future_expiry(self, sample_trades, sample_compliance):
        """Test is_expired with future expiry."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
        assert request.is_expired() is False

    def test_is_expired_with_past_expiry(self, sample_trades, sample_compliance):
        """Test is_expired with past expiry."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
        assert request.is_expired() is True

    def test_approve(self, sample_trades, sample_compliance):
        """Test approve method."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
        )

        request.approve(reviewer_id="reviewer-123", reason="Approved")

        assert request.status == ApprovalStatus.APPROVED
        assert request.decision is not None
        assert request.decision.reviewer_id == "reviewer-123"
        assert request.decision.reason == "Approved"

    def test_reject(self, sample_trades, sample_compliance):
        """Test reject method."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
        )

        request.reject(reviewer_id="reviewer-456", reason="Too risky")

        assert request.status == ApprovalStatus.REJECTED
        assert request.decision is not None
        assert request.decision.reviewer_id == "reviewer-456"
        assert request.decision.reason == "Too risky"

    def test_modify(self, sample_trades, sample_compliance):
        """Test modify method."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
        )

        modified_trades = [{"symbol": "AAPL", "action": "buy", "quantity": 50}]
        request.modify(
            reviewer_id="reviewer-789",
            modified_trades=modified_trades,
            reason="Reduced quantity",
        )

        assert request.status == ApprovalStatus.MODIFIED
        assert request.decision is not None
        assert request.decision.modified_trades == modified_trades

    def test_mark_expired(self, sample_trades, sample_compliance):
        """Test mark_expired method."""
        request = ApprovalRequest(
            workflow_id="wf-123",
            trace_id="trace-456",
            trades=sample_trades,
            compliance=sample_compliance,
        )

        request.mark_expired()

        assert request.status == ApprovalStatus.EXPIRED


# ============================================================================
# ApprovalConfig Tests
# ============================================================================


class TestApprovalConfig:
    """Tests for ApprovalConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ApprovalConfig()

        assert config.high_value_threshold == 10000.0
        assert config.critical_value_threshold == 50000.0
        assert config.concentration_threshold == 0.20
        assert config.critical_concentration_threshold == 0.30
        assert config.low_confidence_threshold == 0.7
        assert "options" in config.high_risk_asset_types
        assert config.default_timeout_hours == 24
        assert config.critical_timeout_hours == 4
        assert config.auto_approve_low_risk is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ApprovalConfig(
            high_value_threshold=5000.0,
            concentration_threshold=0.15,
            default_timeout_hours=12,
            auto_approve_low_risk=False,
        )

        assert config.high_value_threshold == 5000.0
        assert config.concentration_threshold == 0.15
        assert config.default_timeout_hours == 12
        assert config.auto_approve_low_risk is False
