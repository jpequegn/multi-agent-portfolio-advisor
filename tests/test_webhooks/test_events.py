"""Tests for webhook events module."""

from datetime import UTC, datetime

from src.webhooks.events import (
    WebhookEventMetadata,
    WebhookEventType,
    WebhookPayload,
    build_alert_event,
    build_analysis_completed_event,
    build_analysis_failed_event,
    build_analysis_started_event,
    build_approval_decision_event,
    build_approval_required_event,
    build_circuit_breaker_event,
    create_webhook_event,
)

# ============================================================================
# WebhookEventType Tests
# ============================================================================


class TestWebhookEventType:
    """Tests for WebhookEventType enum."""

    def test_analysis_events(self):
        """Test analysis event types exist."""
        assert WebhookEventType.ANALYSIS_STARTED == "analysis.started"
        assert WebhookEventType.ANALYSIS_COMPLETED == "analysis.completed"
        assert WebhookEventType.ANALYSIS_FAILED == "analysis.failed"

    def test_approval_events(self):
        """Test approval event types exist."""
        assert WebhookEventType.APPROVAL_REQUIRED == "approval.required"
        assert WebhookEventType.APPROVAL_GRANTED == "approval.granted"
        assert WebhookEventType.APPROVAL_REJECTED == "approval.rejected"
        assert WebhookEventType.APPROVAL_TIMEOUT == "approval.timeout"

    def test_alert_events(self):
        """Test alert event types exist."""
        assert WebhookEventType.COST_THRESHOLD_EXCEEDED == "alert.cost_threshold"
        assert WebhookEventType.LATENCY_THRESHOLD_EXCEEDED == "alert.latency_threshold"
        assert WebhookEventType.ERROR_RATE_HIGH == "alert.error_rate"

    def test_system_events(self):
        """Test system event types exist."""
        assert WebhookEventType.CIRCUIT_BREAKER_OPENED == "system.circuit_open"
        assert WebhookEventType.CIRCUIT_BREAKER_CLOSED == "system.circuit_closed"


# ============================================================================
# WebhookPayload Tests
# ============================================================================


class TestWebhookPayload:
    """Tests for WebhookPayload model."""

    def test_default_payload(self):
        """Test creating payload with defaults."""
        payload = WebhookPayload(type=WebhookEventType.ANALYSIS_STARTED)

        assert payload.id.startswith("evt_")
        assert payload.type == WebhookEventType.ANALYSIS_STARTED
        assert payload.timestamp is not None
        assert payload.data == {}
        assert payload.metadata is not None

    def test_payload_with_data(self):
        """Test creating payload with data."""
        payload = WebhookPayload(
            type=WebhookEventType.ANALYSIS_COMPLETED,
            data={"trace_id": "trace-123", "duration_ms": 1500},
        )

        assert payload.type == WebhookEventType.ANALYSIS_COMPLETED
        assert payload.data["trace_id"] == "trace-123"
        assert payload.data["duration_ms"] == 1500

    def test_to_json_dict(self):
        """Test converting payload to JSON dict."""
        payload = WebhookPayload(
            type=WebhookEventType.ANALYSIS_STARTED,
            data={"test": True},
        )
        payload.id = "evt_test123"

        json_dict = payload.to_json_dict()

        assert json_dict["id"] == "evt_test123"
        assert json_dict["type"] == "analysis.started"
        assert "timestamp" in json_dict
        assert json_dict["data"] == {"test": True}
        assert "metadata" in json_dict
        assert json_dict["metadata"]["source"] == "portfolio-advisor"


# ============================================================================
# WebhookEventMetadata Tests
# ============================================================================


class TestWebhookEventMetadata:
    """Tests for WebhookEventMetadata model."""

    def test_default_metadata(self):
        """Test default metadata values."""
        metadata = WebhookEventMetadata()

        assert metadata.version == "1.0.0"
        assert metadata.source == "portfolio-advisor"
        # environment comes from env var with default

    def test_custom_metadata(self):
        """Test custom metadata values."""
        metadata = WebhookEventMetadata(
            environment="production",
            version="2.0.0",
            source="custom-source",
        )

        assert metadata.environment == "production"
        assert metadata.version == "2.0.0"
        assert metadata.source == "custom-source"


# ============================================================================
# Event Builder Tests
# ============================================================================


class TestEventBuilders:
    """Tests for event builder functions."""

    def test_create_webhook_event(self):
        """Test basic event creation."""
        event = create_webhook_event(
            WebhookEventType.ANALYSIS_STARTED,
            {"trace_id": "trace-123"},
        )

        assert event.type == WebhookEventType.ANALYSIS_STARTED
        assert event.data["trace_id"] == "trace-123"

    def test_create_webhook_event_with_custom_id(self):
        """Test event creation with custom ID."""
        event = create_webhook_event(
            WebhookEventType.ANALYSIS_STARTED,
            {"test": True},
            event_id="custom-id",
        )

        assert event.id == "custom-id"

    def test_create_webhook_event_with_custom_timestamp(self):
        """Test event creation with custom timestamp."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        event = create_webhook_event(
            WebhookEventType.ANALYSIS_STARTED,
            {"test": True},
            timestamp=custom_time,
        )

        assert event.timestamp == custom_time

    def test_build_analysis_started_event(self):
        """Test building analysis started event."""
        event = build_analysis_started_event(
            trace_id="trace-123",
            workflow_id="wf-456",
            user_id="user-789",
            portfolio_id="port-012",
            symbol_count=5,
        )

        assert event.type == WebhookEventType.ANALYSIS_STARTED
        assert event.data["trace_id"] == "trace-123"
        assert event.data["workflow_id"] == "wf-456"
        assert event.data["user_id"] == "user-789"
        assert event.data["portfolio_id"] == "port-012"
        assert event.data["symbol_count"] == 5

    def test_build_analysis_completed_event(self):
        """Test building analysis completed event."""
        event = build_analysis_completed_event(
            trace_id="trace-123",
            workflow_id="wf-456",
            duration_ms=1500.5,
            cost_usd=0.05,
            recommendation_count=3,
            has_errors=False,
        )

        assert event.type == WebhookEventType.ANALYSIS_COMPLETED
        assert event.data["duration_ms"] == 1500.5
        assert event.data["cost_usd"] == 0.05
        assert event.data["recommendation_count"] == 3
        assert event.data["has_errors"] is False

    def test_build_analysis_failed_event(self):
        """Test building analysis failed event."""
        event = build_analysis_failed_event(
            trace_id="trace-123",
            workflow_id="wf-456",
            error_message="Something went wrong",
            duration_ms=500.0,
        )

        assert event.type == WebhookEventType.ANALYSIS_FAILED
        assert event.data["error_message"] == "Something went wrong"
        assert event.data["duration_ms"] == 500.0

    def test_build_approval_required_event(self):
        """Test building approval required event."""
        expires = datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC)
        event = build_approval_required_event(
            approval_id="apr-123",
            trace_id="trace-456",
            workflow_id="wf-789",
            risk_level="high",
            total_value=15000.0,
            trade_count=3,
            triggers=["high_value", "first_time_symbol"],
            expires_at=expires,
        )

        assert event.type == WebhookEventType.APPROVAL_REQUIRED
        assert event.data["approval_id"] == "apr-123"
        assert event.data["risk_level"] == "high"
        assert event.data["total_value"] == 15000.0
        assert event.data["trade_count"] == 3
        assert event.data["triggers"] == ["high_value", "first_time_symbol"]
        assert event.data["expires_at"] == expires.isoformat()

    def test_build_approval_decision_event(self):
        """Test building approval decision event."""
        event = build_approval_decision_event(
            event_type=WebhookEventType.APPROVAL_GRANTED,
            approval_id="apr-123",
            trace_id="trace-456",
            workflow_id="wf-789",
            reviewer_id="reviewer-012",
            reason="Approved after review",
        )

        assert event.type == WebhookEventType.APPROVAL_GRANTED
        assert event.data["approval_id"] == "apr-123"
        assert event.data["reviewer_id"] == "reviewer-012"
        assert event.data["reason"] == "Approved after review"

    def test_build_alert_event(self):
        """Test building alert event."""
        event = build_alert_event(
            event_type=WebhookEventType.COST_THRESHOLD_EXCEEDED,
            alert_name="High API Cost",
            current_value=1.50,
            threshold=1.00,
            trace_id="trace-123",
            details={"model": "gpt-4", "tokens": 50000},
        )

        assert event.type == WebhookEventType.COST_THRESHOLD_EXCEEDED
        assert event.data["alert_name"] == "High API Cost"
        assert event.data["current_value"] == 1.50
        assert event.data["threshold"] == 1.00
        assert event.data["trace_id"] == "trace-123"
        assert event.data["details"]["model"] == "gpt-4"

    def test_build_circuit_breaker_event(self):
        """Test building circuit breaker event."""
        event = build_circuit_breaker_event(
            event_type=WebhookEventType.CIRCUIT_BREAKER_OPENED,
            service_name="market-data-api",
            failure_count=10,
            recovery_time_seconds=60,
        )

        assert event.type == WebhookEventType.CIRCUIT_BREAKER_OPENED
        assert event.data["service_name"] == "market-data-api"
        assert event.data["failure_count"] == 10
        assert event.data["recovery_time_seconds"] == 60
