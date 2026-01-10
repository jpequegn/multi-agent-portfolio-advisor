"""Tests for webhook manager module."""

import pytest

from src.webhooks.events import WebhookEventType
from src.webhooks.manager import (
    Webhook,
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookManager,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def manager():
    """Create test webhook manager."""
    return WebhookManager()


@pytest.fixture
def sample_config():
    """Create sample webhook config."""
    return WebhookConfig(
        events=[WebhookEventType.ANALYSIS_STARTED, WebhookEventType.ANALYSIS_COMPLETED],
        max_retries=3,
        timeout_seconds=15,
        custom_headers={"X-Custom": "test"},
    )


# ============================================================================
# WebhookConfig Tests
# ============================================================================


class TestWebhookConfig:
    """Tests for WebhookConfig model."""

    def test_default_config(self):
        """Test default config values."""
        config = WebhookConfig()

        assert config.events == []
        assert config.max_retries == 5
        assert config.timeout_seconds == 30
        assert config.custom_headers == {}

    def test_custom_config(self, sample_config):
        """Test custom config values."""
        assert len(sample_config.events) == 2
        assert sample_config.max_retries == 3
        assert sample_config.timeout_seconds == 15
        assert sample_config.custom_headers["X-Custom"] == "test"


# ============================================================================
# Webhook Tests
# ============================================================================


class TestWebhook:
    """Tests for Webhook model."""

    def test_create_webhook(self):
        """Test creating a webhook."""
        webhook = Webhook(url="https://example.com/webhook")

        assert webhook.id.startswith("wh_")
        assert webhook.secret.startswith("whsec_")
        assert str(webhook.url) == "https://example.com/webhook"
        assert webhook.enabled is True
        assert webhook.total_deliveries == 0

    def test_should_receive_event_with_empty_events(self):
        """Test that empty events list receives all events."""
        webhook = Webhook(
            url="https://example.com/webhook",
            config=WebhookConfig(events=[]),
        )

        assert webhook.should_receive_event(WebhookEventType.ANALYSIS_STARTED) is True
        assert webhook.should_receive_event(WebhookEventType.APPROVAL_REQUIRED) is True

    def test_should_receive_event_with_specific_events(self, sample_config):
        """Test that specific events list filters correctly."""
        webhook = Webhook(
            url="https://example.com/webhook",
            config=sample_config,
        )

        assert webhook.should_receive_event(WebhookEventType.ANALYSIS_STARTED) is True
        assert webhook.should_receive_event(WebhookEventType.ANALYSIS_COMPLETED) is True
        assert webhook.should_receive_event(WebhookEventType.APPROVAL_REQUIRED) is False

    def test_record_successful_delivery(self):
        """Test recording a successful delivery."""
        webhook = Webhook(url="https://example.com/webhook")

        webhook.record_delivery(success=True)

        assert webhook.total_deliveries == 1
        assert webhook.successful_deliveries == 1
        assert webhook.failed_deliveries == 0
        assert webhook.last_delivery_at is not None
        assert webhook.last_success_at is not None
        assert webhook.last_failure_at is None

    def test_record_failed_delivery(self):
        """Test recording a failed delivery."""
        webhook = Webhook(url="https://example.com/webhook")

        webhook.record_delivery(success=False)

        assert webhook.total_deliveries == 1
        assert webhook.successful_deliveries == 0
        assert webhook.failed_deliveries == 1
        assert webhook.last_delivery_at is not None
        assert webhook.last_success_at is None
        assert webhook.last_failure_at is not None


# ============================================================================
# WebhookDelivery Tests
# ============================================================================


class TestWebhookDelivery:
    """Tests for WebhookDelivery model."""

    def test_create_delivery(self):
        """Test creating a delivery."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
            payload={"test": True},
        )

        assert delivery.id.startswith("dlv_")
        assert delivery.webhook_id == "wh_123"
        assert delivery.event_id == "evt_456"
        assert delivery.status == WebhookDeliveryStatus.PENDING
        assert delivery.attempt_count == 0

    def test_mark_success(self):
        """Test marking delivery as successful."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
        )

        delivery.mark_success(response_status=200, response_body="OK")

        assert delivery.status == WebhookDeliveryStatus.SUCCESS
        assert delivery.response_status == 200
        assert delivery.response_body == "OK"
        assert delivery.completed_at is not None
        assert delivery.next_attempt_at is None

    def test_mark_success_truncates_response(self):
        """Test that long response bodies are truncated."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
        )

        long_response = "x" * 2000
        delivery.mark_success(response_status=200, response_body=long_response)

        assert len(delivery.response_body) == 1000

    def test_mark_failed_with_retry(self):
        """Test marking delivery as failed with retry."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
            max_attempts=5,
        )

        delivery.mark_failed(
            error_message="Connection timeout",
            response_status=None,
            can_retry=True,
        )

        assert delivery.status == WebhookDeliveryStatus.RETRYING
        assert delivery.attempt_count == 1
        assert delivery.error_message == "Connection timeout"
        assert delivery.next_attempt_at is not None
        assert delivery.completed_at is None

    def test_mark_failed_no_more_retries(self):
        """Test marking delivery as permanently failed."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
            max_attempts=3,
        )
        delivery.attempt_count = 2  # Already tried twice

        delivery.mark_failed(error_message="Final failure", can_retry=True)

        assert delivery.status == WebhookDeliveryStatus.FAILED
        assert delivery.attempt_count == 3
        assert delivery.completed_at is not None
        assert delivery.next_attempt_at is None

    def test_mark_failed_no_retry_allowed(self):
        """Test marking delivery as failed without retry."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
            max_attempts=5,
        )

        delivery.mark_failed(
            error_message="400 Bad Request",
            response_status=400,
            can_retry=False,
        )

        assert delivery.status == WebhookDeliveryStatus.FAILED
        assert delivery.response_status == 400
        assert delivery.completed_at is not None

    def test_exponential_backoff(self):
        """Test exponential backoff for retries."""
        delivery = WebhookDelivery(
            webhook_id="wh_123",
            event_id="evt_456",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            url="https://example.com/webhook",
            max_attempts=5,
        )

        # First failure - 1 second delay (2^0)
        delivery.mark_failed(error_message="Error", can_retry=True)
        first_delay = delivery.next_attempt_at

        # Second failure - 2 second delay (2^1)
        delivery.mark_failed(error_message="Error", can_retry=True)
        second_delay = delivery.next_attempt_at

        # The second delay should be later than the first
        assert second_delay > first_delay


# ============================================================================
# WebhookManager Registration Tests
# ============================================================================


class TestWebhookManagerRegistration:
    """Tests for webhook registration."""

    def test_register_webhook(self, manager):
        """Test registering a new webhook."""
        webhook = manager.register(
            url="https://example.com/webhook",
            description="Test webhook",
        )

        assert webhook.id is not None
        assert str(webhook.url) == "https://example.com/webhook"
        assert webhook.description == "Test webhook"

    def test_register_webhook_with_config(self, manager, sample_config):
        """Test registering a webhook with config."""
        webhook = manager.register(
            url="https://example.com/webhook",
            config=sample_config,
        )

        assert len(webhook.config.events) == 2
        assert webhook.config.max_retries == 3

    def test_register_webhook_with_events(self, manager):
        """Test registering a webhook with events list."""
        webhook = manager.register(
            url="https://example.com/webhook",
            events=[WebhookEventType.APPROVAL_REQUIRED],
        )

        assert WebhookEventType.APPROVAL_REQUIRED in webhook.config.events

    def test_get_webhook(self, manager):
        """Test retrieving a webhook by ID."""
        created = manager.register(url="https://example.com/webhook")

        retrieved = manager.get(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_webhook_not_found(self, manager):
        """Test retrieving a non-existent webhook."""
        retrieved = manager.get("non-existent-id")
        assert retrieved is None

    def test_list_webhooks(self, manager):
        """Test listing all webhooks."""
        manager.register(url="https://example1.com/webhook")
        manager.register(url="https://example2.com/webhook")

        webhooks = manager.list_all()

        assert len(webhooks) == 2

    def test_list_enabled_only(self, manager):
        """Test listing only enabled webhooks."""
        wh1 = manager.register(url="https://example1.com/webhook")
        wh2 = manager.register(url="https://example2.com/webhook")
        manager.update(wh2.id, enabled=False)

        webhooks = manager.list_all(enabled_only=True)

        assert len(webhooks) == 1
        assert webhooks[0].id == wh1.id

    def test_list_by_event_type(self, manager):
        """Test listing webhooks by event type."""
        manager.register(
            url="https://example1.com/webhook",
            events=[WebhookEventType.ANALYSIS_STARTED],
        )
        manager.register(
            url="https://example2.com/webhook",
            events=[WebhookEventType.APPROVAL_REQUIRED],
        )
        # This one subscribes to all events
        manager.register(url="https://example3.com/webhook")

        # Should get webhooks subscribed to analysis.started
        webhooks = manager.list_all(event_type=WebhookEventType.ANALYSIS_STARTED)

        assert len(webhooks) == 2


# ============================================================================
# WebhookManager Update/Delete Tests
# ============================================================================


class TestWebhookManagerUpdateDelete:
    """Tests for webhook update and delete."""

    def test_update_url(self, manager):
        """Test updating webhook URL."""
        webhook = manager.register(url="https://old.example.com/webhook")

        updated = manager.update(webhook.id, url="https://new.example.com/webhook")

        assert updated is not None
        assert str(updated.url) == "https://new.example.com/webhook"

    def test_update_description(self, manager):
        """Test updating webhook description."""
        webhook = manager.register(
            url="https://example.com/webhook",
            description="Old description",
        )

        updated = manager.update(webhook.id, description="New description")

        assert updated.description == "New description"

    def test_update_enabled(self, manager):
        """Test enabling/disabling webhook."""
        webhook = manager.register(url="https://example.com/webhook")

        updated = manager.update(webhook.id, enabled=False)

        assert updated.enabled is False

    def test_update_config(self, manager, sample_config):
        """Test updating webhook config."""
        webhook = manager.register(url="https://example.com/webhook")

        updated = manager.update(webhook.id, config=sample_config)

        assert len(updated.config.events) == 2
        assert updated.config.max_retries == 3

    def test_update_not_found(self, manager):
        """Test updating a non-existent webhook."""
        updated = manager.update("non-existent-id", description="New")
        assert updated is None

    def test_delete_webhook(self, manager):
        """Test deleting a webhook."""
        webhook = manager.register(url="https://example.com/webhook")

        deleted = manager.delete(webhook.id)

        assert deleted is True
        assert manager.get(webhook.id) is None

    def test_delete_not_found(self, manager):
        """Test deleting a non-existent webhook."""
        deleted = manager.delete("non-existent-id")
        assert deleted is False


# ============================================================================
# WebhookManager Event Subscription Tests
# ============================================================================


class TestWebhookManagerEventSubscription:
    """Tests for event subscription filtering."""

    def test_get_webhooks_for_event(self, manager):
        """Test getting webhooks subscribed to an event."""
        manager.register(
            url="https://example1.com/webhook",
            events=[WebhookEventType.ANALYSIS_STARTED],
        )
        manager.register(
            url="https://example2.com/webhook",
            events=[WebhookEventType.APPROVAL_REQUIRED],
        )

        webhooks = manager.get_webhooks_for_event(WebhookEventType.ANALYSIS_STARTED)

        assert len(webhooks) == 1

    def test_get_webhooks_for_event_includes_all_subscribers(self, manager):
        """Test that webhooks with empty events get all events."""
        manager.register(
            url="https://specific.example.com/webhook",
            events=[WebhookEventType.ANALYSIS_STARTED],
        )
        manager.register(url="https://all.example.com/webhook")  # No filter

        webhooks = manager.get_webhooks_for_event(WebhookEventType.ANALYSIS_STARTED)

        assert len(webhooks) == 2

    def test_get_webhooks_for_event_excludes_disabled(self, manager):
        """Test that disabled webhooks are excluded."""
        wh = manager.register(
            url="https://example.com/webhook",
            events=[WebhookEventType.ANALYSIS_STARTED],
        )
        manager.update(wh.id, enabled=False)

        webhooks = manager.get_webhooks_for_event(WebhookEventType.ANALYSIS_STARTED)

        assert len(webhooks) == 0


# ============================================================================
# WebhookManager Delivery Tracking Tests
# ============================================================================


class TestWebhookManagerDeliveryTracking:
    """Tests for delivery tracking."""

    def test_create_delivery(self, manager):
        """Test creating a delivery record."""
        webhook = manager.register(url="https://example.com/webhook")

        delivery = manager.create_delivery(
            webhook=webhook,
            event_id="evt_123",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={"test": True},
        )

        assert delivery.id is not None
        assert delivery.webhook_id == webhook.id
        assert delivery.event_id == "evt_123"
        assert delivery.max_attempts == webhook.config.max_retries + 1

    def test_get_delivery(self, manager):
        """Test retrieving a delivery by ID."""
        webhook = manager.register(url="https://example.com/webhook")
        delivery = manager.create_delivery(
            webhook=webhook,
            event_id="evt_123",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={},
        )

        retrieved = manager.get_delivery(delivery.id)

        assert retrieved is not None
        assert retrieved.id == delivery.id

    def test_list_deliveries(self, manager):
        """Test listing deliveries for a webhook."""
        webhook = manager.register(url="https://example.com/webhook")
        manager.create_delivery(
            webhook=webhook,
            event_id="evt_1",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={},
        )
        manager.create_delivery(
            webhook=webhook,
            event_id="evt_2",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={},
        )

        deliveries = manager.list_deliveries(webhook.id)

        assert len(deliveries) == 2

    def test_list_deliveries_with_limit(self, manager):
        """Test listing deliveries with limit."""
        webhook = manager.register(url="https://example.com/webhook")
        for i in range(10):
            manager.create_delivery(
                webhook=webhook,
                event_id=f"evt_{i}",
                event_type=WebhookEventType.ANALYSIS_STARTED,
                payload={},
            )

        deliveries = manager.list_deliveries(webhook.id, limit=5)

        assert len(deliveries) == 5

    def test_list_deliveries_by_status(self, manager):
        """Test listing deliveries filtered by status."""
        webhook = manager.register(url="https://example.com/webhook")
        d1 = manager.create_delivery(
            webhook=webhook,
            event_id="evt_1",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={},
        )
        manager.create_delivery(
            webhook=webhook,
            event_id="evt_2",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={},
        )

        # Mark one as success
        d1.mark_success(response_status=200)

        deliveries = manager.list_deliveries(
            webhook.id, status=WebhookDeliveryStatus.SUCCESS
        )

        assert len(deliveries) == 1
        assert deliveries[0].id == d1.id

    def test_get_pending_retries(self, manager):
        """Test getting pending retry deliveries."""
        webhook = manager.register(url="https://example.com/webhook")
        d1 = manager.create_delivery(
            webhook=webhook,
            event_id="evt_1",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={},
        )
        manager.create_delivery(
            webhook=webhook,
            event_id="evt_2",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={},
        )

        # Mark one for retry (it sets next_attempt_at in the past for immediate retry)
        d1.mark_failed(error_message="Timeout", can_retry=True)
        # Simulate immediate retry is due
        from datetime import UTC, datetime
        d1.next_attempt_at = datetime.now(UTC)

        pending = manager.get_pending_retries()

        assert len(pending) == 1
        assert pending[0].id == d1.id
