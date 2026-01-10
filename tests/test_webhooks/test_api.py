"""Tests for webhook API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes import create_app
from src.webhooks.events import WebhookEventType
from src.webhooks.manager import (
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookManager,
    set_webhook_manager,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def manager():
    """Create and set test webhook manager."""
    mgr = WebhookManager()
    set_webhook_manager(mgr)
    return mgr


@pytest.fixture
def client(manager):  # noqa: ARG001
    """Create test client (manager fixture ensures manager is set)."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_webhook(manager):
    """Create a sample webhook."""
    return manager.register(
        url="https://example.com/webhook",
        description="Test webhook",
        config=WebhookConfig(
            events=[WebhookEventType.ANALYSIS_STARTED],
            max_retries=3,
        ),
    )


# ============================================================================
# Create Webhook Tests
# ============================================================================


class TestCreateWebhook:
    """Tests for POST /webhooks endpoint."""

    def test_create_webhook(self, client):
        """Test creating a new webhook."""
        response = client.post(
            "/webhooks",
            json={
                "url": "https://example.com/webhook",
                "description": "Test webhook",
                "events": ["analysis.started", "analysis.completed"],
                "max_retries": 3,
                "timeout_seconds": 15,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["id"].startswith("wh_")
        assert data["url"] == "https://example.com/webhook"
        assert data["description"] == "Test webhook"
        assert len(data["events"]) == 2
        assert data["max_retries"] == 3
        assert data["enabled"] is True
        assert data["secret"].startswith("whsec_")

    def test_create_webhook_minimal(self, client):
        """Test creating webhook with minimal fields."""
        response = client.post(
            "/webhooks",
            json={"url": "https://example.com/webhook"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["url"] == "https://example.com/webhook"
        assert data["description"] == ""
        assert data["events"] == []  # All events
        assert data["max_retries"] == 5  # Default

    def test_create_webhook_invalid_url(self, client):
        """Test creating webhook with invalid URL."""
        response = client.post(
            "/webhooks",
            json={"url": "not-a-valid-url"},
        )

        assert response.status_code == 422

    def test_create_webhook_with_custom_headers(self, client):
        """Test creating webhook with custom headers."""
        response = client.post(
            "/webhooks",
            json={
                "url": "https://example.com/webhook",
                "custom_headers": {"Authorization": "Bearer token"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["custom_headers"]["Authorization"] == "Bearer token"


# ============================================================================
# List Webhooks Tests
# ============================================================================


class TestListWebhooks:
    """Tests for GET /webhooks endpoint."""

    def test_list_webhooks_empty(self, client):
        """Test listing webhooks when none exist."""
        response = client.get("/webhooks")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_webhooks(self, client, sample_webhook, manager):  # noqa: ARG002
        """Test listing all webhooks (sample_webhook creates first webhook)."""
        # Create another webhook
        manager.register(url="https://example2.com/webhook")

        response = client.get("/webhooks")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_webhooks_enabled_only(self, client, sample_webhook, manager):
        """Test listing only enabled webhooks."""
        # Disable the sample webhook
        manager.update(sample_webhook.id, enabled=False)
        # Create an enabled one
        manager.register(url="https://example2.com/webhook")

        response = client.get("/webhooks?enabled_only=true")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["enabled"] is True

    def test_list_webhooks_by_event_type(self, client, manager):
        """Test listing webhooks by event type."""
        manager.register(
            url="https://example1.com/webhook",
            events=[WebhookEventType.ANALYSIS_STARTED],
        )
        manager.register(
            url="https://example2.com/webhook",
            events=[WebhookEventType.APPROVAL_REQUIRED],
        )

        response = client.get("/webhooks?event_type=analysis.started")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1


# ============================================================================
# Get Webhook Tests
# ============================================================================


class TestGetWebhook:
    """Tests for GET /webhooks/{webhook_id} endpoint."""

    def test_get_webhook(self, client, sample_webhook):
        """Test getting a specific webhook."""
        response = client.get(f"/webhooks/{sample_webhook.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_webhook.id
        assert data["url"] == str(sample_webhook.url)

    def test_get_webhook_not_found(self, client):
        """Test getting a non-existent webhook."""
        response = client.get("/webhooks/non-existent-id")

        assert response.status_code == 404
        # The app has a custom error handler that puts the message in "error" field
        data = response.json()
        error_msg = data.get("error") or data.get("detail") or ""
        assert "not found" in error_msg.lower()


# ============================================================================
# Update Webhook Tests
# ============================================================================


class TestUpdateWebhook:
    """Tests for PUT /webhooks/{webhook_id} endpoint."""

    def test_update_webhook_url(self, client, sample_webhook):
        """Test updating webhook URL."""
        response = client.put(
            f"/webhooks/{sample_webhook.id}",
            json={"url": "https://new.example.com/webhook"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["url"] == "https://new.example.com/webhook"

    def test_update_webhook_description(self, client, sample_webhook):
        """Test updating webhook description."""
        response = client.put(
            f"/webhooks/{sample_webhook.id}",
            json={"description": "New description"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "New description"

    def test_update_webhook_enabled(self, client, sample_webhook):
        """Test enabling/disabling webhook."""
        response = client.put(
            f"/webhooks/{sample_webhook.id}",
            json={"enabled": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

    def test_update_webhook_events(self, client, sample_webhook):
        """Test updating webhook events."""
        response = client.put(
            f"/webhooks/{sample_webhook.id}",
            json={"events": ["approval.required", "approval.granted"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) == 2

    def test_update_webhook_not_found(self, client):
        """Test updating non-existent webhook."""
        response = client.put(
            "/webhooks/non-existent-id",
            json={"description": "Test"},
        )

        assert response.status_code == 404


# ============================================================================
# Delete Webhook Tests
# ============================================================================


class TestDeleteWebhook:
    """Tests for DELETE /webhooks/{webhook_id} endpoint."""

    def test_delete_webhook(self, client, sample_webhook):
        """Test deleting a webhook."""
        response = client.delete(f"/webhooks/{sample_webhook.id}")

        assert response.status_code == 204

        # Verify it's gone
        response = client.get(f"/webhooks/{sample_webhook.id}")
        assert response.status_code == 404

    def test_delete_webhook_not_found(self, client):
        """Test deleting non-existent webhook."""
        response = client.delete("/webhooks/non-existent-id")

        assert response.status_code == 404


# ============================================================================
# Test Webhook Tests
# ============================================================================


class TestTestWebhook:
    """Tests for POST /webhooks/{webhook_id}/test endpoint."""

    def test_test_webhook_success(self, client, sample_webhook):
        """Test sending test event to webhook."""

        mock_delivery = WebhookDelivery(
            webhook_id=sample_webhook.id,
            event_id="test_evt_123",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            url=str(sample_webhook.url),
            status=WebhookDeliveryStatus.SUCCESS,
            response_status=200,
        )

        with patch(
            "src.api.webhooks.get_webhook_dispatcher"
        ) as mock_get_dispatcher:
            mock_dispatcher = MagicMock()
            mock_dispatcher.send_test_event = AsyncMock(return_value=mock_delivery)
            mock_get_dispatcher.return_value = mock_dispatcher

            response = client.post(f"/webhooks/{sample_webhook.id}/test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "success"
        assert data["response_status"] == 200

    def test_test_webhook_failure(self, client, sample_webhook):
        """Test failed test event delivery."""
        mock_delivery = WebhookDelivery(
            webhook_id=sample_webhook.id,
            event_id="test_evt_123",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            url=str(sample_webhook.url),
            status=WebhookDeliveryStatus.FAILED,
            response_status=500,
            error_message="Server error",
        )

        with patch(
            "src.api.webhooks.get_webhook_dispatcher"
        ) as mock_get_dispatcher:
            mock_dispatcher = MagicMock()
            mock_dispatcher.send_test_event = AsyncMock(return_value=mock_delivery)
            mock_get_dispatcher.return_value = mock_dispatcher

            response = client.post(f"/webhooks/{sample_webhook.id}/test")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["status"] == "failed"
        assert data["error_message"] == "Server error"

    def test_test_webhook_not_found(self, client):
        """Test sending test event to non-existent webhook."""
        with patch(
            "src.api.webhooks.get_webhook_dispatcher"
        ) as mock_get_dispatcher:
            mock_dispatcher = MagicMock()
            mock_dispatcher.send_test_event = AsyncMock(return_value=None)
            mock_get_dispatcher.return_value = mock_dispatcher

            response = client.post("/webhooks/non-existent-id/test")

        assert response.status_code == 404


# ============================================================================
# List Deliveries Tests
# ============================================================================


class TestListDeliveries:
    """Tests for GET /webhooks/{webhook_id}/deliveries endpoint."""

    def test_list_deliveries_empty(self, client, sample_webhook):
        """Test listing deliveries when none exist."""
        response = client.get(f"/webhooks/{sample_webhook.id}/deliveries")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_deliveries(self, client, sample_webhook, manager):
        """Test listing deliveries for a webhook."""
        # Create some deliveries
        manager.create_delivery(
            webhook=sample_webhook,
            event_id="evt_1",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={"test": True},
        )
        manager.create_delivery(
            webhook=sample_webhook,
            event_id="evt_2",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={"test": True},
        )

        response = client.get(f"/webhooks/{sample_webhook.id}/deliveries")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_deliveries_with_limit(self, client, sample_webhook, manager):
        """Test listing deliveries with limit."""
        for i in range(10):
            manager.create_delivery(
                webhook=sample_webhook,
                event_id=f"evt_{i}",
                event_type=WebhookEventType.ANALYSIS_STARTED,
                payload={},
            )

        response = client.get(f"/webhooks/{sample_webhook.id}/deliveries?limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_list_deliveries_by_status(self, client, sample_webhook, manager):
        """Test listing deliveries by status."""
        d1 = manager.create_delivery(
            webhook=sample_webhook,
            event_id="evt_1",
            event_type=WebhookEventType.ANALYSIS_STARTED,
            payload={},
        )
        manager.create_delivery(
            webhook=sample_webhook,
            event_id="evt_2",
            event_type=WebhookEventType.ANALYSIS_COMPLETED,
            payload={},
        )
        d1.mark_success(response_status=200)

        response = client.get(
            f"/webhooks/{sample_webhook.id}/deliveries?status=success"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "success"

    def test_list_deliveries_webhook_not_found(self, client):
        """Test listing deliveries for non-existent webhook."""
        response = client.get("/webhooks/non-existent-id/deliveries")

        assert response.status_code == 404
