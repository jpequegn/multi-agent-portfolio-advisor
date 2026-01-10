"""Webhook registration and management.

Provides storage and management of webhook configurations,
delivery tracking, and subscription management.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field, HttpUrl

from src.webhooks.events import WebhookEventType

logger = structlog.get_logger(__name__)


class WebhookDeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookConfig(BaseModel):
    """Configuration for a webhook subscription."""

    # Event filtering
    events: list[WebhookEventType] = Field(
        default_factory=list,
        description="Event types to subscribe to (empty = all events)",
    )

    # Retry configuration
    max_retries: int = Field(
        default=5,
        description="Maximum retry attempts",
        ge=0,
        le=10,
    )
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout in seconds",
        ge=5,
        le=120,
    )

    # Headers to include
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include in requests",
    )


class Webhook(BaseModel):
    """A registered webhook subscription."""

    id: str = Field(
        default_factory=lambda: f"wh_{uuid.uuid4().hex[:12]}",
        description="Unique webhook identifier",
    )
    url: HttpUrl = Field(
        ..., description="Webhook endpoint URL"
    )
    secret: str = Field(
        default_factory=lambda: f"whsec_{uuid.uuid4().hex}",
        description="Secret key for HMAC signature",
    )
    description: str = Field(
        default="",
        description="Human-readable description",
    )
    config: WebhookConfig = Field(
        default_factory=WebhookConfig,
        description="Webhook configuration",
    )
    enabled: bool = Field(
        default=True,
        description="Whether webhook is active",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When webhook was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When webhook was last updated",
    )

    # Statistics
    total_deliveries: int = Field(
        default=0,
        description="Total delivery attempts",
    )
    successful_deliveries: int = Field(
        default=0,
        description="Successful deliveries",
    )
    failed_deliveries: int = Field(
        default=0,
        description="Failed deliveries",
    )
    last_delivery_at: datetime | None = Field(
        default=None,
        description="Last delivery timestamp",
    )
    last_success_at: datetime | None = Field(
        default=None,
        description="Last successful delivery",
    )
    last_failure_at: datetime | None = Field(
        default=None,
        description="Last failed delivery",
    )

    def should_receive_event(self, event_type: WebhookEventType) -> bool:
        """Check if this webhook should receive an event type.

        Args:
            event_type: Event type to check.

        Returns:
            True if webhook subscribes to this event type.
        """
        # Empty events list means subscribe to all
        if not self.config.events:
            return True
        return event_type in self.config.events

    def record_delivery(self, success: bool) -> None:
        """Record a delivery attempt.

        Args:
            success: Whether delivery was successful.
        """
        self.total_deliveries += 1
        self.last_delivery_at = datetime.now(UTC)

        if success:
            self.successful_deliveries += 1
            self.last_success_at = datetime.now(UTC)
        else:
            self.failed_deliveries += 1
            self.last_failure_at = datetime.now(UTC)


class WebhookDelivery(BaseModel):
    """Record of a webhook delivery attempt."""

    id: str = Field(
        default_factory=lambda: f"dlv_{uuid.uuid4().hex[:12]}",
        description="Unique delivery identifier",
    )
    webhook_id: str = Field(
        ..., description="Associated webhook ID"
    )
    event_id: str = Field(
        ..., description="Event ID that triggered delivery"
    )
    event_type: WebhookEventType = Field(
        ..., description="Type of event"
    )
    status: WebhookDeliveryStatus = Field(
        default=WebhookDeliveryStatus.PENDING,
        description="Current delivery status",
    )
    url: str = Field(
        ..., description="Target URL"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Payload that was sent",
    )

    # Attempt tracking
    attempt_count: int = Field(
        default=0,
        description="Number of delivery attempts",
    )
    max_attempts: int = Field(
        default=5,
        description="Maximum attempts before failure",
    )
    last_attempt_at: datetime | None = Field(
        default=None,
        description="Last attempt timestamp",
    )
    next_attempt_at: datetime | None = Field(
        default=None,
        description="Next scheduled attempt",
    )

    # Response tracking
    response_status: int | None = Field(
        default=None,
        description="HTTP response status code",
    )
    response_body: str | None = Field(
        default=None,
        description="Response body (truncated)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if failed",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When delivery was created",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="When delivery completed (success or final failure)",
    )

    def mark_success(self, response_status: int, response_body: str | None = None) -> None:
        """Mark delivery as successful.

        Args:
            response_status: HTTP status code.
            response_body: Optional response body.
        """
        self.status = WebhookDeliveryStatus.SUCCESS
        self.response_status = response_status
        self.response_body = response_body[:1000] if response_body else None
        self.completed_at = datetime.now(UTC)
        self.next_attempt_at = None

    def mark_failed(
        self,
        error_message: str,
        response_status: int | None = None,
        can_retry: bool = True,
    ) -> None:
        """Mark delivery attempt as failed.

        Args:
            error_message: Error description.
            response_status: Optional HTTP status code.
            can_retry: Whether to schedule a retry.
        """
        self.attempt_count += 1
        self.last_attempt_at = datetime.now(UTC)
        self.error_message = error_message
        self.response_status = response_status

        if can_retry and self.attempt_count < self.max_attempts:
            self.status = WebhookDeliveryStatus.RETRYING
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            delay_seconds = 2 ** (self.attempt_count - 1)
            from datetime import timedelta
            self.next_attempt_at = datetime.now(UTC) + timedelta(seconds=delay_seconds)
        else:
            self.status = WebhookDeliveryStatus.FAILED
            self.completed_at = datetime.now(UTC)
            self.next_attempt_at = None


# Global webhook manager instance
_webhook_manager: WebhookManager | None = None


class WebhookManager:
    """Manages webhook registrations and deliveries.

    Provides CRUD operations for webhooks and tracking of
    delivery attempts.
    """

    def __init__(self) -> None:
        """Initialize the webhook manager."""
        self._webhooks: dict[str, Webhook] = {}
        self._deliveries: dict[str, WebhookDelivery] = {}
        self._event_handlers: list[Callable[[WebhookEventType, dict[str, Any]], None]] = []
        self._logger = logger.bind(component="webhook_manager")

    def register(
        self,
        url: str,
        *,
        description: str = "",
        events: list[WebhookEventType] | None = None,
        config: WebhookConfig | None = None,
    ) -> Webhook:
        """Register a new webhook.

        Args:
            url: Webhook endpoint URL.
            description: Human-readable description.
            events: Event types to subscribe to (None = all).
            config: Optional webhook configuration.

        Returns:
            Created webhook.
        """
        webhook_config = config or WebhookConfig()
        if events:
            webhook_config.events = events

        webhook = Webhook(
            url=url,  # type: ignore[arg-type]
            description=description,
            config=webhook_config,
        )

        self._webhooks[webhook.id] = webhook

        self._logger.info(
            "webhook_registered",
            webhook_id=webhook.id,
            url=str(webhook.url),
            event_count=len(webhook_config.events),
        )

        return webhook

    def get(self, webhook_id: str) -> Webhook | None:
        """Get a webhook by ID.

        Args:
            webhook_id: Webhook identifier.

        Returns:
            Webhook if found, None otherwise.
        """
        return self._webhooks.get(webhook_id)

    def list_all(
        self,
        *,
        enabled_only: bool = False,
        event_type: WebhookEventType | None = None,
    ) -> list[Webhook]:
        """List registered webhooks.

        Args:
            enabled_only: Only return enabled webhooks.
            event_type: Filter by event type subscription.

        Returns:
            List of matching webhooks.
        """
        webhooks = list(self._webhooks.values())

        if enabled_only:
            webhooks = [w for w in webhooks if w.enabled]

        if event_type:
            webhooks = [w for w in webhooks if w.should_receive_event(event_type)]

        return webhooks

    def update(
        self,
        webhook_id: str,
        *,
        url: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        config: WebhookConfig | None = None,
    ) -> Webhook | None:
        """Update a webhook.

        Args:
            webhook_id: Webhook identifier.
            url: New URL.
            description: New description.
            enabled: New enabled state.
            config: New configuration.

        Returns:
            Updated webhook if found, None otherwise.
        """
        webhook = self._webhooks.get(webhook_id)
        if not webhook:
            return None

        if url is not None:
            webhook.url = url  # type: ignore[assignment]
        if description is not None:
            webhook.description = description
        if enabled is not None:
            webhook.enabled = enabled
        if config is not None:
            webhook.config = config

        webhook.updated_at = datetime.now(UTC)

        self._logger.info(
            "webhook_updated",
            webhook_id=webhook_id,
        )

        return webhook

    def delete(self, webhook_id: str) -> bool:
        """Delete a webhook.

        Args:
            webhook_id: Webhook identifier.

        Returns:
            True if deleted, False if not found.
        """
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            self._logger.info("webhook_deleted", webhook_id=webhook_id)
            return True
        return False

    def get_webhooks_for_event(self, event_type: WebhookEventType) -> list[Webhook]:
        """Get all webhooks that should receive an event type.

        Args:
            event_type: Event type.

        Returns:
            List of enabled webhooks subscribed to event.
        """
        return [
            webhook
            for webhook in self._webhooks.values()
            if webhook.enabled and webhook.should_receive_event(event_type)
        ]

    def create_delivery(
        self,
        webhook: Webhook,
        event_id: str,
        event_type: WebhookEventType,
        payload: dict[str, Any],
    ) -> WebhookDelivery:
        """Create a delivery record for tracking.

        Args:
            webhook: Target webhook.
            event_id: Event identifier.
            event_type: Type of event.
            payload: Event payload.

        Returns:
            Created delivery record.
        """
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event_id,
            event_type=event_type,
            url=str(webhook.url),
            payload=payload,
            max_attempts=webhook.config.max_retries + 1,  # +1 for initial attempt
        )

        self._deliveries[delivery.id] = delivery

        self._logger.debug(
            "delivery_created",
            delivery_id=delivery.id,
            webhook_id=webhook.id,
            event_id=event_id,
        )

        return delivery

    def get_delivery(self, delivery_id: str) -> WebhookDelivery | None:
        """Get a delivery by ID.

        Args:
            delivery_id: Delivery identifier.

        Returns:
            Delivery if found, None otherwise.
        """
        return self._deliveries.get(delivery_id)

    def list_deliveries(
        self,
        webhook_id: str,
        *,
        limit: int = 50,
        status: WebhookDeliveryStatus | None = None,
    ) -> list[WebhookDelivery]:
        """List deliveries for a webhook.

        Args:
            webhook_id: Webhook identifier.
            limit: Maximum results.
            status: Filter by status.

        Returns:
            List of deliveries.
        """
        deliveries = [
            d for d in self._deliveries.values()
            if d.webhook_id == webhook_id
        ]

        if status:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by created_at descending
        deliveries.sort(key=lambda d: d.created_at, reverse=True)

        return deliveries[:limit]

    def get_pending_retries(self) -> list[WebhookDelivery]:
        """Get deliveries that are pending retry.

        Returns:
            List of deliveries ready for retry.
        """
        now = datetime.now(UTC)
        return [
            d for d in self._deliveries.values()
            if d.status == WebhookDeliveryStatus.RETRYING
            and d.next_attempt_at
            and d.next_attempt_at <= now
        ]


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance.

    Returns:
        Singleton WebhookManager.
    """
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


def set_webhook_manager(manager: WebhookManager) -> None:
    """Set the global webhook manager instance.

    Useful for testing.

    Args:
        manager: WebhookManager instance.
    """
    global _webhook_manager
    _webhook_manager = manager
