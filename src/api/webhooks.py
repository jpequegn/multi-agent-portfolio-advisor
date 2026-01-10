"""Webhook management API endpoints.

Provides REST API for managing webhook registrations and
viewing delivery history.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from src.webhooks.dispatcher import get_webhook_dispatcher
from src.webhooks.events import WebhookEventType
from src.webhooks.manager import (
    Webhook,
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    get_webhook_manager,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ============================================================================
# Request Models
# ============================================================================


class WebhookCreateRequest(BaseModel):
    """Request to create a new webhook."""

    url: HttpUrl = Field(
        ..., description="Webhook endpoint URL"
    )
    description: str = Field(
        default="",
        description="Human-readable description",
    )
    events: list[WebhookEventType] | None = Field(
        default=None,
        description="Event types to subscribe to (null = all)",
    )
    max_retries: int = Field(
        default=5,
        description="Maximum retry attempts",
        ge=0,
        le=10,
    )
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout",
        ge=5,
        le=120,
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include",
    )


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook."""

    url: HttpUrl | None = Field(
        default=None, description="New URL"
    )
    description: str | None = Field(
        default=None, description="New description"
    )
    enabled: bool | None = Field(
        default=None, description="Enable/disable webhook"
    )
    events: list[WebhookEventType] | None = Field(
        default=None, description="New event subscriptions"
    )
    max_retries: int | None = Field(
        default=None, description="New max retries"
    )
    timeout_seconds: int | None = Field(
        default=None, description="New timeout"
    )
    custom_headers: dict[str, str] | None = Field(
        default=None, description="New custom headers"
    )


# ============================================================================
# Response Models
# ============================================================================


class WebhookResponse(BaseModel):
    """Webhook details response."""

    id: str
    url: str
    secret: str
    description: str
    events: list[WebhookEventType]
    enabled: bool
    max_retries: int
    timeout_seconds: int
    custom_headers: dict[str, str]
    created_at: str
    updated_at: str
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    last_delivery_at: str | None
    last_success_at: str | None
    last_failure_at: str | None

    @classmethod
    def from_webhook(cls, webhook: Webhook) -> "WebhookResponse":
        """Create response from Webhook model."""
        return cls(
            id=webhook.id,
            url=str(webhook.url),
            secret=webhook.secret,
            description=webhook.description,
            events=webhook.config.events,
            enabled=webhook.enabled,
            max_retries=webhook.config.max_retries,
            timeout_seconds=webhook.config.timeout_seconds,
            custom_headers=webhook.config.custom_headers,
            created_at=webhook.created_at.isoformat(),
            updated_at=webhook.updated_at.isoformat(),
            total_deliveries=webhook.total_deliveries,
            successful_deliveries=webhook.successful_deliveries,
            failed_deliveries=webhook.failed_deliveries,
            last_delivery_at=webhook.last_delivery_at.isoformat() if webhook.last_delivery_at else None,
            last_success_at=webhook.last_success_at.isoformat() if webhook.last_success_at else None,
            last_failure_at=webhook.last_failure_at.isoformat() if webhook.last_failure_at else None,
        )


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery details response."""

    id: str
    webhook_id: str
    event_id: str
    event_type: WebhookEventType
    status: WebhookDeliveryStatus
    url: str
    payload: dict[str, Any]
    attempt_count: int
    max_attempts: int
    last_attempt_at: str | None
    next_attempt_at: str | None
    response_status: int | None
    error_message: str | None
    created_at: str
    completed_at: str | None

    @classmethod
    def from_delivery(cls, delivery: WebhookDelivery) -> "WebhookDeliveryResponse":
        """Create response from WebhookDelivery model."""
        return cls(
            id=delivery.id,
            webhook_id=delivery.webhook_id,
            event_id=delivery.event_id,
            event_type=delivery.event_type,
            status=delivery.status,
            url=delivery.url,
            payload=delivery.payload,
            attempt_count=delivery.attempt_count,
            max_attempts=delivery.max_attempts,
            last_attempt_at=delivery.last_attempt_at.isoformat() if delivery.last_attempt_at else None,
            next_attempt_at=delivery.next_attempt_at.isoformat() if delivery.next_attempt_at else None,
            response_status=delivery.response_status,
            error_message=delivery.error_message,
            created_at=delivery.created_at.isoformat(),
            completed_at=delivery.completed_at.isoformat() if delivery.completed_at else None,
        )


class TestWebhookResponse(BaseModel):
    """Response from test webhook endpoint."""

    success: bool
    delivery_id: str
    status: WebhookDeliveryStatus
    response_status: int | None
    error_message: str | None


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "",
    response_model=WebhookResponse,
    responses={
        201: {"description": "Webhook created"},
        400: {"description": "Invalid request"},
    },
    status_code=201,
)
async def create_webhook(request: WebhookCreateRequest) -> WebhookResponse:
    """Register a new webhook.

    Creates a webhook subscription that will receive events at the specified URL.
    A secret key is generated for HMAC signature verification.
    """
    manager = get_webhook_manager()

    config = WebhookConfig(
        events=request.events or [],
        max_retries=request.max_retries,
        timeout_seconds=request.timeout_seconds,
        custom_headers=request.custom_headers,
    )

    webhook = manager.register(
        url=str(request.url),
        description=request.description,
        config=config,
    )

    logger.info(
        "webhook_created",
        webhook_id=webhook.id,
        url=str(webhook.url),
    )

    return WebhookResponse.from_webhook(webhook)


@router.get(
    "",
    response_model=list[WebhookResponse],
)
async def list_webhooks(
    enabled_only: bool = False,
    event_type: WebhookEventType | None = None,
) -> list[WebhookResponse]:
    """List all registered webhooks.

    Optionally filter by enabled status or event type subscription.
    """
    manager = get_webhook_manager()
    webhooks = manager.list_all(enabled_only=enabled_only, event_type=event_type)
    return [WebhookResponse.from_webhook(w) for w in webhooks]


@router.get(
    "/{webhook_id}",
    response_model=WebhookResponse,
    responses={
        404: {"description": "Webhook not found"},
    },
)
async def get_webhook(webhook_id: str) -> WebhookResponse:
    """Get webhook details by ID."""
    manager = get_webhook_manager()
    webhook = manager.get(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    return WebhookResponse.from_webhook(webhook)


@router.put(
    "/{webhook_id}",
    response_model=WebhookResponse,
    responses={
        404: {"description": "Webhook not found"},
    },
)
async def update_webhook(
    webhook_id: str,
    request: WebhookUpdateRequest,
) -> WebhookResponse:
    """Update a webhook."""
    manager = get_webhook_manager()
    webhook = manager.get(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    # Build updated config if needed
    config = None
    if any([
        request.events is not None,
        request.max_retries is not None,
        request.timeout_seconds is not None,
        request.custom_headers is not None,
    ]):
        config = WebhookConfig(
            events=request.events if request.events is not None else webhook.config.events,
            max_retries=request.max_retries if request.max_retries is not None else webhook.config.max_retries,
            timeout_seconds=request.timeout_seconds if request.timeout_seconds is not None else webhook.config.timeout_seconds,
            custom_headers=request.custom_headers if request.custom_headers is not None else webhook.config.custom_headers,
        )

    updated = manager.update(
        webhook_id,
        url=str(request.url) if request.url else None,
        description=request.description,
        enabled=request.enabled,
        config=config,
    )

    if not updated:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    logger.info(
        "webhook_updated",
        webhook_id=webhook_id,
    )

    return WebhookResponse.from_webhook(updated)


@router.delete(
    "/{webhook_id}",
    responses={
        204: {"description": "Webhook deleted"},
        404: {"description": "Webhook not found"},
    },
    status_code=204,
)
async def delete_webhook(webhook_id: str) -> None:
    """Delete a webhook."""
    manager = get_webhook_manager()
    deleted = manager.delete(webhook_id)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    logger.info("webhook_deleted", webhook_id=webhook_id)


@router.post(
    "/{webhook_id}/test",
    response_model=TestWebhookResponse,
    responses={
        404: {"description": "Webhook not found"},
    },
)
async def test_webhook(webhook_id: str) -> TestWebhookResponse:
    """Send a test event to a webhook.

    Sends a test event with a sample payload to verify
    the webhook endpoint is working correctly.
    """
    dispatcher = get_webhook_dispatcher()
    delivery = await dispatcher.send_test_event(webhook_id)

    if not delivery:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    logger.info(
        "webhook_tested",
        webhook_id=webhook_id,
        delivery_id=delivery.id,
        status=delivery.status.value,
    )

    return TestWebhookResponse(
        success=delivery.status == WebhookDeliveryStatus.SUCCESS,
        delivery_id=delivery.id,
        status=delivery.status,
        response_status=delivery.response_status,
        error_message=delivery.error_message,
    )


@router.get(
    "/{webhook_id}/deliveries",
    response_model=list[WebhookDeliveryResponse],
    responses={
        404: {"description": "Webhook not found"},
    },
)
async def list_webhook_deliveries(
    webhook_id: str,
    limit: int = 50,
    status: WebhookDeliveryStatus | None = None,
) -> list[WebhookDeliveryResponse]:
    """List delivery attempts for a webhook.

    Returns recent delivery attempts with status and response details.
    """
    manager = get_webhook_manager()
    webhook = manager.get(webhook_id)

    if not webhook:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found")

    deliveries = manager.list_deliveries(webhook_id, limit=limit, status=status)
    return [WebhookDeliveryResponse.from_delivery(d) for d in deliveries]
