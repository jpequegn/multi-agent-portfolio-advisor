"""Webhook notification system for external integrations.

This module provides:
- WebhookEventType: Enumeration of all webhook event types
- WebhookPayload: Standard payload structure for webhook deliveries
- WebhookManager: Registration and management of webhooks
- WebhookDispatcher: Event dispatch with retry logic
- HMAC signature verification for security
"""

from src.webhooks.dispatcher import WebhookDispatcher, get_webhook_dispatcher
from src.webhooks.events import WebhookEventType, WebhookPayload, create_webhook_event
from src.webhooks.manager import (
    Webhook,
    WebhookConfig,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookManager,
    get_webhook_manager,
)
from src.webhooks.security import generate_signature, verify_signature

__all__ = [
    # Events
    "WebhookEventType",
    "WebhookPayload",
    "create_webhook_event",
    # Manager
    "Webhook",
    "WebhookConfig",
    "WebhookDelivery",
    "WebhookDeliveryStatus",
    "WebhookManager",
    "get_webhook_manager",
    # Dispatcher
    "WebhookDispatcher",
    "get_webhook_dispatcher",
    # Security
    "generate_signature",
    "verify_signature",
]
