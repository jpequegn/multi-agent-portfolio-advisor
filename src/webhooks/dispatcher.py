"""Webhook event dispatcher with retry logic.

Handles sending webhook payloads to registered endpoints with
exponential backoff retry for failures.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from src.webhooks.events import WebhookEventType, WebhookPayload
from src.webhooks.manager import (
    Webhook,
    WebhookDelivery,
    WebhookDeliveryStatus,
    WebhookManager,
    get_webhook_manager,
)
from src.webhooks.security import create_signature_headers

logger = structlog.get_logger(__name__)

# Type for event listeners
EventListener = Callable[[WebhookPayload], Awaitable[None] | None]


class WebhookDispatcher:
    """Dispatches webhook events to registered endpoints.

    Features:
    - Async HTTP delivery with configurable timeout
    - Exponential backoff retry for failures
    - HMAC signature for security
    - Delivery tracking for debugging
    - Event listeners for local handling
    """

    def __init__(
        self,
        manager: WebhookManager | None = None,
        *,
        delivery_timeout: float = 30.0,
        max_concurrent_deliveries: int = 10,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            manager: Webhook manager (uses global if not provided).
            delivery_timeout: HTTP request timeout in seconds.
            max_concurrent_deliveries: Max concurrent delivery requests.
        """
        self._manager = manager or get_webhook_manager()
        self._delivery_timeout = delivery_timeout
        self._max_concurrent = max_concurrent_deliveries
        self._semaphore = asyncio.Semaphore(max_concurrent_deliveries)
        self._listeners: list[EventListener] = []
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._logger = logger.bind(component="webhook_dispatcher")

    def add_listener(self, listener: EventListener) -> None:
        """Add a local event listener.

        Listeners are called synchronously for every event,
        useful for local processing in addition to webhooks.

        Args:
            listener: Async or sync function to call with events.
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        """Remove a local event listener.

        Args:
            listener: Listener to remove.
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def dispatch(
        self,
        event: WebhookPayload,
        *,
        wait: bool = False,
    ) -> list[WebhookDelivery]:
        """Dispatch an event to all subscribed webhooks.

        Args:
            event: Event payload to send.
            wait: If True, wait for all deliveries to complete.

        Returns:
            List of delivery records created.
        """
        self._logger.info(
            "dispatching_event",
            event_id=event.id,
            event_type=event.type.value,
        )

        # Notify local listeners
        await self._notify_listeners(event)

        # Get webhooks subscribed to this event type
        webhooks = self._manager.get_webhooks_for_event(event.type)

        if not webhooks:
            self._logger.debug(
                "no_webhooks_subscribed",
                event_type=event.type.value,
            )
            return []

        # Create delivery records
        deliveries: list[WebhookDelivery] = []
        tasks: list[asyncio.Task[None]] = []

        for webhook in webhooks:
            delivery = self._manager.create_delivery(
                webhook=webhook,
                event_id=event.id,
                event_type=event.type,
                payload=event.to_json_dict(),
            )
            deliveries.append(delivery)

            # Schedule delivery task
            task = asyncio.create_task(
                self._deliver_with_retry(webhook, delivery)
            )
            tasks.append(task)

            if not wait:
                # Track background task
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        if wait and tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._logger.info(
            "event_dispatched",
            event_id=event.id,
            webhook_count=len(webhooks),
            delivery_count=len(deliveries),
        )

        return deliveries

    async def dispatch_event(
        self,
        event_type: WebhookEventType,
        data: dict[str, Any],
        *,
        wait: bool = False,
    ) -> list[WebhookDelivery]:
        """Convenience method to dispatch by event type and data.

        Args:
            event_type: Type of event.
            data: Event data.
            wait: If True, wait for deliveries.

        Returns:
            List of delivery records.
        """
        event = WebhookPayload(type=event_type, data=data)
        return await self.dispatch(event, wait=wait)

    async def _notify_listeners(self, event: WebhookPayload) -> None:
        """Notify local event listeners.

        Args:
            event: Event to notify about.
        """
        for listener in self._listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._logger.warning(
                    "listener_error",
                    event_id=event.id,
                    error=str(e),
                )

    async def _deliver_with_retry(
        self,
        webhook: Webhook,
        delivery: WebhookDelivery,
    ) -> None:
        """Deliver a webhook with retry logic.

        Args:
            webhook: Target webhook.
            delivery: Delivery record to update.
        """
        max_attempts = webhook.config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                async with self._semaphore:
                    success = await self._attempt_delivery(webhook, delivery)

                if success:
                    webhook.record_delivery(success=True)
                    return

            except Exception as e:
                self._logger.warning(
                    "delivery_attempt_failed",
                    delivery_id=delivery.id,
                    attempt=attempt,
                    error=str(e),
                )

            # Check if we should retry
            if attempt < max_attempts:
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                delay = 2 ** (attempt - 1)
                self._logger.debug(
                    "scheduling_retry",
                    delivery_id=delivery.id,
                    delay_seconds=delay,
                    next_attempt=attempt + 1,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        webhook.record_delivery(success=False)
        self._logger.error(
            "delivery_failed_permanently",
            delivery_id=delivery.id,
            webhook_id=webhook.id,
            attempts=max_attempts,
        )

    async def _attempt_delivery(
        self,
        webhook: Webhook,
        delivery: WebhookDelivery,
    ) -> bool:
        """Make a single delivery attempt.

        Args:
            webhook: Target webhook.
            delivery: Delivery record.

        Returns:
            True if successful.
        """
        delivery.attempt_count += 1
        delivery.last_attempt_at = datetime.now(UTC)
        delivery.status = WebhookDeliveryStatus.RETRYING

        payload = delivery.payload

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PortfolioAdvisor-Webhook/1.0",
            **webhook.config.custom_headers,
        }

        # Add HMAC signature
        signature_headers = create_signature_headers(payload, webhook.secret)
        headers.update(signature_headers)

        self._logger.debug(
            "attempting_delivery",
            delivery_id=delivery.id,
            attempt=delivery.attempt_count,
            url=str(webhook.url),
        )

        try:
            async with httpx.AsyncClient(timeout=webhook.config.timeout_seconds) as client:
                response = await client.post(
                    str(webhook.url),
                    json=payload,
                    headers=headers,
                )

                if response.is_success:
                    delivery.mark_success(
                        response_status=response.status_code,
                        response_body=response.text,
                    )
                    self._logger.info(
                        "delivery_success",
                        delivery_id=delivery.id,
                        webhook_id=webhook.id,
                        status_code=response.status_code,
                    )
                    return True
                else:
                    # Non-2xx response
                    can_retry = response.status_code >= 500  # Only retry 5xx
                    delivery.mark_failed(
                        error_message=f"HTTP {response.status_code}",
                        response_status=response.status_code,
                        can_retry=can_retry,
                    )
                    self._logger.warning(
                        "delivery_non_success_response",
                        delivery_id=delivery.id,
                        status_code=response.status_code,
                        can_retry=can_retry,
                    )
                    return False

        except httpx.TimeoutException:
            delivery.mark_failed(
                error_message="Request timeout",
                can_retry=True,
            )
            self._logger.warning(
                "delivery_timeout",
                delivery_id=delivery.id,
                timeout=webhook.config.timeout_seconds,
            )
            return False

        except httpx.ConnectError as e:
            delivery.mark_failed(
                error_message=f"Connection error: {e}",
                can_retry=True,
            )
            self._logger.warning(
                "delivery_connection_error",
                delivery_id=delivery.id,
                error=str(e),
            )
            return False

        except Exception as e:
            delivery.mark_failed(
                error_message=str(e),
                can_retry=True,
            )
            self._logger.warning(
                "delivery_unexpected_error",
                delivery_id=delivery.id,
                error=str(e),
            )
            return False

    async def retry_pending(self) -> int:
        """Retry all pending deliveries.

        Returns:
            Number of deliveries retried.
        """
        pending = self._manager.get_pending_retries()
        if not pending:
            return 0

        self._logger.info(
            "retrying_pending_deliveries",
            count=len(pending),
        )

        tasks = []
        for delivery in pending:
            webhook = self._manager.get(delivery.webhook_id)
            if webhook and webhook.enabled:
                task = asyncio.create_task(
                    self._deliver_with_retry(webhook, delivery)
                )
                tasks.append(task)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        return len(tasks)

    async def send_test_event(self, webhook_id: str) -> WebhookDelivery | None:
        """Send a test event to a specific webhook.

        Args:
            webhook_id: Webhook to test.

        Returns:
            Delivery record if webhook exists, None otherwise.
        """
        webhook = self._manager.get(webhook_id)
        if not webhook:
            return None

        # Create test event
        test_event = WebhookPayload(
            type=WebhookEventType.ANALYSIS_COMPLETED,
            data={
                "test": True,
                "message": "This is a test webhook delivery",
                "webhook_id": webhook_id,
            },
        )
        test_event.id = f"test_{test_event.id}"

        # Create delivery
        delivery = self._manager.create_delivery(
            webhook=webhook,
            event_id=test_event.id,
            event_type=test_event.type,
            payload=test_event.to_json_dict(),
        )

        # Attempt delivery synchronously
        await self._deliver_with_retry(webhook, delivery)

        return delivery

    async def shutdown(self) -> None:
        """Shutdown dispatcher and wait for pending deliveries."""
        if self._background_tasks:
            self._logger.info(
                "waiting_for_pending_deliveries",
                count=len(self._background_tasks),
            )
            await asyncio.gather(*self._background_tasks, return_exceptions=True)


# Global dispatcher instance
_dispatcher: WebhookDispatcher | None = None


def get_webhook_dispatcher() -> WebhookDispatcher:
    """Get the global webhook dispatcher.

    Returns:
        Singleton WebhookDispatcher.
    """
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WebhookDispatcher()
    return _dispatcher


def set_webhook_dispatcher(dispatcher: WebhookDispatcher) -> None:
    """Set the global webhook dispatcher.

    Useful for testing.

    Args:
        dispatcher: WebhookDispatcher instance.
    """
    global _dispatcher
    _dispatcher = dispatcher
