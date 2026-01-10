"""Tests for webhook dispatcher module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.webhooks.dispatcher import WebhookDispatcher, get_webhook_dispatcher
from src.webhooks.events import WebhookEventType, WebhookPayload
from src.webhooks.manager import (
    WebhookConfig,
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
def dispatcher(manager):
    """Create test webhook dispatcher."""
    return WebhookDispatcher(manager=manager, max_concurrent_deliveries=5)


@pytest.fixture
def sample_webhook(manager):
    """Create a sample webhook."""
    return manager.register(
        url="https://example.com/webhook",
        description="Test webhook",
    )


@pytest.fixture
def sample_event():
    """Create a sample event."""
    return WebhookPayload(
        type=WebhookEventType.ANALYSIS_STARTED,
        data={"trace_id": "trace-123", "workflow_id": "wf-456"},
    )


# ============================================================================
# Listener Tests
# ============================================================================


class TestListeners:
    """Tests for event listeners."""

    def test_add_listener(self, dispatcher):
        """Test adding a listener."""
        listener = MagicMock()
        dispatcher.add_listener(listener)

        assert listener in dispatcher._listeners

    def test_remove_listener(self, dispatcher):
        """Test removing a listener."""
        listener = MagicMock()
        dispatcher.add_listener(listener)
        dispatcher.remove_listener(listener)

        assert listener not in dispatcher._listeners

    def test_remove_nonexistent_listener(self, dispatcher):
        """Test removing a listener that doesn't exist."""
        listener = MagicMock()
        # Should not raise
        dispatcher.remove_listener(listener)

    @pytest.mark.asyncio
    async def test_listener_called_on_dispatch(self, dispatcher, sample_event):
        """Test that listeners are called when dispatching."""
        listener = MagicMock()
        dispatcher.add_listener(listener)

        await dispatcher.dispatch(sample_event)

        listener.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_async_listener_called(self, dispatcher, sample_event):
        """Test that async listeners are awaited."""
        async_listener = AsyncMock()
        dispatcher.add_listener(async_listener)

        await dispatcher.dispatch(sample_event)

        async_listener.assert_awaited_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_listener_error_does_not_stop_dispatch(
        self, dispatcher, sample_event
    ):
        """Test that listener errors don't stop other processing."""
        failing_listener = MagicMock(side_effect=Exception("Listener error"))
        success_listener = MagicMock()

        dispatcher.add_listener(failing_listener)
        dispatcher.add_listener(success_listener)

        # Should not raise
        await dispatcher.dispatch(sample_event)

        # Both should have been called
        failing_listener.assert_called_once()
        success_listener.assert_called_once()


# ============================================================================
# Dispatch Tests
# ============================================================================


class TestDispatch:
    """Tests for event dispatching."""

    @pytest.mark.asyncio
    async def test_dispatch_no_webhooks(self, dispatcher, sample_event):
        """Test dispatching with no webhooks registered."""
        deliveries = await dispatcher.dispatch(sample_event)

        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_dispatch_creates_deliveries(
        self, dispatcher, sample_webhook, sample_event, manager  # noqa: ARG002
    ):
        """Test that dispatching creates delivery records (manager unused but loads fixture)."""
        with patch.object(dispatcher, "_deliver_with_retry", new_callable=AsyncMock):
            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 1
        assert deliveries[0].webhook_id == sample_webhook.id
        assert deliveries[0].event_id == sample_event.id

    @pytest.mark.asyncio
    async def test_dispatch_filters_by_event_type(
        self, dispatcher, manager, sample_event
    ):
        """Test that dispatch only sends to subscribed webhooks."""
        # Register webhook for different event
        manager.register(
            url="https://example.com/webhook",
            events=[WebhookEventType.APPROVAL_REQUIRED],
        )

        with patch.object(dispatcher, "_deliver_with_retry", new_callable=AsyncMock):
            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_dispatch_to_multiple_webhooks(
        self, dispatcher, manager, sample_event
    ):
        """Test dispatching to multiple webhooks."""
        manager.register(url="https://example1.com/webhook")
        manager.register(url="https://example2.com/webhook")
        manager.register(url="https://example3.com/webhook")

        with patch.object(dispatcher, "_deliver_with_retry", new_callable=AsyncMock):
            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 3


# ============================================================================
# dispatch_event Tests
# ============================================================================


class TestDispatchEvent:
    """Tests for dispatch_event convenience method."""

    @pytest.mark.asyncio
    async def test_dispatch_event(self, dispatcher, sample_webhook):  # noqa: ARG002
        """Test dispatch_event creates payload and dispatches (sample_webhook registers webhook)."""
        with patch.object(dispatcher, "_deliver_with_retry", new_callable=AsyncMock):
            deliveries = await dispatcher.dispatch_event(
                event_type=WebhookEventType.ANALYSIS_STARTED,
                data={"trace_id": "trace-123"},
                wait=True,
            )

        assert len(deliveries) == 1


# ============================================================================
# Delivery Tests
# ============================================================================


class TestDelivery:
    """Tests for webhook delivery."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self, dispatcher, sample_webhook, sample_event):  # noqa: ARG002
        """Test successful HTTP delivery (sample_webhook registers webhook)."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = "OK"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 1
        assert deliveries[0].status == WebhookDeliveryStatus.SUCCESS
        assert deliveries[0].response_status == 200

    @pytest.mark.asyncio
    async def test_failed_delivery_5xx(self, dispatcher, sample_webhook, sample_event):
        """Test failed delivery with 5xx response."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500

        # Set max_retries to 0 to avoid retries
        sample_webhook.config.max_retries = 0

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 1
        assert deliveries[0].status == WebhookDeliveryStatus.FAILED
        assert deliveries[0].response_status == 500

    @pytest.mark.asyncio
    async def test_failed_delivery_4xx(self, dispatcher, sample_webhook, sample_event):  # noqa: ARG002
        """Test failed delivery with 4xx response (no retry, sample_webhook registers webhook)."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 400

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 1
        assert deliveries[0].status == WebhookDeliveryStatus.FAILED
        assert deliveries[0].response_status == 400

    @pytest.mark.asyncio
    async def test_delivery_timeout(self, dispatcher, sample_webhook, sample_event):
        """Test delivery timeout handling."""
        import httpx

        sample_webhook.config.max_retries = 0

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 1
        assert deliveries[0].status == WebhookDeliveryStatus.FAILED
        assert "timeout" in deliveries[0].error_message.lower()

    @pytest.mark.asyncio
    async def test_delivery_connection_error(
        self, dispatcher, sample_webhook, sample_event
    ):
        """Test delivery connection error handling."""
        import httpx

        sample_webhook.config.max_retries = 0

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert len(deliveries) == 1
        assert deliveries[0].status == WebhookDeliveryStatus.FAILED
        assert "connection" in deliveries[0].error_message.lower()


# ============================================================================
# Retry Tests
# ============================================================================


class TestRetry:
    """Tests for retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_5xx(self, dispatcher, manager, sample_event):
        """Test that 5xx errors trigger retry."""
        manager.register(
            url="https://example.com/webhook",
            config=WebhookConfig(max_retries=2),
        )

        call_count = 0

        async def mock_post(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1

            mock_response = MagicMock()
            if call_count < 3:
                mock_response.is_success = False
                mock_response.status_code = 500
            else:
                mock_response.is_success = True
                mock_response.status_code = 200
                mock_response.text = "OK"
            return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post
            with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip delays
                deliveries = await dispatcher.dispatch(sample_event, wait=True)

        # Should have retried and eventually succeeded
        assert call_count == 3
        assert deliveries[0].status == WebhookDeliveryStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, dispatcher, manager, sample_event):
        """Test that delivery fails after max retries."""
        manager.register(
            url="https://example.com/webhook",
            config=WebhookConfig(max_retries=2),
        )

        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                deliveries = await dispatcher.dispatch(sample_event, wait=True)

        assert deliveries[0].status == WebhookDeliveryStatus.FAILED


# ============================================================================
# Test Event Tests
# ============================================================================


class TestTestEvent:
    """Tests for test event functionality."""

    @pytest.mark.asyncio
    async def test_send_test_event(self, dispatcher, sample_webhook):
        """Test sending a test event to a webhook."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = "OK"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            delivery = await dispatcher.send_test_event(sample_webhook.id)

        assert delivery is not None
        assert delivery.event_id.startswith("test_")
        assert delivery.status == WebhookDeliveryStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_send_test_event_not_found(self, dispatcher):
        """Test sending test event to non-existent webhook."""
        delivery = await dispatcher.send_test_event("non-existent-id")

        assert delivery is None


# ============================================================================
# Background Tasks Tests
# ============================================================================


class TestBackgroundTasks:
    """Tests for background task handling."""

    @pytest.mark.asyncio
    async def test_dispatch_without_wait(
        self, dispatcher, sample_webhook, sample_event  # noqa: ARG002
    ):
        """Test that dispatch without wait returns immediately (sample_webhook registers webhook)."""
        with patch.object(
            dispatcher, "_deliver_with_retry", new_callable=AsyncMock
        ) as mock_deliver:
            # Make delivery take some time
            async def slow_deliver(*_args):
                await asyncio.sleep(0.1)

            mock_deliver.side_effect = slow_deliver

            import time

            start = time.monotonic()
            deliveries = await dispatcher.dispatch(sample_event, wait=False)
            elapsed = time.monotonic() - start

        # Should return very quickly
        assert elapsed < 0.05
        assert len(deliveries) == 1

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_pending(
        self, dispatcher, sample_webhook, sample_event  # noqa: ARG002
    ):
        """Test that shutdown waits for pending deliveries (sample_webhook registers webhook)."""
        completed = False

        async def slow_deliver(_webhook, _delivery):
            nonlocal completed
            await asyncio.sleep(0.1)
            completed = True

        with patch.object(dispatcher, "_deliver_with_retry", side_effect=slow_deliver):
            await dispatcher.dispatch(sample_event, wait=False)

            # Shutdown should wait for the delivery
            await dispatcher.shutdown()

        assert completed is True


# ============================================================================
# Global Dispatcher Tests
# ============================================================================


class TestGlobalDispatcher:
    """Tests for global dispatcher singleton."""

    def test_get_webhook_dispatcher(self):
        """Test getting global dispatcher."""
        from src.webhooks.dispatcher import set_webhook_dispatcher

        # Reset global
        set_webhook_dispatcher(None)

        dispatcher = get_webhook_dispatcher()

        assert dispatcher is not None
        assert isinstance(dispatcher, WebhookDispatcher)

    def test_get_webhook_dispatcher_singleton(self):
        """Test that global dispatcher is a singleton."""
        from src.webhooks.dispatcher import set_webhook_dispatcher

        # Reset global
        set_webhook_dispatcher(None)

        dispatcher1 = get_webhook_dispatcher()
        dispatcher2 = get_webhook_dispatcher()

        assert dispatcher1 is dispatcher2
