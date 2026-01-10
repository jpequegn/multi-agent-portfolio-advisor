"""Tests for the workflow event emitter."""

import asyncio

import pytest

from src.streaming.emitter import (
    DEFAULT_HEARTBEAT_INTERVAL,
    WorkflowEventEmitter,
    get_emitter_from_state,
)
from src.streaming.events import StreamEventType


class TestWorkflowEventEmitter:
    """Tests for WorkflowEventEmitter."""

    @pytest.fixture
    def emitter(self) -> WorkflowEventEmitter:
        """Create an emitter for tests."""
        return WorkflowEventEmitter(
            workflow_id="wf-123",
            trace_id="trace-456",
        )

    def test_init(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitter initialization."""
        assert emitter.workflow_id == "wf-123"
        assert emitter.trace_id == "trace-456"
        assert emitter.heartbeat_interval == DEFAULT_HEARTBEAT_INTERVAL
        assert emitter.is_closed is False

    def test_custom_heartbeat_interval(self) -> None:
        """Test emitter with custom heartbeat interval."""
        emitter = WorkflowEventEmitter(
            workflow_id="wf-123",
            trace_id="trace-456",
            heartbeat_interval=5.0,
        )
        assert emitter.heartbeat_interval == 5.0

    @pytest.mark.asyncio
    async def test_emit_workflow_started(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting workflow started event."""
        await emitter.emit_workflow_started(
            symbols=["AAPL", "GOOGL"],
            user_request="Analyze portfolio",
        )

        # Get the event from queue
        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.WORKFLOW_STARTED
        assert event.data["symbols"] == ["AAPL", "GOOGL"]
        assert event.data["user_request"] == "Analyze portfolio"

    @pytest.mark.asyncio
    async def test_emit_agent_started(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting agent started event."""
        await emitter.emit_agent_started("research")

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.AGENT_STARTED
        assert event.agent == "research"

    @pytest.mark.asyncio
    async def test_emit_agent_thinking(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting agent thinking event."""
        await emitter.emit_agent_thinking("analysis", "Calculating metrics...")

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.AGENT_THINKING
        assert event.data["thought"] == "Calculating metrics..."

    @pytest.mark.asyncio
    async def test_emit_agent_progress(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting agent progress event."""
        await emitter.emit_agent_progress("research", "Fetching data...", progress=50.0)

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.AGENT_PROGRESS
        assert event.data["message"] == "Fetching data..."
        assert event.data["progress"] == 50.0

    @pytest.mark.asyncio
    async def test_emit_agent_completed(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting agent completed event."""
        await emitter.emit_agent_completed(
            "research",
            output={"count": 5},
            duration_ms=1500.0,
        )

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.AGENT_COMPLETED
        assert event.data["output"]["count"] == 5
        assert event.data["duration_ms"] == 1500.0

    @pytest.mark.asyncio
    async def test_emit_tool_called(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting tool called event."""
        await emitter.emit_tool_called(
            "research",
            "get_price",
            tool_input={"symbol": "AAPL"},
        )

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.TOOL_CALLED
        assert event.data["tool"] == "get_price"
        assert event.data["input"]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_emit_tool_result(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting tool result event."""
        await emitter.emit_tool_result(
            "research",
            "get_price",
            result={"price": 150.0},
            success=True,
        )

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.TOOL_RESULT
        assert event.data["result"]["price"] == 150.0
        assert event.data["success"] is True

    @pytest.mark.asyncio
    async def test_emit_error(self, emitter: WorkflowEventEmitter) -> None:
        """Test emitting error event."""
        await emitter.emit_error(
            "Something went wrong",
            agent="research",
            error_type="RuntimeError",
        )

        event = await asyncio.wait_for(emitter._queue.get(), timeout=1.0)

        assert event.type == StreamEventType.ERROR
        assert event.data["message"] == "Something went wrong"
        assert event.data["agent"] == "research"
        assert event.data["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_emit_workflow_completed_closes_emitter(
        self, emitter: WorkflowEventEmitter
    ) -> None:
        """Test that workflow completed closes the emitter."""
        await emitter.emit_workflow_completed(
            status="completed",
            has_errors=False,
        )

        # Should be closed after completion
        assert emitter.is_closed is True

    @pytest.mark.asyncio
    async def test_close(self, emitter: WorkflowEventEmitter) -> None:
        """Test closing the emitter."""
        await emitter.close()

        assert emitter.is_closed is True

    @pytest.mark.asyncio
    async def test_close_idempotent(self, emitter: WorkflowEventEmitter) -> None:
        """Test that closing multiple times is safe."""
        await emitter.close()
        await emitter.close()  # Should not raise

        assert emitter.is_closed is True

    @pytest.mark.asyncio
    async def test_emit_after_close_is_ignored(
        self, emitter: WorkflowEventEmitter
    ) -> None:
        """Test that emitting after close is ignored."""
        await emitter.close()

        # Should not raise, just be ignored
        await emitter.emit_agent_started("research")

        # Queue should only have the end sentinel
        assert emitter._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_events_iterator(self, emitter: WorkflowEventEmitter) -> None:
        """Test iterating over events."""
        # Emit some events
        await emitter.emit_agent_started("research")
        await emitter.emit_agent_completed("research")
        await emitter.close()

        events = []
        async for event in emitter.events():
            events.append(event)

        assert len(events) == 2
        assert events[0].type == StreamEventType.AGENT_STARTED
        assert events[1].type == StreamEventType.AGENT_COMPLETED

    @pytest.mark.asyncio
    async def test_events_as_sse(self, emitter: WorkflowEventEmitter) -> None:
        """Test iterating over events as SSE strings."""
        await emitter.emit_agent_started("research")
        await emitter.close()

        sse_events = []
        async for sse in emitter.events_as_sse():
            sse_events.append(sse)

        assert len(sse_events) == 1
        assert sse_events[0].startswith("data: ")
        assert sse_events[0].endswith("\n\n")
        assert '"type": "agent_started"' in sse_events[0]

    @pytest.mark.asyncio
    async def test_heartbeat_start_stop(self, emitter: WorkflowEventEmitter) -> None:
        """Test starting and stopping heartbeat."""
        # Use very short interval for testing
        emitter.heartbeat_interval = 0.05

        emitter.start_heartbeat()
        assert emitter._heartbeat_task is not None

        # Wait for at least one heartbeat
        await asyncio.sleep(0.1)

        await emitter.stop_heartbeat()
        assert emitter._heartbeat_task is None

        # Should have received at least one heartbeat
        events = []
        while not emitter._queue.empty():
            events.append(await emitter._queue.get())

        heartbeats = [e for e in events if hasattr(e, "type") and e.type == StreamEventType.HEARTBEAT]
        assert len(heartbeats) >= 1


class TestGetEmitterFromState:
    """Tests for get_emitter_from_state function."""

    def test_get_emitter_present(self) -> None:
        """Test getting emitter when present in state."""
        emitter = WorkflowEventEmitter(
            workflow_id="wf-123",
            trace_id="trace-456",
        )
        state = {"_emitter": emitter, "other": "data"}

        result = get_emitter_from_state(state)

        assert result is emitter

    def test_get_emitter_not_present(self) -> None:
        """Test getting emitter when not present in state."""
        state = {"other": "data"}

        result = get_emitter_from_state(state)

        assert result is None

    def test_get_emitter_empty_state(self) -> None:
        """Test getting emitter from empty state."""
        state: dict = {}

        result = get_emitter_from_state(state)

        assert result is None
