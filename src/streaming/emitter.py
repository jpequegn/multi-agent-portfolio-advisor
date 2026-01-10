"""Event emitter for streaming workflow events.

This module provides the WorkflowEventEmitter class that manages
event queuing and streaming for SSE endpoints.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import structlog

from src.streaming.events import StreamEvent

logger = structlog.get_logger(__name__)

# Sentinel value to signal end of stream
_END_OF_STREAM = object()

# Default heartbeat interval in seconds
DEFAULT_HEARTBEAT_INTERVAL = 15.0


class WorkflowEventEmitter:
    """Async event emitter for streaming workflow events.

    This class provides a queue-based event system that allows
    workflow nodes to emit events that are consumed by SSE endpoints.

    Attributes:
        workflow_id: Unique identifier for the workflow.
        trace_id: Trace ID for observability.
    """

    def __init__(
        self,
        workflow_id: str,
        trace_id: str,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    ) -> None:
        """Initialize the event emitter.

        Args:
            workflow_id: Unique identifier for the workflow.
            trace_id: Trace ID for observability.
            heartbeat_interval: Interval between heartbeat events in seconds.
        """
        self.workflow_id = workflow_id
        self.trace_id = trace_id
        self.heartbeat_interval = heartbeat_interval
        self._queue: asyncio.Queue[StreamEvent | object] = asyncio.Queue()
        self._closed = False
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def emit(self, event: StreamEvent) -> None:
        """Emit an event to the stream.

        Args:
            event: The event to emit.

        Raises:
            RuntimeError: If the emitter has been closed.
        """
        if self._closed:
            logger.warning(
                "emit_after_close",
                workflow_id=self.workflow_id,
                event_type=event.type.value,
            )
            return

        await self._queue.put(event)
        logger.debug(
            "event_emitted",
            workflow_id=self.workflow_id,
            event_type=event.type.value,
            agent=event.agent,
        )

    async def emit_workflow_started(
        self,
        symbols: list[str],
        user_request: str | None = None,
    ) -> None:
        """Emit a workflow started event.

        Args:
            symbols: List of symbols being analyzed.
            user_request: Optional user request string.
        """
        event = StreamEvent.workflow_started(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            symbols=symbols,
            user_request=user_request,
        )
        await self.emit(event)

    async def emit_workflow_completed(
        self,
        status: str,
        has_errors: bool = False,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Emit a workflow completed event and close the stream.

        Args:
            status: Final workflow status.
            has_errors: Whether errors occurred.
            result: Optional final result data.
        """
        event = StreamEvent.workflow_completed(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            status=status,
            has_errors=has_errors,
            result=result,
        )
        await self.emit(event)
        await self.close()

    async def emit_agent_started(self, agent: str) -> None:
        """Emit an agent started event.

        Args:
            agent: Name of the agent starting.
        """
        event = StreamEvent.agent_started(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            agent=agent,
        )
        await self.emit(event)

    async def emit_agent_thinking(self, agent: str, thought: str) -> None:
        """Emit an agent thinking event.

        Args:
            agent: Name of the thinking agent.
            thought: The thought/reasoning being performed.
        """
        event = StreamEvent.agent_thinking(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            agent=agent,
            thought=thought,
        )
        await self.emit(event)

    async def emit_agent_progress(
        self,
        agent: str,
        message: str,
        progress: float | None = None,
    ) -> None:
        """Emit an agent progress event.

        Args:
            agent: Name of the agent.
            message: Progress message.
            progress: Optional progress percentage (0-100).
        """
        event = StreamEvent.agent_progress(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            agent=agent,
            message=message,
            progress=progress,
        )
        await self.emit(event)

    async def emit_agent_completed(
        self,
        agent: str,
        output: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Emit an agent completed event.

        Args:
            agent: Name of the completed agent.
            output: Optional agent output summary.
            duration_ms: Optional execution duration in milliseconds.
        """
        event = StreamEvent.agent_completed(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            agent=agent,
            output=output,
            duration_ms=duration_ms,
        )
        await self.emit(event)

    async def emit_tool_called(
        self,
        agent: str,
        tool: str,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        """Emit a tool called event.

        Args:
            agent: Name of the agent calling the tool.
            tool: Name of the tool being called.
            tool_input: Optional tool input parameters.
        """
        event = StreamEvent.tool_called(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            agent=agent,
            tool=tool,
            tool_input=tool_input,
        )
        await self.emit(event)

    async def emit_tool_result(
        self,
        agent: str,
        tool: str,
        result: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """Emit a tool result event.

        Args:
            agent: Name of the agent that called the tool.
            tool: Name of the tool that returned.
            result: Optional tool result data.
            success: Whether the tool call succeeded.
        """
        event = StreamEvent.tool_result(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            agent=agent,
            tool=tool,
            result=result,
            success=success,
        )
        await self.emit(event)

    async def emit_error(
        self,
        message: str,
        agent: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """Emit an error event.

        Args:
            message: Error message.
            agent: Optional agent where error occurred.
            error_type: Optional error type/category.
        """
        event = StreamEvent.error(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            message=message,
            agent=agent,
            error_type=error_type,
        )
        await self.emit(event)

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat event."""
        event = StreamEvent.heartbeat(
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
        )
        await self._queue.put(event)

    async def _heartbeat_loop(self) -> None:
        """Background task that sends periodic heartbeats."""
        try:
            while not self._closed:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._closed:
                    await self._send_heartbeat()
        except asyncio.CancelledError:
            pass

    def start_heartbeat(self) -> None:
        """Start the heartbeat background task."""
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.debug(
                "heartbeat_started",
                workflow_id=self.workflow_id,
                interval=self.heartbeat_interval,
            )

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat background task."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None
            logger.debug("heartbeat_stopped", workflow_id=self.workflow_id)

    async def close(self) -> None:
        """Close the event stream.

        This signals to consumers that no more events will be emitted.
        """
        if self._closed:
            return

        self._closed = True
        await self.stop_heartbeat()
        await self._queue.put(_END_OF_STREAM)
        logger.debug("emitter_closed", workflow_id=self.workflow_id)

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Iterate over events as they are emitted.

        Yields:
            StreamEvent objects as they are emitted.

        Note:
            This iterator will block until events are available
            and will complete when the emitter is closed.
        """
        while True:
            item = await self._queue.get()
            if item is _END_OF_STREAM:
                break
            if isinstance(item, StreamEvent):
                yield item

    async def events_as_sse(self) -> AsyncIterator[str]:
        """Iterate over events formatted as SSE strings.

        Yields:
            SSE-formatted strings ready to be sent to clients.
        """
        async for event in self.events():
            yield event.to_sse()

    @property
    def is_closed(self) -> bool:
        """Check if the emitter has been closed.

        Returns:
            True if closed, False otherwise.
        """
        return self._closed


def get_emitter_from_state(state: dict[str, Any]) -> WorkflowEventEmitter | None:
    """Extract the event emitter from workflow state.

    Args:
        state: Workflow state dictionary.

    Returns:
        WorkflowEventEmitter if present, None otherwise.
    """
    return state.get("_emitter")
