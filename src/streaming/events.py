"""Stream event types and models for SSE streaming.

This module defines the event types and data structures used for
streaming workflow progress to clients via Server-Sent Events.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class StreamEventType(str, Enum):
    """Types of events that can be streamed to clients."""

    # Workflow lifecycle events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"

    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_THINKING = "agent_thinking"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"

    # Tool events
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"

    # Error and control events
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """A single event to be streamed to the client.

    Attributes:
        type: The type of event.
        workflow_id: Unique identifier for the workflow.
        trace_id: Trace ID for observability.
        data: Event-specific payload data.
        timestamp: When the event occurred (ISO format).
        agent: Optional agent name for agent-specific events.
    """

    type: StreamEventType
    workflow_id: str
    trace_id: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    agent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the event.
        """
        result = {
            "type": self.type.value,
            "workflow_id": self.workflow_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }
        if self.agent:
            result["agent"] = self.agent
        return result

    def to_json(self) -> str:
        """Convert event to JSON string.

        Returns:
            JSON string representation of the event.
        """
        return json.dumps(self.to_dict())

    def to_sse(self) -> str:
        """Format event as Server-Sent Event.

        Returns:
            SSE-formatted string with data field.
        """
        return f"data: {self.to_json()}\n\n"

    @classmethod
    def workflow_started(
        cls,
        workflow_id: str,
        trace_id: str,
        symbols: list[str],
        user_request: str | None = None,
    ) -> "StreamEvent":
        """Create a workflow started event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            symbols: List of symbols being analyzed.
            user_request: Optional user request string.

        Returns:
            StreamEvent for workflow start.
        """
        return cls(
            type=StreamEventType.WORKFLOW_STARTED,
            workflow_id=workflow_id,
            trace_id=trace_id,
            data={
                "symbols": symbols,
                "symbol_count": len(symbols),
                "user_request": user_request,
            },
        )

    @classmethod
    def workflow_completed(
        cls,
        workflow_id: str,
        trace_id: str,
        status: str,
        has_errors: bool = False,
        result: dict[str, Any] | None = None,
    ) -> "StreamEvent":
        """Create a workflow completed event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            status: Final workflow status.
            has_errors: Whether errors occurred.
            result: Optional final result data.

        Returns:
            StreamEvent for workflow completion.
        """
        return cls(
            type=StreamEventType.WORKFLOW_COMPLETED,
            workflow_id=workflow_id,
            trace_id=trace_id,
            data={
                "status": status,
                "has_errors": has_errors,
                "result": result or {},
            },
        )

    @classmethod
    def agent_started(
        cls,
        workflow_id: str,
        trace_id: str,
        agent: str,
    ) -> "StreamEvent":
        """Create an agent started event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            agent: Name of the agent starting.

        Returns:
            StreamEvent for agent start.
        """
        return cls(
            type=StreamEventType.AGENT_STARTED,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data={"agent": agent},
        )

    @classmethod
    def agent_thinking(
        cls,
        workflow_id: str,
        trace_id: str,
        agent: str,
        thought: str,
    ) -> "StreamEvent":
        """Create an agent thinking event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            agent: Name of the thinking agent.
            thought: The thought/reasoning being performed.

        Returns:
            StreamEvent for agent thinking.
        """
        return cls(
            type=StreamEventType.AGENT_THINKING,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data={"agent": agent, "thought": thought},
        )

    @classmethod
    def agent_progress(
        cls,
        workflow_id: str,
        trace_id: str,
        agent: str,
        message: str,
        progress: float | None = None,
    ) -> "StreamEvent":
        """Create an agent progress event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            agent: Name of the agent.
            message: Progress message.
            progress: Optional progress percentage (0-100).

        Returns:
            StreamEvent for agent progress.
        """
        data: dict[str, Any] = {"agent": agent, "message": message}
        if progress is not None:
            data["progress"] = progress
        return cls(
            type=StreamEventType.AGENT_PROGRESS,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data=data,
        )

    @classmethod
    def agent_completed(
        cls,
        workflow_id: str,
        trace_id: str,
        agent: str,
        output: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> "StreamEvent":
        """Create an agent completed event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            agent: Name of the completed agent.
            output: Optional agent output summary.
            duration_ms: Optional execution duration in milliseconds.

        Returns:
            StreamEvent for agent completion.
        """
        data: dict[str, Any] = {"agent": agent}
        if output is not None:
            data["output"] = output
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        return cls(
            type=StreamEventType.AGENT_COMPLETED,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data=data,
        )

    @classmethod
    def tool_called(
        cls,
        workflow_id: str,
        trace_id: str,
        agent: str,
        tool: str,
        tool_input: dict[str, Any] | None = None,
    ) -> "StreamEvent":
        """Create a tool called event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            agent: Name of the agent calling the tool.
            tool: Name of the tool being called.
            tool_input: Optional tool input parameters.

        Returns:
            StreamEvent for tool call.
        """
        return cls(
            type=StreamEventType.TOOL_CALLED,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data={
                "agent": agent,
                "tool": tool,
                "input": tool_input or {},
            },
        )

    @classmethod
    def tool_result(
        cls,
        workflow_id: str,
        trace_id: str,
        agent: str,
        tool: str,
        result: dict[str, Any] | None = None,
        success: bool = True,
    ) -> "StreamEvent":
        """Create a tool result event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            agent: Name of the agent that called the tool.
            tool: Name of the tool that returned.
            result: Optional tool result data.
            success: Whether the tool call succeeded.

        Returns:
            StreamEvent for tool result.
        """
        return cls(
            type=StreamEventType.TOOL_RESULT,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data={
                "agent": agent,
                "tool": tool,
                "result": result or {},
                "success": success,
            },
        )

    @classmethod
    def error(
        cls,
        workflow_id: str,
        trace_id: str,
        message: str,
        agent: str | None = None,
        error_type: str | None = None,
    ) -> "StreamEvent":
        """Create an error event.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.
            message: Error message.
            agent: Optional agent where error occurred.
            error_type: Optional error type/category.

        Returns:
            StreamEvent for error.
        """
        data: dict[str, Any] = {"message": message}
        if agent:
            data["agent"] = agent
        if error_type:
            data["error_type"] = error_type
        return cls(
            type=StreamEventType.ERROR,
            workflow_id=workflow_id,
            trace_id=trace_id,
            agent=agent,
            data=data,
        )

    @classmethod
    def heartbeat(
        cls,
        workflow_id: str,
        trace_id: str,
    ) -> "StreamEvent":
        """Create a heartbeat event for keep-alive.

        Args:
            workflow_id: Unique workflow identifier.
            trace_id: Trace ID for observability.

        Returns:
            StreamEvent for heartbeat.
        """
        return cls(
            type=StreamEventType.HEARTBEAT,
            workflow_id=workflow_id,
            trace_id=trace_id,
            data={},
        )
