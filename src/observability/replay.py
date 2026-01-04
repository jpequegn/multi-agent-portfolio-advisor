"""Replay system for debugging and reproducing requests.

This module provides the ability to capture and replay traced requests
for debugging failures, reproducing issues, and testing fixes.

Features:
- Capture request state from Langfuse traces
- Multiple replay modes (full, mock tools, step-by-step)
- Compare original vs replay results
- Debug step-by-step through agent execution

Example:
    engine = ReplayEngine()

    # Capture a request for replay
    request = await engine.capture_request("trace-123")

    # Full replay with live LLM and tools
    result = await engine.replay(request, mode=ReplayMode.FULL)

    # Mock replay using recorded tool responses
    result = await engine.replay(request, mode=ReplayMode.MOCK_TOOLS)

    # Step-by-step debugging
    result = await engine.replay(request, mode=ReplayMode.STEP_BY_STEP)
"""

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

from src.observability.tracing import get_langfuse_client

logger = structlog.get_logger(__name__)


# ============================================================================
# Replay Mode Enumeration
# ============================================================================


class ReplayMode(Enum):
    """Modes for replaying a request."""

    FULL = "full"  # Full replay with live LLM and tools
    MOCK_TOOLS = "mock_tools"  # Replay with mocked tool responses from original
    STEP_BY_STEP = "step_by_step"  # Pause between agents for debugging


# ============================================================================
# Replay Status
# ============================================================================


class ReplayStatus(Enum):
    """Status of a replay execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # For step-by-step mode
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CapturedSpan:
    """A captured span from a trace.

    Represents a single operation (agent, tool, generation) from the original trace.
    """

    span_id: str
    name: str
    span_type: str  # "span", "generation", "tool"
    parent_span_id: str | None
    input_data: dict[str, Any] | None
    output_data: dict[str, Any] | None
    metadata: dict[str, Any]
    start_time: datetime
    end_time: datetime | None
    level: str | None  # SpanLevel value if present
    status: str  # "success", "error"
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "span_type": self.span_type,
            "parent_span_id": self.parent_span_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "level": self.level,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapturedSpan":
        """Create from dictionary."""
        return cls(
            span_id=data["span_id"],
            name=data["name"],
            span_type=data["span_type"],
            parent_span_id=data.get("parent_span_id"),
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            metadata=data.get("metadata", {}),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            level=data.get("level"),
            status=data.get("status", "success"),
            error=data.get("error"),
        )


@dataclass
class CapturedToolResponse:
    """A captured tool response for replay mocking."""

    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    execution_time_ms: float
    was_error: bool
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "execution_time_ms": self.execution_time_ms,
            "was_error": self.was_error,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapturedToolResponse":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            was_error=data.get("was_error", False),
            error_message=data.get("error_message"),
        )


@dataclass
class ReplayableRequest:
    """All inputs needed to replay a request.

    Contains the complete state and captured data from a traced request,
    allowing it to be replayed for debugging.
    """

    request_id: str
    trace_id: str
    workflow_id: str | None

    # Initial state
    input_state: dict[str, Any]

    # Captured execution data
    spans: list[CapturedSpan]
    tool_responses: list[CapturedToolResponse]

    # Metadata
    timestamp: datetime
    user_id: str | None
    session_id: str | None
    original_status: str  # "completed", "failed"
    original_error: str | None

    # Execution summary
    agent_sequence: list[str]  # Order of agents executed
    total_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "workflow_id": self.workflow_id,
            "input_state": self.input_state,
            "spans": [s.to_dict() for s in self.spans],
            "tool_responses": [t.to_dict() for t in self.tool_responses],
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "original_status": self.original_status,
            "original_error": self.original_error,
            "agent_sequence": self.agent_sequence,
            "total_duration_ms": self.total_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayableRequest":
        """Create from dictionary."""
        return cls(
            request_id=data["request_id"],
            trace_id=data["trace_id"],
            workflow_id=data.get("workflow_id"),
            input_state=data.get("input_state", {}),
            spans=[CapturedSpan.from_dict(s) for s in data.get("spans", [])],
            tool_responses=[
                CapturedToolResponse.from_dict(t) for t in data.get("tool_responses", [])
            ],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            original_status=data.get("original_status", "unknown"),
            original_error=data.get("original_error"),
            agent_sequence=data.get("agent_sequence", []),
            total_duration_ms=data.get("total_duration_ms", 0.0),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ReplayableRequest":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ReplayStepResult:
    """Result of a single step in replay execution."""

    step_name: str
    agent_name: str | None
    input_data: dict[str, Any] | None
    output_data: dict[str, Any] | None
    duration_ms: float
    status: str  # "success", "error", "skipped"
    error: str | None
    matches_original: bool | None  # For comparison

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "agent_name": self.agent_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "matches_original": self.matches_original,
        }


@dataclass
class ReplayComparison:
    """Comparison between original and replay execution."""

    field: str
    original_value: Any
    replay_value: Any
    matches: bool
    difference_type: str  # "missing", "added", "changed", "type_mismatch"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "original_value": self.original_value,
            "replay_value": self.replay_value,
            "matches": self.matches,
            "difference_type": self.difference_type,
        }


@dataclass
class ReplayResult:
    """Result of a replay execution.

    Contains the replay output, comparisons with original,
    and debugging information.
    """

    replay_id: str
    request_id: str
    mode: ReplayMode
    status: ReplayStatus

    # Timing
    started_at: datetime
    completed_at: datetime | None
    total_duration_ms: float

    # Results
    final_state: dict[str, Any] | None
    step_results: list[ReplayStepResult]
    error: str | None

    # Comparison with original
    comparisons: list[ReplayComparison]
    overall_match: bool
    match_percentage: float

    # Debugging info
    divergence_point: str | None  # Where replay diverged from original
    divergence_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "replay_id": self.replay_id,
            "request_id": self.request_id,
            "mode": self.mode.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "final_state": self.final_state,
            "step_results": [s.to_dict() for s in self.step_results],
            "error": self.error,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "overall_match": self.overall_match,
            "match_percentage": self.match_percentage,
            "divergence_point": self.divergence_point,
            "divergence_reason": self.divergence_reason,
        }


# ============================================================================
# Trace Retrieval
# ============================================================================


class TraceRetriever:
    """Retrieves and parses traces from Langfuse.

    Handles querying the Langfuse API to fetch trace data
    and converting it into CapturedSpan objects.
    """

    def __init__(self) -> None:
        """Initialize trace retriever."""
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get Langfuse client."""
        if self._client is None:
            self._client = get_langfuse_client()
        return self._client

    async def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Fetch a trace by ID from Langfuse.

        Args:
            trace_id: The trace identifier.

        Returns:
            Trace data or None if not found.
        """
        try:
            client = self._get_client()
            # Langfuse SDK provides synchronous methods
            trace = client.fetch_trace(trace_id)
            if trace:
                result: dict[str, Any] = trace.data if hasattr(trace, "data") else trace
                return result
            return None
        except Exception as e:
            logger.error("trace_fetch_failed", trace_id=trace_id, error=str(e))
            return None

    async def get_trace_observations(self, trace_id: str) -> list[dict[str, Any]]:
        """Fetch all observations (spans) for a trace.

        Args:
            trace_id: The trace identifier.

        Returns:
            List of observation data.
        """
        try:
            client = self._get_client()
            # Fetch observations for the trace
            observations = client.fetch_observations(trace_id=trace_id)
            if observations:
                data = observations.data if hasattr(observations, "data") else observations
                return list(data) if data else []
            return []
        except Exception as e:
            logger.error("observations_fetch_failed", trace_id=trace_id, error=str(e))
            return []

    def parse_observation(self, obs: dict[str, Any]) -> CapturedSpan:
        """Parse a Langfuse observation into a CapturedSpan.

        Args:
            obs: Raw observation data from Langfuse.

        Returns:
            Parsed CapturedSpan.
        """
        # Extract timestamps
        start_time = obs.get("startTime") or obs.get("start_time")
        end_time = obs.get("endTime") or obs.get("end_time")

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        elif start_time is None:
            start_time = datetime.now(UTC)

        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # Extract metadata
        metadata = obs.get("metadata", {}) or {}
        level = metadata.get("level")

        # Determine status
        status = "success"
        error = None
        if obs.get("level") == "ERROR" or obs.get("statusMessage"):
            status = "error"
            error = obs.get("statusMessage")

        return CapturedSpan(
            span_id=obs.get("id", str(uuid.uuid4())),
            name=obs.get("name", "unknown"),
            span_type=obs.get("type", "span"),
            parent_span_id=obs.get("parentObservationId"),
            input_data=obs.get("input"),
            output_data=obs.get("output"),
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
            level=level,
            status=status,
            error=error,
        )

    def extract_tool_responses(self, spans: list[CapturedSpan]) -> list[CapturedToolResponse]:
        """Extract tool responses from captured spans.

        Args:
            spans: List of captured spans.

        Returns:
            List of captured tool responses.
        """
        tool_responses = []

        for span in spans:
            # Look for tool-type spans or spans with tool metadata
            is_tool = (
                span.span_type == "tool"
                or span.level == "tool"
                or span.metadata.get("level") == "tool"
            )

            if is_tool and span.output_data is not None:
                duration_ms = 0.0
                if span.start_time and span.end_time:
                    duration_ms = (span.end_time - span.start_time).total_seconds() * 1000

                tool_responses.append(
                    CapturedToolResponse(
                        tool_name=span.name,
                        input_data=span.input_data or {},
                        output_data=span.output_data,
                        execution_time_ms=duration_ms,
                        was_error=span.status == "error",
                        error_message=span.error,
                    )
                )

        return tool_responses

    def extract_agent_sequence(self, spans: list[CapturedSpan]) -> list[str]:
        """Extract the sequence of agents executed.

        Args:
            spans: List of captured spans.

        Returns:
            Ordered list of agent names.
        """
        agents = []

        # Sort by start time
        sorted_spans = sorted(spans, key=lambda s: s.start_time)

        for span in sorted_spans:
            # Look for agent-level spans
            is_agent = (
                span.level == "agent"
                or span.metadata.get("level") == "agent"
                or span.metadata.get("agent_name")
            )

            if is_agent:
                agent_name = span.metadata.get("agent_name") or span.name
                if agent_name not in agents:
                    agents.append(agent_name)

        return agents


# ============================================================================
# Tool Mock Manager
# ============================================================================


class ToolMockManager:
    """Manages mocked tool responses for replay.

    Stores captured tool responses and returns them when
    the corresponding tool is called during replay.
    """

    def __init__(self) -> None:
        """Initialize mock manager."""
        self._responses: dict[str, list[CapturedToolResponse]] = {}
        self._call_counts: dict[str, int] = {}

    def load_responses(self, responses: list[CapturedToolResponse]) -> None:
        """Load captured responses for mocking.

        Args:
            responses: List of captured tool responses.
        """
        self._responses.clear()
        self._call_counts.clear()

        for response in responses:
            if response.tool_name not in self._responses:
                self._responses[response.tool_name] = []
            self._responses[response.tool_name].append(response)

        logger.debug(
            "mock_responses_loaded",
            tool_count=len(self._responses),
            total_responses=len(responses),
        )

    def get_mock_response(
        self,
        tool_name: str,
        input_data: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> CapturedToolResponse | None:
        """Get a mock response for a tool call.

        Args:
            tool_name: Name of the tool.
            input_data: Input data for the tool (for future matching support).

        Returns:
            Matching captured response or None.
        """
        if tool_name not in self._responses:
            logger.warning("no_mock_for_tool", tool_name=tool_name)
            return None

        # Get call index for this tool
        call_idx = self._call_counts.get(tool_name, 0)
        responses = self._responses[tool_name]

        if call_idx >= len(responses):
            logger.warning(
                "mock_exhausted",
                tool_name=tool_name,
                call_idx=call_idx,
                available=len(responses),
            )
            return None

        response = responses[call_idx]
        self._call_counts[tool_name] = call_idx + 1

        logger.debug(
            "mock_response_returned",
            tool_name=tool_name,
            call_idx=call_idx,
        )

        return response

    def reset(self) -> None:
        """Reset call counts for a new replay."""
        self._call_counts.clear()


# ============================================================================
# Comparison Engine
# ============================================================================


class ComparisonEngine:
    """Compares original and replay results."""

    def compare_states(
        self,
        original: dict[str, Any],
        replay: dict[str, Any],
    ) -> list[ReplayComparison]:
        """Compare two state dictionaries.

        Args:
            original: Original state.
            replay: Replay state.

        Returns:
            List of comparisons.
        """
        comparisons = []
        all_keys = set(original.keys()) | set(replay.keys())

        for key in all_keys:
            orig_val = original.get(key)
            replay_val = replay.get(key)

            if key not in original:
                comparisons.append(
                    ReplayComparison(
                        field=key,
                        original_value=None,
                        replay_value=replay_val,
                        matches=False,
                        difference_type="added",
                    )
                )
            elif key not in replay:
                comparisons.append(
                    ReplayComparison(
                        field=key,
                        original_value=orig_val,
                        replay_value=None,
                        matches=False,
                        difference_type="missing",
                    )
                )
            elif type(orig_val) is not type(replay_val):
                comparisons.append(
                    ReplayComparison(
                        field=key,
                        original_value=orig_val,
                        replay_value=replay_val,
                        matches=False,
                        difference_type="type_mismatch",
                    )
                )
            elif orig_val != replay_val:
                comparisons.append(
                    ReplayComparison(
                        field=key,
                        original_value=orig_val,
                        replay_value=replay_val,
                        matches=False,
                        difference_type="changed",
                    )
                )
            else:
                comparisons.append(
                    ReplayComparison(
                        field=key,
                        original_value=orig_val,
                        replay_value=replay_val,
                        matches=True,
                        difference_type="",
                    )
                )

        return comparisons

    def calculate_match_percentage(self, comparisons: list[ReplayComparison]) -> float:
        """Calculate the percentage of matching fields.

        Args:
            comparisons: List of comparisons.

        Returns:
            Match percentage (0-100).
        """
        if not comparisons:
            return 100.0

        matching = sum(1 for c in comparisons if c.matches)
        return (matching / len(comparisons)) * 100

    def find_divergence_point(
        self,
        original_spans: list[CapturedSpan],
        replay_steps: list[ReplayStepResult],
    ) -> tuple[str | None, str | None]:
        """Find where replay diverged from original.

        Args:
            original_spans: Original execution spans.
            replay_steps: Replay step results.

        Returns:
            Tuple of (divergence_point, divergence_reason).
        """
        # Sort original spans by time
        sorted_original = sorted(original_spans, key=lambda s: s.start_time)

        for step in replay_steps:
            if step.status == "error":
                return (step.step_name, f"Error during replay: {step.error}")

            if step.matches_original is False:
                # Find corresponding original span
                original_output = None
                for span in sorted_original:
                    if span.name == step.step_name:
                        original_output = span.output_data
                        break

                return (
                    step.step_name,
                    f"Output mismatch at {step.step_name}: "
                    f"original had {type(original_output).__name__}, "
                    f"replay had {type(step.output_data).__name__}",
                )

        return (None, None)


# ============================================================================
# Replay Engine
# ============================================================================


class ReplayEngine:
    """Engine for capturing and replaying requests.

    Provides the main interface for:
    - Capturing replayable state from traces
    - Executing replays in different modes
    - Comparing original vs replay results
    """

    def __init__(self) -> None:
        """Initialize replay engine."""
        self._retriever = TraceRetriever()
        self._mock_manager = ToolMockManager()
        self._comparison = ComparisonEngine()
        self._step_callback: Callable[[ReplayStepResult], None] | None = None
        self._paused = False

    def set_step_callback(
        self,
        callback: Callable[[ReplayStepResult], None] | None,
    ) -> None:
        """Set callback for step-by-step mode.

        Args:
            callback: Function called after each step.
        """
        self._step_callback = callback

    async def capture_request(self, trace_id: str) -> ReplayableRequest:
        """Capture all inputs needed to replay a request.

        Args:
            trace_id: The trace ID to capture.

        Returns:
            ReplayableRequest with all captured data.

        Raises:
            ValueError: If trace not found or invalid.
        """
        logger.info("capturing_request", trace_id=trace_id)

        # Fetch trace data
        trace = await self._retriever.get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")

        # Fetch all observations
        observations = await self._retriever.get_trace_observations(trace_id)

        # Parse observations into spans
        spans = [self._retriever.parse_observation(obs) for obs in observations]

        # Extract tool responses
        tool_responses = self._retriever.extract_tool_responses(spans)

        # Extract agent sequence
        agent_sequence = self._retriever.extract_agent_sequence(spans)

        # Calculate total duration
        total_duration_ms = 0.0
        if spans:
            start = min(s.start_time for s in spans)
            end = max(s.end_time or s.start_time for s in spans)
            total_duration_ms = (end - start).total_seconds() * 1000

        # Extract input state from trace or first span
        input_state = trace.get("input", {})
        if not input_state and spans:
            # Try to find request-level span
            for span in spans:
                if span.level == "request" or span.name.endswith("_request"):
                    input_state = span.input_data or {}
                    break

        # Determine original status
        original_status = "completed"
        original_error = None
        trace_status = trace.get("status") or trace.get("output", {}).get("status")
        if trace_status == "failed" or any(s.status == "error" for s in spans):
            original_status = "failed"
            # Find error message
            for span in spans:
                if span.error:
                    original_error = span.error
                    break

        request = ReplayableRequest(
            request_id=str(uuid.uuid4()),
            trace_id=trace_id,
            workflow_id=trace.get("metadata", {}).get("workflow_id"),
            input_state=input_state,
            spans=spans,
            tool_responses=tool_responses,
            timestamp=datetime.now(UTC),
            user_id=trace.get("userId"),
            session_id=trace.get("sessionId"),
            original_status=original_status,
            original_error=original_error,
            agent_sequence=agent_sequence,
            total_duration_ms=total_duration_ms,
        )

        logger.info(
            "request_captured",
            request_id=request.request_id,
            trace_id=trace_id,
            span_count=len(spans),
            tool_response_count=len(tool_responses),
            agent_count=len(agent_sequence),
        )

        return request

    async def replay(
        self,
        request: ReplayableRequest,
        mode: ReplayMode = ReplayMode.FULL,
    ) -> ReplayResult:
        """Replay a captured request.

        Args:
            request: The captured request to replay.
            mode: Replay mode to use.

        Returns:
            ReplayResult with execution results and comparisons.
        """
        replay_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)
        self._paused = False

        logger.info(
            "replay_started",
            replay_id=replay_id,
            request_id=request.request_id,
            mode=mode.value,
        )

        step_results: list[ReplayStepResult] = []
        final_state: dict[str, Any] | None = None
        error: str | None = None
        status = ReplayStatus.RUNNING

        try:
            if mode == ReplayMode.FULL:
                final_state, step_results = await self._full_replay(request)
            elif mode == ReplayMode.MOCK_TOOLS:
                final_state, step_results = await self._mock_replay(request)
            elif mode == ReplayMode.STEP_BY_STEP:
                final_state, step_results = await self._step_replay(request)

            status = ReplayStatus.COMPLETED

        except Exception as e:
            logger.error("replay_failed", replay_id=replay_id, error=str(e))
            error = str(e)
            status = ReplayStatus.FAILED

        completed_at = datetime.now(UTC)
        total_duration_ms = (completed_at - started_at).total_seconds() * 1000

        # Compare with original
        comparisons: list[ReplayComparison] = []
        overall_match = True
        match_percentage = 100.0

        if final_state and request.spans:
            # Find original final state from last span output
            original_final = {}
            for span in reversed(request.spans):
                if span.output_data:
                    original_final = span.output_data
                    break

            comparisons = self._comparison.compare_states(original_final, final_state)
            match_percentage = self._comparison.calculate_match_percentage(comparisons)
            overall_match = match_percentage >= 95.0  # Allow small differences

        # Find divergence point if not matching
        divergence_point = None
        divergence_reason = None
        if not overall_match:
            divergence_point, divergence_reason = self._comparison.find_divergence_point(
                request.spans,
                step_results,
            )

        result = ReplayResult(
            replay_id=replay_id,
            request_id=request.request_id,
            mode=mode,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_ms=total_duration_ms,
            final_state=final_state,
            step_results=step_results,
            error=error,
            comparisons=comparisons,
            overall_match=overall_match,
            match_percentage=match_percentage,
            divergence_point=divergence_point,
            divergence_reason=divergence_reason,
        )

        logger.info(
            "replay_completed",
            replay_id=replay_id,
            status=status.value,
            duration_ms=total_duration_ms,
            match_percentage=match_percentage,
        )

        return result

    async def _full_replay(
        self,
        request: ReplayableRequest,
    ) -> tuple[dict[str, Any] | None, list[ReplayStepResult]]:
        """Execute full replay with live LLM and tools.

        Args:
            request: Request to replay.

        Returns:
            Tuple of (final_state, step_results).
        """
        # Import workflow execution
        from src.orchestration.state import create_initial_state
        from src.orchestration.workflow import create_workflow

        step_results: list[ReplayStepResult] = []

        # Recreate initial state from captured input
        portfolio = request.input_state.get("portfolio", {})
        user_request = request.input_state.get("user_request", "")
        user_id = request.user_id

        initial_state = create_initial_state(
            portfolio=portfolio,
            user_request=user_request,
            user_id=user_id,
            trace_id=f"replay-{request.request_id}",
        )

        # Create and run workflow
        workflow = create_workflow()

        step_start = datetime.now(UTC)
        try:
            result = await workflow.ainvoke(initial_state)
            step_end = datetime.now(UTC)

            step_results.append(
                ReplayStepResult(
                    step_name="full_workflow",
                    agent_name=None,
                    input_data=dict(initial_state),
                    output_data=result,
                    duration_ms=(step_end - step_start).total_seconds() * 1000,
                    status="success",
                    error=None,
                    matches_original=None,  # Will be determined in comparison
                )
            )

            return result, step_results

        except Exception as e:
            step_end = datetime.now(UTC)
            step_results.append(
                ReplayStepResult(
                    step_name="full_workflow",
                    agent_name=None,
                    input_data=dict(initial_state),
                    output_data=None,
                    duration_ms=(step_end - step_start).total_seconds() * 1000,
                    status="error",
                    error=str(e),
                    matches_original=request.original_status == "failed",
                )
            )
            raise

    async def _mock_replay(
        self,
        request: ReplayableRequest,
    ) -> tuple[dict[str, Any] | None, list[ReplayStepResult]]:
        """Execute replay with mocked tool responses.

        Args:
            request: Request to replay.

        Returns:
            Tuple of (final_state, step_results).
        """
        from src.orchestration.state import create_initial_state
        from src.orchestration.workflow import create_workflow
        from src.tools.base import default_registry

        step_results: list[ReplayStepResult] = []

        # Load mock responses
        self._mock_manager.load_responses(request.tool_responses)

        # Enable mock mode on tool registry
        default_registry.set_mock_mode(True)

        try:
            # Recreate initial state
            portfolio = request.input_state.get("portfolio", {})
            user_request = request.input_state.get("user_request", "")

            initial_state = create_initial_state(
                portfolio=portfolio,
                user_request=user_request,
                user_id=request.user_id,
                trace_id=f"replay-mock-{request.request_id}",
            )

            # Create and run workflow
            workflow = create_workflow()

            step_start = datetime.now(UTC)
            result = await workflow.ainvoke(initial_state)
            step_end = datetime.now(UTC)

            step_results.append(
                ReplayStepResult(
                    step_name="mock_workflow",
                    agent_name=None,
                    input_data=dict(initial_state),
                    output_data=result,
                    duration_ms=(step_end - step_start).total_seconds() * 1000,
                    status="success",
                    error=None,
                    matches_original=None,
                )
            )

            return result, step_results

        finally:
            # Disable mock mode
            default_registry.set_mock_mode(False)
            self._mock_manager.reset()

    async def _step_replay(
        self,
        request: ReplayableRequest,
    ) -> tuple[dict[str, Any] | None, list[ReplayStepResult]]:
        """Execute step-by-step replay with pauses.

        Args:
            request: Request to replay.

        Returns:
            Tuple of (final_state, step_results).
        """
        from src.orchestration.state import (
            AgentName,
            create_initial_state,
            update_state_for_agent,
            update_state_with_result,
        )

        step_results: list[ReplayStepResult] = []

        # Recreate initial state
        portfolio = request.input_state.get("portfolio", {})
        user_request = request.input_state.get("user_request", "")

        state = create_initial_state(
            portfolio=portfolio,
            user_request=user_request,
            user_id=request.user_id,
            trace_id=f"replay-step-{request.request_id}",
        )

        # Execute each agent in sequence
        agent_map = {
            "research": AgentName.RESEARCH,
            "research_agent": AgentName.RESEARCH,
            "analysis": AgentName.ANALYSIS,
            "analysis_agent": AgentName.ANALYSIS,
            "recommendation": AgentName.RECOMMENDATION,
            "recommendation_agent": AgentName.RECOMMENDATION,
        }

        for agent_name in request.agent_sequence:
            if self._paused:
                break

            agent_enum = agent_map.get(agent_name.lower())
            if not agent_enum:
                continue

            step_start = datetime.now(UTC)

            try:
                # Update state for agent
                state = update_state_for_agent(state, agent_enum)

                # Find original span for this agent
                original_span = None
                for span in request.spans:
                    if span.metadata.get("agent_name") == agent_name or span.name == agent_name:
                        original_span = span
                        break

                # Execute agent (in step mode, we use original outputs)
                if original_span and original_span.output_data:
                    state = update_state_with_result(
                        state,
                        agent_enum,
                        original_span.output_data,
                    )

                step_end = datetime.now(UTC)

                step_result = ReplayStepResult(
                    step_name=agent_name,
                    agent_name=agent_name,
                    input_data=original_span.input_data if original_span else None,
                    output_data=original_span.output_data if original_span else None,
                    duration_ms=(step_end - step_start).total_seconds() * 1000,
                    status="success",
                    error=None,
                    matches_original=True,  # Using original outputs
                )

                step_results.append(step_result)

                # Call callback and wait if step-by-step
                if self._step_callback:
                    self._step_callback(step_result)

                # Small delay to allow inspection
                await asyncio.sleep(0.1)

            except Exception as e:
                step_end = datetime.now(UTC)
                step_results.append(
                    ReplayStepResult(
                        step_name=agent_name,
                        agent_name=agent_name,
                        input_data=None,
                        output_data=None,
                        duration_ms=(step_end - step_start).total_seconds() * 1000,
                        status="error",
                        error=str(e),
                        matches_original=False,
                    )
                )
                raise

        return dict(state), step_results

    def pause(self) -> None:
        """Pause step-by-step replay."""
        self._paused = True
        logger.info("replay_paused")

    def resume(self) -> None:
        """Resume step-by-step replay."""
        self._paused = False
        logger.info("replay_resumed")


# ============================================================================
# Convenience Functions
# ============================================================================


async def capture_and_replay(
    trace_id: str,
    mode: ReplayMode = ReplayMode.MOCK_TOOLS,
) -> ReplayResult:
    """Capture and replay a trace in one call.

    Args:
        trace_id: Trace ID to replay.
        mode: Replay mode to use.

    Returns:
        ReplayResult.
    """
    engine = ReplayEngine()
    request = await engine.capture_request(trace_id)
    return await engine.replay(request, mode)


def create_replay_report(result: ReplayResult) -> str:
    """Create a human-readable replay report.

    Args:
        result: ReplayResult to report on.

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        f"REPLAY REPORT - {result.replay_id}",
        "=" * 60,
        "",
        f"Request ID: {result.request_id}",
        f"Mode: {result.mode.value}",
        f"Status: {result.status.value}",
        f"Duration: {result.total_duration_ms:.2f}ms",
        "",
        "-" * 40,
        "COMPARISON SUMMARY",
        "-" * 40,
        f"Overall Match: {'YES' if result.overall_match else 'NO'}",
        f"Match Percentage: {result.match_percentage:.1f}%",
    ]

    if result.divergence_point:
        lines.extend(
            [
                "",
                f"Divergence Point: {result.divergence_point}",
                f"Reason: {result.divergence_reason}",
            ]
        )

    if result.error:
        lines.extend(
            [
                "",
                "-" * 40,
                "ERROR",
                "-" * 40,
                result.error,
            ]
        )

    lines.extend(
        [
            "",
            "-" * 40,
            "STEP RESULTS",
            "-" * 40,
        ]
    )

    for step in result.step_results:
        status_icon = "✓" if step.status == "success" else "✗"
        lines.append(f"  {status_icon} {step.step_name} ({step.duration_ms:.2f}ms)")
        if step.error:
            lines.append(f"      Error: {step.error}")

    if result.comparisons:
        lines.extend(
            [
                "",
                "-" * 40,
                "FIELD COMPARISONS",
                "-" * 40,
            ]
        )
        mismatches = [c for c in result.comparisons if not c.matches]
        if mismatches:
            for c in mismatches[:10]:  # Show first 10 mismatches
                lines.append(f"  • {c.field}: {c.difference_type}")
            if len(mismatches) > 10:
                lines.append(f"  ... and {len(mismatches) - 10} more")
        else:
            lines.append("  All fields match!")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
