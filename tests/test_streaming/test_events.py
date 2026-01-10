"""Tests for stream events."""

import json
from datetime import datetime

import pytest

from src.streaming.events import StreamEvent, StreamEventType


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_workflow_started_value(self) -> None:
        """Test WORKFLOW_STARTED type has correct value."""
        assert StreamEventType.WORKFLOW_STARTED.value == "workflow_started"

    def test_agent_started_value(self) -> None:
        """Test AGENT_STARTED type has correct value."""
        assert StreamEventType.AGENT_STARTED.value == "agent_started"

    def test_agent_completed_value(self) -> None:
        """Test AGENT_COMPLETED type has correct value."""
        assert StreamEventType.AGENT_COMPLETED.value == "agent_completed"

    def test_heartbeat_value(self) -> None:
        """Test HEARTBEAT type has correct value."""
        assert StreamEventType.HEARTBEAT.value == "heartbeat"

    def test_error_value(self) -> None:
        """Test ERROR type has correct value."""
        assert StreamEventType.ERROR.value == "error"


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_create_basic_event(self) -> None:
        """Test creating a basic stream event."""
        event = StreamEvent(
            type=StreamEventType.WORKFLOW_STARTED,
            workflow_id="wf-123",
            trace_id="trace-456",
        )

        assert event.type == StreamEventType.WORKFLOW_STARTED
        assert event.workflow_id == "wf-123"
        assert event.trace_id == "trace-456"
        assert event.data == {}
        assert event.agent is None

    def test_create_event_with_data(self) -> None:
        """Test creating an event with data payload."""
        event = StreamEvent(
            type=StreamEventType.AGENT_COMPLETED,
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            data={"symbols": ["AAPL", "GOOGL"]},
        )

        assert event.agent == "research"
        assert event.data == {"symbols": ["AAPL", "GOOGL"]}

    def test_timestamp_auto_generated(self) -> None:
        """Test that timestamp is auto-generated."""
        before = datetime.now().isoformat()[:10]  # Just date part
        event = StreamEvent(
            type=StreamEventType.HEARTBEAT,
            workflow_id="wf-123",
            trace_id="trace-456",
        )
        after = datetime.now().isoformat()[:10]

        # Timestamp should be a valid ISO format
        assert before <= event.timestamp[:10] <= after

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        event = StreamEvent(
            type=StreamEventType.AGENT_STARTED,
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="analysis",
            data={"agent": "analysis"},
        )

        data = event.to_dict()

        assert data["type"] == "agent_started"
        assert data["workflow_id"] == "wf-123"
        assert data["trace_id"] == "trace-456"
        assert data["agent"] == "analysis"
        assert "timestamp" in data
        assert data["data"] == {"agent": "analysis"}

    def test_to_dict_without_agent(self) -> None:
        """Test serialization excludes agent when None."""
        event = StreamEvent(
            type=StreamEventType.HEARTBEAT,
            workflow_id="wf-123",
            trace_id="trace-456",
        )

        data = event.to_dict()

        assert "agent" not in data

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        event = StreamEvent(
            type=StreamEventType.WORKFLOW_COMPLETED,
            workflow_id="wf-123",
            trace_id="trace-456",
            data={"status": "completed"},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "workflow_completed"
        assert parsed["data"]["status"] == "completed"

    def test_to_sse(self) -> None:
        """Test SSE formatting."""
        event = StreamEvent(
            type=StreamEventType.HEARTBEAT,
            workflow_id="wf-123",
            trace_id="trace-456",
        )

        sse = event.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        # Parse the JSON part
        json_part = sse[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        parsed = json.loads(json_part)
        assert parsed["type"] == "heartbeat"


class TestStreamEventFactoryMethods:
    """Tests for StreamEvent factory methods."""

    def test_workflow_started(self) -> None:
        """Test workflow_started factory."""
        event = StreamEvent.workflow_started(
            workflow_id="wf-123",
            trace_id="trace-456",
            symbols=["AAPL", "GOOGL"],
            user_request="Analyze my portfolio",
        )

        assert event.type == StreamEventType.WORKFLOW_STARTED
        assert event.data["symbols"] == ["AAPL", "GOOGL"]
        assert event.data["symbol_count"] == 2
        assert event.data["user_request"] == "Analyze my portfolio"

    def test_workflow_completed(self) -> None:
        """Test workflow_completed factory."""
        event = StreamEvent.workflow_completed(
            workflow_id="wf-123",
            trace_id="trace-456",
            status="completed",
            has_errors=False,
            result={"latency_ms": 1500},
        )

        assert event.type == StreamEventType.WORKFLOW_COMPLETED
        assert event.data["status"] == "completed"
        assert event.data["has_errors"] is False
        assert event.data["result"]["latency_ms"] == 1500

    def test_agent_started(self) -> None:
        """Test agent_started factory."""
        event = StreamEvent.agent_started(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
        )

        assert event.type == StreamEventType.AGENT_STARTED
        assert event.agent == "research"
        assert event.data["agent"] == "research"

    def test_agent_thinking(self) -> None:
        """Test agent_thinking factory."""
        event = StreamEvent.agent_thinking(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="analysis",
            thought="Calculating risk metrics...",
        )

        assert event.type == StreamEventType.AGENT_THINKING
        assert event.agent == "analysis"
        assert event.data["thought"] == "Calculating risk metrics..."

    def test_agent_progress(self) -> None:
        """Test agent_progress factory."""
        event = StreamEvent.agent_progress(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            message="Fetching AAPL data...",
            progress=50.0,
        )

        assert event.type == StreamEventType.AGENT_PROGRESS
        assert event.data["message"] == "Fetching AAPL data..."
        assert event.data["progress"] == 50.0

    def test_agent_progress_without_progress(self) -> None:
        """Test agent_progress without progress percentage."""
        event = StreamEvent.agent_progress(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            message="Working...",
        )

        assert "progress" not in event.data

    def test_agent_completed(self) -> None:
        """Test agent_completed factory."""
        event = StreamEvent.agent_completed(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            output={"symbols_researched": ["AAPL"]},
            duration_ms=1234.5,
        )

        assert event.type == StreamEventType.AGENT_COMPLETED
        assert event.agent == "research"
        assert event.data["output"]["symbols_researched"] == ["AAPL"]
        assert event.data["duration_ms"] == 1234.5

    def test_tool_called(self) -> None:
        """Test tool_called factory."""
        event = StreamEvent.tool_called(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            tool="get_stock_price",
            tool_input={"symbol": "AAPL"},
        )

        assert event.type == StreamEventType.TOOL_CALLED
        assert event.data["tool"] == "get_stock_price"
        assert event.data["input"]["symbol"] == "AAPL"

    def test_tool_result(self) -> None:
        """Test tool_result factory."""
        event = StreamEvent.tool_result(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            tool="get_stock_price",
            result={"price": 150.0},
            success=True,
        )

        assert event.type == StreamEventType.TOOL_RESULT
        assert event.data["tool"] == "get_stock_price"
        assert event.data["result"]["price"] == 150.0
        assert event.data["success"] is True

    def test_error(self) -> None:
        """Test error factory."""
        event = StreamEvent.error(
            workflow_id="wf-123",
            trace_id="trace-456",
            message="API rate limit exceeded",
            agent="research",
            error_type="RateLimitError",
        )

        assert event.type == StreamEventType.ERROR
        assert event.data["message"] == "API rate limit exceeded"
        assert event.data["agent"] == "research"
        assert event.data["error_type"] == "RateLimitError"

    def test_error_without_agent(self) -> None:
        """Test error factory without agent."""
        event = StreamEvent.error(
            workflow_id="wf-123",
            trace_id="trace-456",
            message="Internal error",
        )

        assert event.type == StreamEventType.ERROR
        assert "agent" not in event.data

    def test_heartbeat(self) -> None:
        """Test heartbeat factory."""
        event = StreamEvent.heartbeat(
            workflow_id="wf-123",
            trace_id="trace-456",
        )

        assert event.type == StreamEventType.HEARTBEAT
        assert event.data == {}
