"""Tests for the streaming API endpoint."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.routes import create_app
from src.streaming.events import StreamEvent, StreamEventType


class TestStreamingEndpointBasic:
    """Basic tests for the /analyze/stream endpoint."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def sample_portfolio(self) -> dict:
        """Create sample portfolio request."""
        return {
            "positions": [
                {"symbol": "AAPL", "quantity": 100},
            ],
        }

    def test_streaming_endpoint_validates_empty_positions(
        self, client: TestClient
    ) -> None:
        """Test that endpoint validates empty positions."""
        response = client.post(
            "/analyze/stream",
            json={"positions": []},  # Empty positions should fail
        )

        assert response.status_code == 422  # Validation error

    def test_streaming_endpoint_requires_positions(
        self, client: TestClient
    ) -> None:
        """Test that endpoint requires positions field."""
        response = client.post(
            "/analyze/stream",
            json={"user_request": "Analyze something"},
        )

        assert response.status_code == 422  # Validation error

    def test_streaming_endpoint_validates_symbol_format(
        self, client: TestClient
    ) -> None:
        """Test that endpoint validates symbol format."""
        response = client.post(
            "/analyze/stream",
            json={
                "positions": [
                    {"symbol": "", "quantity": 100},  # Empty symbol
                ],
            },
        )

        assert response.status_code == 422  # Validation error

    def test_streaming_endpoint_validates_quantity(
        self, client: TestClient
    ) -> None:
        """Test that endpoint validates quantity."""
        response = client.post(
            "/analyze/stream",
            json={
                "positions": [
                    {"symbol": "AAPL", "quantity": -100},  # Negative quantity
                ],
            },
        )

        assert response.status_code == 422  # Validation error


class TestStreamEventParsing:
    """Tests for parsing SSE events."""

    def test_parse_workflow_started_event(self) -> None:
        """Test parsing workflow_started event."""
        event = StreamEvent.workflow_started(
            workflow_id="wf-123",
            trace_id="trace-456",
            symbols=["AAPL", "GOOGL"],
            user_request="Analyze portfolio",
        )

        sse = event.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")

        json_str = sse[6:-2]
        parsed = json.loads(json_str)

        assert parsed["type"] == "workflow_started"
        assert parsed["workflow_id"] == "wf-123"
        assert parsed["trace_id"] == "trace-456"
        assert parsed["data"]["symbols"] == ["AAPL", "GOOGL"]

    def test_parse_agent_started_event(self) -> None:
        """Test parsing agent_started event."""
        event = StreamEvent.agent_started(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
        )

        sse = event.to_sse()
        json_str = sse[6:-2]
        parsed = json.loads(json_str)

        assert parsed["type"] == "agent_started"
        assert parsed["agent"] == "research"

    def test_parse_agent_completed_event(self) -> None:
        """Test parsing agent_completed event."""
        event = StreamEvent.agent_completed(
            workflow_id="wf-123",
            trace_id="trace-456",
            agent="research",
            output={"count": 5},
            duration_ms=1500.0,
        )

        sse = event.to_sse()
        json_str = sse[6:-2]
        parsed = json.loads(json_str)

        assert parsed["type"] == "agent_completed"
        assert parsed["agent"] == "research"
        assert parsed["data"]["output"]["count"] == 5
        assert parsed["data"]["duration_ms"] == 1500.0

    def test_parse_workflow_completed_event(self) -> None:
        """Test parsing workflow_completed event."""
        event = StreamEvent.workflow_completed(
            workflow_id="wf-123",
            trace_id="trace-456",
            status="completed",
            has_errors=False,
            result={"latency_ms": 5000},
        )

        sse = event.to_sse()
        json_str = sse[6:-2]
        parsed = json.loads(json_str)

        assert parsed["type"] == "workflow_completed"
        assert parsed["data"]["status"] == "completed"
        assert parsed["data"]["has_errors"] is False
        assert parsed["data"]["result"]["latency_ms"] == 5000

    def test_parse_error_event(self) -> None:
        """Test parsing error event."""
        event = StreamEvent.error(
            workflow_id="wf-123",
            trace_id="trace-456",
            message="API rate limit exceeded",
            agent="research",
            error_type="RateLimitError",
        )

        sse = event.to_sse()
        json_str = sse[6:-2]
        parsed = json.loads(json_str)

        assert parsed["type"] == "error"
        assert parsed["data"]["message"] == "API rate limit exceeded"
        assert parsed["data"]["agent"] == "research"
        assert parsed["data"]["error_type"] == "RateLimitError"

    def test_parse_heartbeat_event(self) -> None:
        """Test parsing heartbeat event."""
        event = StreamEvent.heartbeat(
            workflow_id="wf-123",
            trace_id="trace-456",
        )

        sse = event.to_sse()
        json_str = sse[6:-2]
        parsed = json.loads(json_str)

        assert parsed["type"] == "heartbeat"
        assert parsed["data"] == {}


class TestStreamEventSequence:
    """Tests for the expected sequence of streaming events."""

    def test_typical_event_sequence(self) -> None:
        """Test creating a typical sequence of events."""
        workflow_id = "wf-123"
        trace_id = "trace-456"

        # Create typical sequence
        events = [
            StreamEvent.workflow_started(
                workflow_id=workflow_id,
                trace_id=trace_id,
                symbols=["AAPL"],
            ),
            StreamEvent.agent_started(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="research",
            ),
            StreamEvent.agent_progress(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="research",
                message="Fetching data...",
            ),
            StreamEvent.agent_completed(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="research",
                duration_ms=1000.0,
            ),
            StreamEvent.agent_started(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="analysis",
            ),
            StreamEvent.agent_completed(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="analysis",
                duration_ms=800.0,
            ),
            StreamEvent.agent_started(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="recommendation",
            ),
            StreamEvent.agent_completed(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="recommendation",
                duration_ms=500.0,
            ),
            StreamEvent.workflow_completed(
                workflow_id=workflow_id,
                trace_id=trace_id,
                status="completed",
            ),
        ]

        # Verify sequence
        assert events[0].type == StreamEventType.WORKFLOW_STARTED
        assert events[-1].type == StreamEventType.WORKFLOW_COMPLETED

        # Verify all have same workflow_id
        for event in events:
            assert event.workflow_id == workflow_id
            assert event.trace_id == trace_id

    def test_event_sequence_with_error(self) -> None:
        """Test event sequence when an error occurs."""
        workflow_id = "wf-123"
        trace_id = "trace-456"

        events = [
            StreamEvent.workflow_started(
                workflow_id=workflow_id,
                trace_id=trace_id,
                symbols=["AAPL"],
            ),
            StreamEvent.agent_started(
                workflow_id=workflow_id,
                trace_id=trace_id,
                agent="research",
            ),
            StreamEvent.error(
                workflow_id=workflow_id,
                trace_id=trace_id,
                message="API error",
                agent="research",
            ),
            StreamEvent.workflow_completed(
                workflow_id=workflow_id,
                trace_id=trace_id,
                status="failed",
                has_errors=True,
            ),
        ]

        # Find error event
        error_events = [e for e in events if e.type == StreamEventType.ERROR]
        assert len(error_events) == 1
        assert error_events[0].data["agent"] == "research"

        # Final event should indicate failure
        assert events[-1].data["has_errors"] is True
