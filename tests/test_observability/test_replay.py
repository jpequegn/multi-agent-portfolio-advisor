"""Tests for the replay system module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.observability.replay import (
    CapturedSpan,
    CapturedToolResponse,
    ComparisonEngine,
    ReplayableRequest,
    ReplayComparison,
    ReplayEngine,
    ReplayMode,
    ReplayResult,
    ReplayStatus,
    ReplayStepResult,
    ToolMockManager,
    TraceRetriever,
    create_replay_report,
)


# ============================================================================
# ReplayMode Tests
# ============================================================================


class TestReplayMode:
    """Tests for ReplayMode enum."""

    def test_modes_exist(self) -> None:
        """Test all replay modes are defined."""
        assert ReplayMode.FULL.value == "full"
        assert ReplayMode.MOCK_TOOLS.value == "mock_tools"
        assert ReplayMode.STEP_BY_STEP.value == "step_by_step"


class TestReplayStatus:
    """Tests for ReplayStatus enum."""

    def test_statuses_exist(self) -> None:
        """Test all replay statuses are defined."""
        assert ReplayStatus.PENDING.value == "pending"
        assert ReplayStatus.RUNNING.value == "running"
        assert ReplayStatus.PAUSED.value == "paused"
        assert ReplayStatus.COMPLETED.value == "completed"
        assert ReplayStatus.FAILED.value == "failed"


# ============================================================================
# CapturedSpan Tests
# ============================================================================


class TestCapturedSpan:
    """Tests for CapturedSpan dataclass."""

    def test_create_captured_span(self) -> None:
        """Test creating a captured span."""
        now = datetime.now(UTC)
        span = CapturedSpan(
            span_id="span-123",
            name="research_agent",
            span_type="span",
            parent_span_id="parent-456",
            input_data={"symbols": ["AAPL"]},
            output_data={"result": "data"},
            metadata={"level": "agent"},
            start_time=now,
            end_time=now + timedelta(seconds=5),
            level="agent",
            status="success",
            error=None,
        )

        assert span.span_id == "span-123"
        assert span.name == "research_agent"
        assert span.span_type == "span"
        assert span.input_data == {"symbols": ["AAPL"]}
        assert span.output_data == {"result": "data"}
        assert span.level == "agent"
        assert span.status == "success"

    def test_to_dict(self) -> None:
        """Test converting span to dictionary."""
        now = datetime.now(UTC)
        span = CapturedSpan(
            span_id="span-123",
            name="test_span",
            span_type="tool",
            parent_span_id=None,
            input_data={"key": "value"},
            output_data={"result": 42},
            metadata={},
            start_time=now,
            end_time=now + timedelta(seconds=1),
            level="tool",
            status="success",
            error=None,
        )

        d = span.to_dict()
        assert d["span_id"] == "span-123"
        assert d["name"] == "test_span"
        assert d["span_type"] == "tool"
        assert d["input_data"] == {"key": "value"}
        assert d["output_data"] == {"result": 42}
        assert d["status"] == "success"

    def test_from_dict(self) -> None:
        """Test creating span from dictionary."""
        now = datetime.now(UTC)
        data = {
            "span_id": "span-456",
            "name": "analysis_agent",
            "span_type": "span",
            "parent_span_id": "parent-789",
            "input_data": {"data": [1, 2, 3]},
            "output_data": {"analysis": "complete"},
            "metadata": {"agent_name": "analysis"},
            "start_time": now.isoformat(),
            "end_time": (now + timedelta(seconds=10)).isoformat(),
            "level": "agent",
            "status": "success",
            "error": None,
        }

        span = CapturedSpan.from_dict(data)
        assert span.span_id == "span-456"
        assert span.name == "analysis_agent"
        assert span.input_data == {"data": [1, 2, 3]}
        assert span.level == "agent"


class TestCapturedToolResponse:
    """Tests for CapturedToolResponse dataclass."""

    def test_create_tool_response(self) -> None:
        """Test creating a captured tool response."""
        response = CapturedToolResponse(
            tool_name="get_market_data",
            input_data={"symbol": "AAPL"},
            output_data={"price": 150.0},
            execution_time_ms=125.5,
            was_error=False,
            error_message=None,
        )

        assert response.tool_name == "get_market_data"
        assert response.input_data == {"symbol": "AAPL"}
        assert response.output_data == {"price": 150.0}
        assert response.execution_time_ms == 125.5
        assert response.was_error is False

    def test_to_dict(self) -> None:
        """Test converting tool response to dictionary."""
        response = CapturedToolResponse(
            tool_name="search_news",
            input_data={"query": "AAPL earnings"},
            output_data={"articles": []},
            execution_time_ms=500.0,
            was_error=False,
            error_message=None,
        )

        d = response.to_dict()
        assert d["tool_name"] == "search_news"
        assert d["execution_time_ms"] == 500.0
        assert d["was_error"] is False

    def test_from_dict(self) -> None:
        """Test creating tool response from dictionary."""
        data = {
            "tool_name": "calculate_risk",
            "input_data": {"portfolio": []},
            "output_data": {"risk_score": 0.5},
            "execution_time_ms": 200.0,
            "was_error": False,
            "error_message": None,
        }

        response = CapturedToolResponse.from_dict(data)
        assert response.tool_name == "calculate_risk"
        assert response.output_data == {"risk_score": 0.5}


# ============================================================================
# ReplayableRequest Tests
# ============================================================================


class TestReplayableRequest:
    """Tests for ReplayableRequest dataclass."""

    @pytest.fixture
    def sample_request(self) -> ReplayableRequest:
        """Create a sample replayable request."""
        now = datetime.now(UTC)
        return ReplayableRequest(
            request_id="req-123",
            trace_id="trace-456",
            workflow_id="workflow-789",
            input_state={
                "portfolio": {"positions": []},
                "user_request": "Analyze my portfolio",
            },
            spans=[
                CapturedSpan(
                    span_id="span-1",
                    name="research_agent",
                    span_type="span",
                    parent_span_id=None,
                    input_data={},
                    output_data={"research": "data"},
                    metadata={"level": "agent"},
                    start_time=now,
                    end_time=now + timedelta(seconds=5),
                    level="agent",
                    status="success",
                    error=None,
                )
            ],
            tool_responses=[
                CapturedToolResponse(
                    tool_name="get_market_data",
                    input_data={"symbol": "AAPL"},
                    output_data={"price": 150.0},
                    execution_time_ms=100.0,
                    was_error=False,
                    error_message=None,
                )
            ],
            timestamp=now,
            user_id="user-abc",
            session_id="session-xyz",
            original_status="completed",
            original_error=None,
            agent_sequence=["research_agent", "analysis_agent"],
            total_duration_ms=10000.0,
        )

    def test_create_request(self, sample_request: ReplayableRequest) -> None:
        """Test creating a replayable request."""
        assert sample_request.request_id == "req-123"
        assert sample_request.trace_id == "trace-456"
        assert sample_request.workflow_id == "workflow-789"
        assert len(sample_request.spans) == 1
        assert len(sample_request.tool_responses) == 1
        assert sample_request.original_status == "completed"

    def test_to_dict(self, sample_request: ReplayableRequest) -> None:
        """Test converting request to dictionary."""
        d = sample_request.to_dict()
        assert d["request_id"] == "req-123"
        assert d["trace_id"] == "trace-456"
        assert len(d["spans"]) == 1
        assert len(d["tool_responses"]) == 1
        assert d["agent_sequence"] == ["research_agent", "analysis_agent"]

    def test_from_dict(self, sample_request: ReplayableRequest) -> None:
        """Test creating request from dictionary."""
        d = sample_request.to_dict()
        restored = ReplayableRequest.from_dict(d)

        assert restored.request_id == sample_request.request_id
        assert restored.trace_id == sample_request.trace_id
        assert len(restored.spans) == len(sample_request.spans)
        assert len(restored.tool_responses) == len(sample_request.tool_responses)

    def test_json_serialization(self, sample_request: ReplayableRequest) -> None:
        """Test JSON serialization round-trip."""
        json_str = sample_request.to_json()
        restored = ReplayableRequest.from_json(json_str)

        assert restored.request_id == sample_request.request_id
        assert restored.trace_id == sample_request.trace_id


# ============================================================================
# ReplayStepResult Tests
# ============================================================================


class TestReplayStepResult:
    """Tests for ReplayStepResult dataclass."""

    def test_create_step_result(self) -> None:
        """Test creating a step result."""
        result = ReplayStepResult(
            step_name="research_agent",
            agent_name="research",
            input_data={"symbols": ["AAPL"]},
            output_data={"data": "result"},
            duration_ms=500.0,
            status="success",
            error=None,
            matches_original=True,
        )

        assert result.step_name == "research_agent"
        assert result.agent_name == "research"
        assert result.status == "success"
        assert result.matches_original is True

    def test_to_dict(self) -> None:
        """Test converting step result to dictionary."""
        result = ReplayStepResult(
            step_name="analysis",
            agent_name="analysis_agent",
            input_data={},
            output_data={"metrics": {}},
            duration_ms=1000.0,
            status="success",
            error=None,
            matches_original=True,
        )

        d = result.to_dict()
        assert d["step_name"] == "analysis"
        assert d["duration_ms"] == 1000.0
        assert d["matches_original"] is True


# ============================================================================
# ReplayComparison Tests
# ============================================================================


class TestReplayComparison:
    """Tests for ReplayComparison dataclass."""

    def test_create_comparison(self) -> None:
        """Test creating a comparison."""
        comparison = ReplayComparison(
            field="research.summary",
            original_value="Original summary",
            replay_value="Replay summary",
            matches=False,
            difference_type="changed",
        )

        assert comparison.field == "research.summary"
        assert comparison.matches is False
        assert comparison.difference_type == "changed"

    def test_matching_comparison(self) -> None:
        """Test a matching comparison."""
        comparison = ReplayComparison(
            field="status",
            original_value="completed",
            replay_value="completed",
            matches=True,
            difference_type="",
        )

        assert comparison.matches is True

    def test_to_dict(self) -> None:
        """Test converting comparison to dictionary."""
        comparison = ReplayComparison(
            field="error",
            original_value=None,
            replay_value="New error",
            matches=False,
            difference_type="added",
        )

        d = comparison.to_dict()
        assert d["field"] == "error"
        assert d["difference_type"] == "added"


# ============================================================================
# ToolMockManager Tests
# ============================================================================


class TestToolMockManager:
    """Tests for ToolMockManager class."""

    @pytest.fixture
    def mock_manager(self) -> ToolMockManager:
        """Create a mock manager instance."""
        return ToolMockManager()

    def test_load_responses(self, mock_manager: ToolMockManager) -> None:
        """Test loading captured responses."""
        responses = [
            CapturedToolResponse(
                tool_name="get_market_data",
                input_data={"symbol": "AAPL"},
                output_data={"price": 150.0},
                execution_time_ms=100.0,
                was_error=False,
                error_message=None,
            ),
            CapturedToolResponse(
                tool_name="get_market_data",
                input_data={"symbol": "GOOGL"},
                output_data={"price": 2800.0},
                execution_time_ms=120.0,
                was_error=False,
                error_message=None,
            ),
        ]

        mock_manager.load_responses(responses)

        # Should return first response
        response = mock_manager.get_mock_response("get_market_data", {"symbol": "AAPL"})
        assert response is not None
        assert response.output_data == {"price": 150.0}

        # Should return second response
        response = mock_manager.get_mock_response("get_market_data", {"symbol": "GOOGL"})
        assert response is not None
        assert response.output_data == {"price": 2800.0}

    def test_get_mock_response_not_found(self, mock_manager: ToolMockManager) -> None:
        """Test getting mock response for unknown tool."""
        response = mock_manager.get_mock_response("unknown_tool", {})
        assert response is None

    def test_mock_exhausted(self, mock_manager: ToolMockManager) -> None:
        """Test behavior when mock responses are exhausted."""
        responses = [
            CapturedToolResponse(
                tool_name="single_tool",
                input_data={},
                output_data={"data": "only_one"},
                execution_time_ms=50.0,
                was_error=False,
                error_message=None,
            ),
        ]

        mock_manager.load_responses(responses)

        # First call succeeds
        response = mock_manager.get_mock_response("single_tool", {})
        assert response is not None

        # Second call returns None (exhausted)
        response = mock_manager.get_mock_response("single_tool", {})
        assert response is None

    def test_reset(self, mock_manager: ToolMockManager) -> None:
        """Test resetting call counts."""
        responses = [
            CapturedToolResponse(
                tool_name="test_tool",
                input_data={},
                output_data={"data": "value"},
                execution_time_ms=50.0,
                was_error=False,
                error_message=None,
            ),
        ]

        mock_manager.load_responses(responses)

        # Use the response
        mock_manager.get_mock_response("test_tool", {})

        # Reset
        mock_manager.reset()

        # Should work again
        response = mock_manager.get_mock_response("test_tool", {})
        assert response is not None


# ============================================================================
# ComparisonEngine Tests
# ============================================================================


class TestComparisonEngine:
    """Tests for ComparisonEngine class."""

    @pytest.fixture
    def engine(self) -> ComparisonEngine:
        """Create a comparison engine instance."""
        return ComparisonEngine()

    def test_compare_identical_states(self, engine: ComparisonEngine) -> None:
        """Test comparing identical states."""
        state = {"key1": "value1", "key2": 42}
        comparisons = engine.compare_states(state, state.copy())

        assert len(comparisons) == 2
        assert all(c.matches for c in comparisons)

    def test_compare_different_states(self, engine: ComparisonEngine) -> None:
        """Test comparing different states."""
        original = {"key1": "value1", "key2": 42}
        replay = {"key1": "value1", "key2": 100}  # key2 changed

        comparisons = engine.compare_states(original, replay)

        key1_comp = next(c for c in comparisons if c.field == "key1")
        assert key1_comp.matches is True

        key2_comp = next(c for c in comparisons if c.field == "key2")
        assert key2_comp.matches is False
        assert key2_comp.difference_type == "changed"

    def test_compare_missing_field(self, engine: ComparisonEngine) -> None:
        """Test comparing states with missing fields."""
        original = {"key1": "value1", "key2": 42}
        replay = {"key1": "value1"}  # key2 missing

        comparisons = engine.compare_states(original, replay)

        key2_comp = next(c for c in comparisons if c.field == "key2")
        assert key2_comp.matches is False
        assert key2_comp.difference_type == "missing"

    def test_compare_added_field(self, engine: ComparisonEngine) -> None:
        """Test comparing states with added fields."""
        original = {"key1": "value1"}
        replay = {"key1": "value1", "key2": "new"}  # key2 added

        comparisons = engine.compare_states(original, replay)

        key2_comp = next(c for c in comparisons if c.field == "key2")
        assert key2_comp.matches is False
        assert key2_comp.difference_type == "added"

    def test_compare_type_mismatch(self, engine: ComparisonEngine) -> None:
        """Test comparing states with type mismatches."""
        original = {"key1": "string"}
        replay = {"key1": 123}  # Type changed

        comparisons = engine.compare_states(original, replay)

        key1_comp = next(c for c in comparisons if c.field == "key1")
        assert key1_comp.matches is False
        assert key1_comp.difference_type == "type_mismatch"

    def test_calculate_match_percentage_full(self, engine: ComparisonEngine) -> None:
        """Test match percentage calculation with all matches."""
        comparisons = [
            ReplayComparison("f1", "v1", "v1", True, ""),
            ReplayComparison("f2", "v2", "v2", True, ""),
        ]

        percentage = engine.calculate_match_percentage(comparisons)
        assert percentage == 100.0

    def test_calculate_match_percentage_partial(self, engine: ComparisonEngine) -> None:
        """Test match percentage calculation with partial matches."""
        comparisons = [
            ReplayComparison("f1", "v1", "v1", True, ""),
            ReplayComparison("f2", "v2", "v3", False, "changed"),
        ]

        percentage = engine.calculate_match_percentage(comparisons)
        assert percentage == 50.0

    def test_calculate_match_percentage_empty(self, engine: ComparisonEngine) -> None:
        """Test match percentage calculation with empty comparisons."""
        percentage = engine.calculate_match_percentage([])
        assert percentage == 100.0

    def test_find_divergence_point(self, engine: ComparisonEngine) -> None:
        """Test finding divergence point."""
        now = datetime.now(UTC)
        original_spans = [
            CapturedSpan(
                span_id="s1",
                name="step1",
                span_type="span",
                parent_span_id=None,
                input_data={},
                output_data={"result": "ok"},
                metadata={},
                start_time=now,
                end_time=now,
                level="agent",
                status="success",
                error=None,
            )
        ]

        replay_steps = [
            ReplayStepResult(
                step_name="step1",
                agent_name="step1",
                input_data={},
                output_data={"result": "ok"},
                duration_ms=100.0,
                status="success",
                error=None,
                matches_original=False,  # Diverged
            )
        ]

        point, reason = engine.find_divergence_point(original_spans, replay_steps)
        assert point == "step1"
        assert reason is not None


# ============================================================================
# TraceRetriever Tests
# ============================================================================


class TestTraceRetriever:
    """Tests for TraceRetriever class."""

    @pytest.fixture
    def retriever(self) -> TraceRetriever:
        """Create a trace retriever instance."""
        return TraceRetriever()

    def test_parse_observation(self, retriever: TraceRetriever) -> None:
        """Test parsing a Langfuse observation."""
        obs = {
            "id": "obs-123",
            "name": "research_agent",
            "type": "span",
            "parentObservationId": "parent-456",
            "input": {"symbols": ["AAPL"]},
            "output": {"data": "result"},
            "metadata": {"level": "agent", "agent_name": "research"},
            "startTime": "2024-01-01T12:00:00+00:00",
            "endTime": "2024-01-01T12:00:05+00:00",
        }

        span = retriever.parse_observation(obs)

        assert span.span_id == "obs-123"
        assert span.name == "research_agent"
        assert span.span_type == "span"
        assert span.parent_span_id == "parent-456"
        assert span.input_data == {"symbols": ["AAPL"]}
        assert span.output_data == {"data": "result"}

    def test_parse_observation_with_error(self, retriever: TraceRetriever) -> None:
        """Test parsing an observation with an error."""
        obs = {
            "id": "obs-err",
            "name": "failed_span",
            "type": "span",
            "level": "ERROR",
            "statusMessage": "Something went wrong",
            "startTime": "2024-01-01T12:00:00+00:00",
        }

        span = retriever.parse_observation(obs)

        assert span.status == "error"
        assert span.error == "Something went wrong"

    def test_extract_tool_responses(self, retriever: TraceRetriever) -> None:
        """Test extracting tool responses from spans."""
        now = datetime.now(UTC)
        spans = [
            CapturedSpan(
                span_id="s1",
                name="get_market_data",
                span_type="tool",
                parent_span_id=None,
                input_data={"symbol": "AAPL"},
                output_data={"price": 150.0},
                metadata={"level": "tool"},
                start_time=now,
                end_time=now + timedelta(milliseconds=100),
                level="tool",
                status="success",
                error=None,
            ),
            CapturedSpan(
                span_id="s2",
                name="agent_span",
                span_type="span",
                parent_span_id=None,
                input_data={},
                output_data={},
                metadata={"level": "agent"},
                start_time=now,
                end_time=now,
                level="agent",
                status="success",
                error=None,
            ),
        ]

        tool_responses = retriever.extract_tool_responses(spans)

        assert len(tool_responses) == 1
        assert tool_responses[0].tool_name == "get_market_data"
        assert tool_responses[0].output_data == {"price": 150.0}

    def test_extract_agent_sequence(self, retriever: TraceRetriever) -> None:
        """Test extracting agent sequence from spans."""
        now = datetime.now(UTC)
        spans = [
            CapturedSpan(
                span_id="s1",
                name="research_agent",
                span_type="span",
                parent_span_id=None,
                input_data={},
                output_data={},
                metadata={"level": "agent", "agent_name": "research"},
                start_time=now,
                end_time=now,
                level="agent",
                status="success",
                error=None,
            ),
            CapturedSpan(
                span_id="s2",
                name="analysis_agent",
                span_type="span",
                parent_span_id=None,
                input_data={},
                output_data={},
                metadata={"level": "agent", "agent_name": "analysis"},
                start_time=now + timedelta(seconds=5),
                end_time=now + timedelta(seconds=10),
                level="agent",
                status="success",
                error=None,
            ),
        ]

        sequence = retriever.extract_agent_sequence(spans)

        assert sequence == ["research", "analysis"]


# ============================================================================
# ReplayEngine Tests
# ============================================================================


class TestReplayEngine:
    """Tests for ReplayEngine class."""

    @pytest.fixture
    def engine(self) -> ReplayEngine:
        """Create a replay engine instance."""
        return ReplayEngine()

    def test_set_step_callback(self, engine: ReplayEngine) -> None:
        """Test setting step callback."""
        callback = MagicMock()
        engine.set_step_callback(callback)
        assert engine._step_callback == callback

    def test_pause_resume(self, engine: ReplayEngine) -> None:
        """Test pause and resume."""
        assert engine._paused is False

        engine.pause()
        assert engine._paused is True

        engine.resume()
        assert engine._paused is False

    @pytest.mark.asyncio
    async def test_capture_request_not_found(self, engine: ReplayEngine) -> None:
        """Test capturing a non-existent trace."""
        with patch.object(engine._retriever, "get_trace", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            with pytest.raises(ValueError, match="Trace not found"):
                await engine.capture_request("non-existent-trace")

    @pytest.mark.asyncio
    async def test_capture_request_success(self, engine: ReplayEngine) -> None:
        """Test successful request capture."""
        mock_trace = {
            "id": "trace-123",
            "input": {"portfolio": {}, "user_request": "Test"},
            "metadata": {"workflow_id": "wf-456"},
            "userId": "user-1",
            "sessionId": "sess-1",
        }

        mock_observations = [
            {
                "id": "obs-1",
                "name": "research_agent",
                "type": "span",
                "input": {},
                "output": {"data": "result"},
                "metadata": {"level": "agent", "agent_name": "research"},
                "startTime": "2024-01-01T12:00:00+00:00",
                "endTime": "2024-01-01T12:00:05+00:00",
            }
        ]

        with (
            patch.object(
                engine._retriever, "get_trace", new_callable=AsyncMock
            ) as mock_get_trace,
            patch.object(
                engine._retriever, "get_trace_observations", new_callable=AsyncMock
            ) as mock_get_obs,
        ):
            mock_get_trace.return_value = mock_trace
            mock_get_obs.return_value = mock_observations

            request = await engine.capture_request("trace-123")

            assert request.trace_id == "trace-123"
            assert request.workflow_id == "wf-456"
            assert request.user_id == "user-1"
            assert len(request.spans) == 1


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestCreateReplayReport:
    """Tests for create_replay_report function."""

    def test_create_report_success(self) -> None:
        """Test creating a report for successful replay."""
        result = ReplayResult(
            replay_id="replay-123",
            request_id="req-456",
            mode=ReplayMode.MOCK_TOOLS,
            status=ReplayStatus.COMPLETED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            total_duration_ms=1500.0,
            final_state={"status": "completed"},
            step_results=[
                ReplayStepResult(
                    step_name="research",
                    agent_name="research_agent",
                    input_data={},
                    output_data={},
                    duration_ms=500.0,
                    status="success",
                    error=None,
                    matches_original=True,
                )
            ],
            error=None,
            comparisons=[
                ReplayComparison("status", "completed", "completed", True, ""),
            ],
            overall_match=True,
            match_percentage=100.0,
            divergence_point=None,
            divergence_reason=None,
        )

        report = create_replay_report(result)

        assert "REPLAY REPORT" in report
        assert "replay-123" in report
        assert "mock_tools" in report
        assert "completed" in report
        assert "100.0%" in report
        assert "All fields match!" in report

    def test_create_report_with_divergence(self) -> None:
        """Test creating a report with divergence."""
        result = ReplayResult(
            replay_id="replay-789",
            request_id="req-abc",
            mode=ReplayMode.FULL,
            status=ReplayStatus.COMPLETED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            total_duration_ms=2000.0,
            final_state={"status": "failed"},
            step_results=[],
            error=None,
            comparisons=[
                ReplayComparison("status", "completed", "failed", False, "changed"),
            ],
            overall_match=False,
            match_percentage=50.0,
            divergence_point="analysis_agent",
            divergence_reason="Output mismatch at analysis_agent",
        )

        report = create_replay_report(result)

        assert "NO" in report  # Overall match
        assert "50.0%" in report
        assert "analysis_agent" in report
        assert "changed" in report

    def test_create_report_with_error(self) -> None:
        """Test creating a report with error."""
        result = ReplayResult(
            replay_id="replay-err",
            request_id="req-err",
            mode=ReplayMode.FULL,
            status=ReplayStatus.FAILED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            total_duration_ms=500.0,
            final_state=None,
            step_results=[
                ReplayStepResult(
                    step_name="failed_step",
                    agent_name="agent",
                    input_data={},
                    output_data=None,
                    duration_ms=100.0,
                    status="error",
                    error="Connection timeout",
                    matches_original=False,
                )
            ],
            error="Replay failed: Connection timeout",
            comparisons=[],
            overall_match=False,
            match_percentage=0.0,
            divergence_point=None,
            divergence_reason=None,
        )

        report = create_replay_report(result)

        assert "ERROR" in report
        assert "Connection timeout" in report
        assert "failed" in report.lower()
