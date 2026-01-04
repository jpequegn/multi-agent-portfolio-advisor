"""Tests for the failure analysis module."""

from datetime import UTC, datetime, timedelta

import pytest

from src.observability.failures import (
    STANDARD_PATTERNS,
    FailureAlert,
    FailureAlertManager,
    FailureAlertThresholds,
    FailureAlertType,
    FailureAnalysis,
    FailureCategory,
    FailureCategorizer,
    FailurePattern,
    FailureRecord,
    FailureReport,
    FailureSeverity,
    FailureTracker,
    PatternDetector,
    generate_failure_report,
    get_failure_alert_manager,
    get_failure_tracker,
    reset_failure_tracking,
)


# ============================================================================
# FailureCategory Tests
# ============================================================================


class TestFailureCategory:
    """Tests for FailureCategory enum."""

    def test_categories_exist(self) -> None:
        """Test all failure categories are defined."""
        assert FailureCategory.LLM_ERROR.value == "llm_error"
        assert FailureCategory.TOOL_ERROR.value == "tool_error"
        assert FailureCategory.STATE_ERROR.value == "state_error"
        assert FailureCategory.TIMEOUT.value == "timeout"
        assert FailureCategory.VALIDATION.value == "validation"
        assert FailureCategory.EXTERNAL_API.value == "external_api"
        assert FailureCategory.AUTHENTICATION.value == "authentication"
        assert FailureCategory.RESOURCE.value == "resource"
        assert FailureCategory.NETWORK.value == "network"
        assert FailureCategory.UNKNOWN.value == "unknown"


class TestFailureSeverity:
    """Tests for FailureSeverity enum."""

    def test_severities_exist(self) -> None:
        """Test all severity levels are defined."""
        assert FailureSeverity.LOW.value == "low"
        assert FailureSeverity.MEDIUM.value == "medium"
        assert FailureSeverity.HIGH.value == "high"
        assert FailureSeverity.CRITICAL.value == "critical"


# ============================================================================
# FailureRecord Tests
# ============================================================================


class TestFailureRecord:
    """Tests for FailureRecord dataclass."""

    def test_create_failure_record(self) -> None:
        """Test creating a failure record."""
        record = FailureRecord(
            failure_id="failure-1",
            trace_id="trace-123",
            category=FailureCategory.LLM_ERROR,
            severity=FailureSeverity.HIGH,
            agent_name="research_agent",
            error_message="Rate limit exceeded",
        )

        assert record.failure_id == "failure-1"
        assert record.trace_id == "trace-123"
        assert record.category == FailureCategory.LLM_ERROR
        assert record.severity == FailureSeverity.HIGH
        assert record.agent_name == "research_agent"
        assert record.error_message == "Rate limit exceeded"

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        record = FailureRecord(
            failure_id="failure-1",
            trace_id="trace-123",
            category=FailureCategory.TIMEOUT,
            agent_name="test_agent",
            error_message="Test error",
        )

        d = record.to_dict()
        assert d["failure_id"] == "failure-1"
        assert d["category"] == "timeout"
        assert d["severity"] == "medium"
        assert "timestamp" in d

    def test_matches_pattern_category(self) -> None:
        """Test pattern matching by category."""
        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.LLM_ERROR,
            error_message="Rate limit",
        )

        pattern = FailurePattern(
            pattern_id="p1",
            name="Test",
            category=FailureCategory.LLM_ERROR,
        )

        assert record.matches_pattern(pattern) is True

        pattern_wrong = FailurePattern(
            pattern_id="p2",
            name="Test",
            category=FailureCategory.TOOL_ERROR,
        )
        assert record.matches_pattern(pattern_wrong) is False

    def test_matches_pattern_agent(self) -> None:
        """Test pattern matching by agent name."""
        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.TOOL_ERROR,
            agent_name="research_agent",
            error_message="Tool failed",
        )

        pattern = FailurePattern(
            pattern_id="p1",
            name="Test",
            agent_name="research_agent",
        )
        assert record.matches_pattern(pattern) is True

        pattern_wrong = FailurePattern(
            pattern_id="p2",
            name="Test",
            agent_name="analysis_agent",
        )
        assert record.matches_pattern(pattern_wrong) is False

    def test_matches_pattern_message_contains(self) -> None:
        """Test pattern matching by error message substring."""
        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.LLM_ERROR,
            error_message="API rate limit exceeded for model",
        )

        pattern = FailurePattern(
            pattern_id="p1",
            name="Test",
            error_message_contains="rate limit",
        )
        assert record.matches_pattern(pattern) is True

        pattern_wrong = FailurePattern(
            pattern_id="p2",
            name="Test",
            error_message_contains="timeout",
        )
        assert record.matches_pattern(pattern_wrong) is False


# ============================================================================
# FailurePattern Tests
# ============================================================================


class TestFailurePattern:
    """Tests for FailurePattern dataclass."""

    def test_create_pattern(self) -> None:
        """Test creating a failure pattern."""
        pattern = FailurePattern(
            pattern_id="rate_limit",
            name="Rate Limit Exceeded",
            description="API rate limit was hit",
            category=FailureCategory.LLM_ERROR,
            error_message_contains="rate limit",
            suggested_fix="Implement backoff",
        )

        assert pattern.pattern_id == "rate_limit"
        assert pattern.name == "Rate Limit Exceeded"
        assert pattern.category == FailureCategory.LLM_ERROR
        assert pattern.suggested_fix == "Implement backoff"

    def test_to_dict(self) -> None:
        """Test converting pattern to dictionary."""
        pattern = FailurePattern(
            pattern_id="test",
            name="Test Pattern",
            category=FailureCategory.TIMEOUT,
            severity_override=FailureSeverity.HIGH,
        )

        d = pattern.to_dict()
        assert d["pattern_id"] == "test"
        assert d["category"] == "timeout"
        assert d["severity_override"] == "high"


class TestStandardPatterns:
    """Tests for standard failure patterns."""

    def test_standard_patterns_defined(self) -> None:
        """Test standard patterns are defined."""
        assert len(STANDARD_PATTERNS) >= 5

        pattern_ids = [p.pattern_id for p in STANDARD_PATTERNS]
        assert "rate_limit" in pattern_ids
        assert "api_timeout" in pattern_ids
        assert "auth_failure" in pattern_ids

    def test_standard_patterns_have_fixes(self) -> None:
        """Test standard patterns have suggested fixes."""
        for pattern in STANDARD_PATTERNS:
            assert pattern.suggested_fix is not None
            assert len(pattern.suggested_fix) > 0


# ============================================================================
# FailureCategorizer Tests
# ============================================================================


class TestFailureCategorizer:
    """Tests for FailureCategorizer class."""

    def test_categorize_llm_error(self) -> None:
        """Test categorizing LLM errors."""
        categorizer = FailureCategorizer()

        assert (
            categorizer.categorize("Rate limit exceeded for API")
            == FailureCategory.LLM_ERROR
        )
        assert (
            categorizer.categorize("Context length exceeded")
            == FailureCategory.LLM_ERROR
        )
        assert (
            categorizer.categorize("Invalid token count")
            == FailureCategory.LLM_ERROR
        )

    def test_categorize_timeout(self) -> None:
        """Test categorizing timeout errors."""
        categorizer = FailureCategorizer()

        assert (
            categorizer.categorize("Request timed out")
            == FailureCategory.TIMEOUT
        )
        assert (
            categorizer.categorize("Deadline expired")
            == FailureCategory.TIMEOUT
        )

    def test_categorize_authentication(self) -> None:
        """Test categorizing authentication errors."""
        categorizer = FailureCategorizer()

        assert (
            categorizer.categorize("401 Unauthorized")
            == FailureCategory.AUTHENTICATION
        )
        assert (
            categorizer.categorize("Invalid credentials")
            == FailureCategory.AUTHENTICATION
        )

    def test_categorize_network(self) -> None:
        """Test categorizing network errors."""
        categorizer = FailureCategorizer()

        assert (
            categorizer.categorize("Connection refused")
            == FailureCategory.NETWORK
        )
        assert (
            categorizer.categorize("DNS resolution failed")
            == FailureCategory.NETWORK
        )

    def test_categorize_validation(self) -> None:
        """Test categorizing validation errors."""
        categorizer = FailureCategorizer()

        assert (
            categorizer.categorize("Invalid JSON format")
            == FailureCategory.VALIDATION
        )
        assert (
            categorizer.categorize("Schema validation failed")
            == FailureCategory.VALIDATION
        )

    def test_categorize_unknown(self) -> None:
        """Test categorizing unknown errors."""
        categorizer = FailureCategorizer()

        assert (
            categorizer.categorize("Some random error message")
            == FailureCategory.UNKNOWN
        )

    def test_categorize_with_context(self) -> None:
        """Test categorization using context hints."""
        categorizer = FailureCategorizer()

        # Agent context suggests state error
        assert (
            categorizer.categorize(
                "Unknown error",
                context={"agent_name": "test_agent"},
            )
            == FailureCategory.STATE_ERROR
        )

        # Tool context suggests tool error
        assert (
            categorizer.categorize(
                "Unknown error",
                context={"tool_name": "test_tool"},
            )
            == FailureCategory.TOOL_ERROR
        )

    def test_assess_severity(self) -> None:
        """Test severity assessment."""
        categorizer = FailureCategorizer()

        # Critical categories
        assert (
            categorizer.assess_severity(FailureCategory.RESOURCE, "Out of memory")
            == FailureSeverity.CRITICAL
        )

        # High severity categories
        assert (
            categorizer.assess_severity(
                FailureCategory.AUTHENTICATION, "Auth failed"
            )
            == FailureSeverity.HIGH
        )

        # Medium severity categories
        assert (
            categorizer.assess_severity(FailureCategory.LLM_ERROR, "Rate limit")
            == FailureSeverity.MEDIUM
        )

        # Low severity when recovered
        assert (
            categorizer.assess_severity(
                FailureCategory.LLM_ERROR, "Rate limit", recovered=True
            )
            == FailureSeverity.LOW
        )


# ============================================================================
# PatternDetector Tests
# ============================================================================


class TestPatternDetector:
    """Tests for PatternDetector class."""

    def test_create_detector(self) -> None:
        """Test creating a pattern detector."""
        detector = PatternDetector()
        assert len(detector.patterns) == len(STANDARD_PATTERNS)

    def test_create_detector_custom_patterns(self) -> None:
        """Test creating detector with custom patterns."""
        custom = [
            FailurePattern(
                pattern_id="custom",
                name="Custom Pattern",
            )
        ]
        detector = PatternDetector(patterns=custom)
        assert len(detector.patterns) == 1

    def test_add_pattern(self) -> None:
        """Test adding a pattern."""
        detector = PatternDetector(patterns=[])
        detector.add_pattern(
            FailurePattern(pattern_id="new", name="New Pattern")
        )
        assert len(detector.patterns) == 1

    def test_detect_patterns(self) -> None:
        """Test pattern detection."""
        detector = PatternDetector()

        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.LLM_ERROR,
            error_message="API rate limit exceeded",
        )

        patterns = detector.detect_patterns(record)
        assert len(patterns) > 0
        pattern_ids = [p.pattern_id for p in patterns]
        assert "rate_limit" in pattern_ids

    def test_get_suggested_fixes(self) -> None:
        """Test getting suggested fixes."""
        detector = PatternDetector()

        patterns = [
            FailurePattern(
                pattern_id="p1",
                name="Pattern 1",
                suggested_fix="Fix 1",
            ),
            FailurePattern(
                pattern_id="p2",
                name="Pattern 2",
                suggested_fix="Fix 2",
            ),
            FailurePattern(
                pattern_id="p3",
                name="Pattern 3",
                suggested_fix="Fix 1",  # Duplicate
            ),
        ]

        fixes = detector.get_suggested_fixes(patterns)
        assert len(fixes) == 2
        assert "Fix 1" in fixes
        assert "Fix 2" in fixes

    def test_pattern_frequency(self) -> None:
        """Test pattern frequency tracking."""
        detector = PatternDetector()

        # Detect patterns for multiple similar failures
        for i in range(5):
            record = FailureRecord(
                failure_id=f"f{i}",
                trace_id=f"t{i}",
                category=FailureCategory.LLM_ERROR,
                error_message="Rate limit exceeded",
            )
            detector.detect_patterns(record)

        freq = detector.get_pattern_frequency()
        assert "rate_limit" in freq
        assert freq["rate_limit"] >= 5

    def test_get_top_patterns(self) -> None:
        """Test getting top patterns."""
        detector = PatternDetector()

        # Detect different patterns
        for i in range(10):
            record = FailureRecord(
                failure_id=f"f{i}",
                trace_id=f"t{i}",
                category=FailureCategory.LLM_ERROR,
                error_message="Rate limit exceeded",
            )
            detector.detect_patterns(record)

        for i in range(3):
            record = FailureRecord(
                failure_id=f"g{i}",
                trace_id=f"u{i}",
                category=FailureCategory.TIMEOUT,
                error_message="Request timed out",
            )
            detector.detect_patterns(record)

        top = detector.get_top_patterns(2)
        assert len(top) == 2
        assert top[0][0] == "rate_limit"
        assert top[0][1] == 10


# ============================================================================
# FailureTracker Tests
# ============================================================================


class TestFailureTracker:
    """Tests for FailureTracker class."""

    def test_create_tracker(self) -> None:
        """Test creating a failure tracker."""
        tracker = FailureTracker()
        assert tracker.categorizer is not None
        assert tracker.pattern_detector is not None

    def test_record_failure(self) -> None:
        """Test recording a failure."""
        tracker = FailureTracker()

        record = tracker.record_failure(
            trace_id="trace-123",
            error_message="Rate limit exceeded",
            agent_name="research_agent",
        )

        assert record.failure_id.startswith("failure-")
        assert record.trace_id == "trace-123"
        assert record.category == FailureCategory.LLM_ERROR
        assert record.agent_name == "research_agent"

    def test_record_failure_auto_categorization(self) -> None:
        """Test automatic failure categorization."""
        tracker = FailureTracker()

        # Should be categorized as timeout
        record = tracker.record_failure(
            trace_id="t1",
            error_message="Request timed out after 30 seconds",
        )
        assert record.category == FailureCategory.TIMEOUT

        # Should be categorized as authentication
        record = tracker.record_failure(
            trace_id="t2",
            error_message="401 Unauthorized: Invalid API key",
        )
        assert record.category == FailureCategory.AUTHENTICATION

    def test_record_failure_with_recovery(self) -> None:
        """Test recording a recovered failure."""
        tracker = FailureTracker()

        record = tracker.record_failure(
            trace_id="t1",
            error_message="Rate limit exceeded",
            recovered=True,
            recovery_action="Retried after backoff",
        )

        assert record.recovered is True
        assert record.recovery_action == "Retried after backoff"
        assert record.severity == FailureSeverity.LOW

    def test_analyze_failure(self) -> None:
        """Test failure analysis."""
        tracker = FailureTracker()

        # Record some failures
        record1 = tracker.record_failure(
            trace_id="t1",
            error_message="Rate limit exceeded",
            agent_name="research",
        )

        # Record similar failures
        for i in range(3):
            tracker.record_failure(
                trace_id=f"t{i+10}",
                error_message="Rate limit exceeded",
                agent_name="research",
            )

        analysis = tracker.analyze_failure(record1)

        assert analysis.failure.failure_id == record1.failure_id
        assert len(analysis.matched_patterns) > 0
        assert len(analysis.suggested_fixes) > 0
        assert len(analysis.similar_failures) >= 3
        assert analysis.root_cause is not None

    def test_find_similar_failures(self) -> None:
        """Test finding similar failures."""
        tracker = FailureTracker()

        # Record a base failure
        base = tracker.record_failure(
            trace_id="t1",
            error_message="Rate limit exceeded",
            agent_name="research",
        )

        # Record similar failures
        tracker.record_failure(
            trace_id="t2",
            error_message="Rate limit hit",
            agent_name="research",
        )
        tracker.record_failure(
            trace_id="t3",
            error_message="Rate limit exceeded",
            agent_name="research",
        )

        # Record different failure
        tracker.record_failure(
            trace_id="t4",
            error_message="Connection refused",
            agent_name="analysis",
        )

        similar = tracker.find_similar_failures(base)
        assert len(similar) >= 2

    def test_get_failures_with_filters(self) -> None:
        """Test querying failures with filters."""
        tracker = FailureTracker()

        tracker.record_failure(
            trace_id="t1",
            error_message="Rate limit",
            agent_name="agent1",
        )
        tracker.record_failure(
            trace_id="t2",
            error_message="Connection error",
            agent_name="agent2",
        )
        tracker.record_failure(
            trace_id="t3",
            error_message="Rate limit again",
            agent_name="agent1",
        )

        # Filter by category
        llm_failures = tracker.get_failures(category=FailureCategory.LLM_ERROR)
        assert len(llm_failures) == 2

        # Filter by agent
        agent1_failures = tracker.get_failures(agent_name="agent1")
        assert len(agent1_failures) == 2

    def test_get_failure_by_id(self) -> None:
        """Test getting failure by ID."""
        tracker = FailureTracker()

        record = tracker.record_failure(
            trace_id="t1",
            error_message="Test error",
        )

        found = tracker.get_failure_by_id(record.failure_id)
        assert found is not None
        assert found.failure_id == record.failure_id

        not_found = tracker.get_failure_by_id("nonexistent")
        assert not_found is None

    def test_get_failures_by_trace(self) -> None:
        """Test getting failures by trace ID."""
        tracker = FailureTracker()

        tracker.record_failure(
            trace_id="trace-123",
            error_message="Error 1",
        )
        tracker.record_failure(
            trace_id="trace-123",
            error_message="Error 2",
        )
        tracker.record_failure(
            trace_id="trace-456",
            error_message="Error 3",
        )

        trace_failures = tracker.get_failures_by_trace("trace-123")
        assert len(trace_failures) == 2

    def test_clear_tracker(self) -> None:
        """Test clearing the tracker."""
        tracker = FailureTracker()

        tracker.record_failure(trace_id="t1", error_message="Error")
        tracker.record_failure(trace_id="t2", error_message="Error")

        tracker.clear()

        assert len(tracker.get_failures()) == 0


# ============================================================================
# FailureAnalysis Tests
# ============================================================================


class TestFailureAnalysis:
    """Tests for FailureAnalysis dataclass."""

    def test_create_analysis(self) -> None:
        """Test creating a failure analysis."""
        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.LLM_ERROR,
            error_message="Test",
        )

        analysis = FailureAnalysis(
            failure=record,
            matched_patterns=[],
            root_cause="Test cause",
            suggested_fixes=["Fix 1", "Fix 2"],
            similar_failures=["f2", "f3"],
            impact_assessment="Low impact",
        )

        assert analysis.root_cause == "Test cause"
        assert len(analysis.suggested_fixes) == 2

    def test_to_dict(self) -> None:
        """Test converting analysis to dictionary."""
        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.TIMEOUT,
            error_message="Test",
        )

        analysis = FailureAnalysis(
            failure=record,
            similar_failures=["f2", "f3", "f4"],
        )

        d = analysis.to_dict()
        assert "failure" in d
        assert d["similar_failure_count"] == 3


# ============================================================================
# FailureAlertManager Tests
# ============================================================================


class TestFailureAlertManager:
    """Tests for FailureAlertManager class."""

    def test_create_manager(self) -> None:
        """Test creating an alert manager."""
        manager = FailureAlertManager()
        assert manager.thresholds is not None
        assert len(manager.alerts) == 0

    def test_create_manager_custom_thresholds(self) -> None:
        """Test creating manager with custom thresholds."""
        thresholds = FailureAlertThresholds(
            failures_per_minute=10,
            failures_per_hour=50,
        )
        manager = FailureAlertManager(thresholds=thresholds)
        assert manager.thresholds.failures_per_minute == 10

    def test_check_failure_critical(self) -> None:
        """Test alerting on critical failure."""
        manager = FailureAlertManager()

        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.RESOURCE,
            severity=FailureSeverity.CRITICAL,
            error_message="Out of memory",
        )

        alerts = manager.check_failure(record, [])
        assert len(alerts) >= 1

        # Should have a critical severity alert
        critical_alerts = [
            a for a in alerts if a.severity == FailureSeverity.CRITICAL
        ]
        assert len(critical_alerts) >= 1

    def test_check_failure_new_pattern(self) -> None:
        """Test alerting on new pattern."""
        manager = FailureAlertManager()

        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.LLM_ERROR,
            error_message="Rate limit",
        )

        pattern = FailurePattern(
            pattern_id="new_pattern",
            name="New Pattern",
        )

        alerts = manager.check_failure(record, [pattern])

        # Should have a new pattern alert
        new_pattern_alerts = [
            a for a in alerts if a.alert_type == FailureAlertType.NEW_PATTERN
        ]
        assert len(new_pattern_alerts) == 1

    def test_check_failure_spike(self) -> None:
        """Test alerting on failure spike."""
        thresholds = FailureAlertThresholds(failures_per_minute=3)
        manager = FailureAlertManager(thresholds=thresholds)

        # Record enough failures to trigger spike
        for i in range(5):
            record = FailureRecord(
                failure_id=f"f{i}",
                trace_id=f"t{i}",
                category=FailureCategory.LLM_ERROR,
                error_message="Error",
            )
            manager.check_failure(record, [])

        # Should have a spike alert
        spike_alerts = [
            a for a in manager.alerts if a.alert_type == FailureAlertType.SPIKE
        ]
        assert len(spike_alerts) >= 1

    def test_alert_callback(self) -> None:
        """Test alert callback is called."""
        alerts_received: list[FailureAlert] = []

        def callback(alert: FailureAlert) -> None:
            alerts_received.append(alert)

        manager = FailureAlertManager(on_alert=callback)

        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.RESOURCE,
            severity=FailureSeverity.CRITICAL,
            error_message="Critical error",
        )

        manager.check_failure(record, [])
        assert len(alerts_received) >= 1

    def test_clear_alerts(self) -> None:
        """Test clearing alerts."""
        manager = FailureAlertManager()

        record = FailureRecord(
            failure_id="f1",
            trace_id="t1",
            category=FailureCategory.RESOURCE,
            severity=FailureSeverity.CRITICAL,
            error_message="Error",
        )

        manager.check_failure(record, [])
        assert len(manager.alerts) >= 1

        manager.clear_alerts()
        assert len(manager.alerts) == 0


class TestFailureAlert:
    """Tests for FailureAlert dataclass."""

    def test_create_alert(self) -> None:
        """Test creating an alert."""
        alert = FailureAlert(
            alert_id="alert-1",
            alert_type=FailureAlertType.SPIKE,
            message="Failure spike detected",
            severity=FailureSeverity.HIGH,
            failure_ids=["f1", "f2", "f3"],
        )

        assert alert.alert_id == "alert-1"
        assert alert.alert_type == FailureAlertType.SPIKE
        assert len(alert.failure_ids) == 3

    def test_to_dict(self) -> None:
        """Test converting alert to dictionary."""
        alert = FailureAlert(
            alert_id="alert-1",
            alert_type=FailureAlertType.THRESHOLD,
            message="Threshold exceeded",
            severity=FailureSeverity.MEDIUM,
        )

        d = alert.to_dict()
        assert d["alert_type"] == "threshold"
        assert d["severity"] == "medium"
        assert "timestamp" in d


# ============================================================================
# FailureReport Tests
# ============================================================================


class TestFailureReport:
    """Tests for FailureReport dataclass."""

    def test_create_report(self) -> None:
        """Test creating a failure report."""
        now = datetime.now(UTC)
        report = FailureReport(
            period_start=now - timedelta(hours=24),
            period_end=now,
            total_failures=10,
            by_category={"llm_error": 5, "timeout": 3, "network": 2},
            recovery_rate=30.0,
        )

        assert report.total_failures == 10
        assert report.by_category["llm_error"] == 5
        assert report.recovery_rate == 30.0

    def test_to_dict(self) -> None:
        """Test converting report to dictionary."""
        now = datetime.now(UTC)
        report = FailureReport(
            period_start=now,
            period_end=now,
            total_failures=5,
            top_patterns=[("rate_limit", 3), ("timeout", 2)],
        )

        d = report.to_dict()
        assert d["total_failures"] == 5
        assert len(d["top_patterns"]) == 2


class TestGenerateFailureReport:
    """Tests for generate_failure_report function."""

    def test_generate_empty_report(self) -> None:
        """Test generating report with no failures."""
        report = generate_failure_report([])

        assert report.total_failures == 0
        assert report.by_category == {}

    def test_generate_report(self) -> None:
        """Test generating a failure report."""
        failures = [
            FailureRecord(
                failure_id="f1",
                trace_id="t1",
                category=FailureCategory.LLM_ERROR,
                severity=FailureSeverity.MEDIUM,
                agent_name="agent1",
                error_message="Error 1",
            ),
            FailureRecord(
                failure_id="f2",
                trace_id="t2",
                category=FailureCategory.LLM_ERROR,
                severity=FailureSeverity.HIGH,
                agent_name="agent1",
                error_message="Error 2",
            ),
            FailureRecord(
                failure_id="f3",
                trace_id="t3",
                category=FailureCategory.TIMEOUT,
                severity=FailureSeverity.LOW,
                agent_name="agent2",
                error_message="Error 3",
                recovered=True,
            ),
        ]

        report = generate_failure_report(failures)

        assert report.total_failures == 3
        assert report.by_category["llm_error"] == 2
        assert report.by_category["timeout"] == 1
        assert report.by_agent["agent1"] == 2
        assert report.by_agent["agent2"] == 1
        assert report.recovery_rate == pytest.approx(33.33, rel=0.1)

    def test_generate_report_with_alerts(self) -> None:
        """Test generating report with alert count."""
        failures = [
            FailureRecord(
                failure_id="f1",
                trace_id="t1",
                category=FailureCategory.LLM_ERROR,
                error_message="Error",
            )
        ]
        alerts = [
            FailureAlert(
                alert_id="a1",
                alert_type=FailureAlertType.SPIKE,
                message="Spike",
                severity=FailureSeverity.HIGH,
            )
        ]

        report = generate_failure_report(failures, alerts=alerts)
        assert report.alerts_count == 1


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstances:
    """Tests for global instance management."""

    def setup_method(self) -> None:
        """Reset global instances before each test."""
        reset_failure_tracking()

    def test_get_failure_tracker_singleton(self) -> None:
        """Test failure tracker is a singleton."""
        tracker1 = get_failure_tracker()
        tracker2 = get_failure_tracker()
        assert tracker1 is tracker2

    def test_get_failure_alert_manager_singleton(self) -> None:
        """Test alert manager is a singleton."""
        manager1 = get_failure_alert_manager()
        manager2 = get_failure_alert_manager()
        assert manager1 is manager2

    def test_reset_failure_tracking(self) -> None:
        """Test resetting global instances."""
        tracker1 = get_failure_tracker()
        reset_failure_tracking()
        tracker2 = get_failure_tracker()
        assert tracker1 is not tracker2


# ============================================================================
# Integration Tests
# ============================================================================


class TestFailureIntegration:
    """Integration tests for failure analysis."""

    def test_full_failure_workflow(self) -> None:
        """Test complete failure tracking workflow."""
        tracker = FailureTracker()
        alert_manager = FailureAlertManager()

        # Record failures
        for i in range(5):
            record = tracker.record_failure(
                trace_id=f"trace-{i}",
                error_message="Rate limit exceeded for API",
                agent_name="research_agent",
            )

            # Check for alerts
            patterns = tracker.pattern_detector.detect_patterns(record)
            alert_manager.check_failure(record, patterns)

        # Analyze the first failure
        failures = tracker.get_failures()
        analysis = tracker.analyze_failure(failures[0])

        # Verify analysis
        assert analysis.matched_patterns
        assert analysis.suggested_fixes
        assert len(analysis.similar_failures) >= 4

        # Generate report
        report = generate_failure_report(
            failures,
            pattern_detector=tracker.pattern_detector,
            alerts=alert_manager.alerts,
        )

        assert report.total_failures == 5
        assert report.by_category["llm_error"] == 5
        assert "rate_limit" in [p for p, _ in report.top_patterns]

    def test_mixed_failure_types(self) -> None:
        """Test handling mixed failure types."""
        tracker = FailureTracker()

        # Record different failure types
        tracker.record_failure(
            trace_id="t1",
            error_message="Rate limit exceeded",
            agent_name="agent1",
        )
        tracker.record_failure(
            trace_id="t2",
            error_message="Connection refused",
            agent_name="agent2",
        )
        tracker.record_failure(
            trace_id="t3",
            error_message="401 Unauthorized",
            tool_name="api_tool",
        )
        tracker.record_failure(
            trace_id="t4",
            error_message="Request timed out",
            recovered=True,
        )

        # Generate report
        failures = tracker.get_failures()
        report = generate_failure_report(failures)

        assert report.total_failures == 4
        assert len(report.by_category) >= 3
        assert report.recovery_rate == 25.0
