"""Failure analysis system for observability.

This module provides tools for analyzing and understanding failures
in the multi-agent system, including categorization, pattern detection,
root cause identification, and alerting.

Features:
- Automatic failure categorization
- Pattern detection for recurring failures
- Root cause identification helpers
- Failure reports and statistics
- Alerting for failure spikes
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Failure Categories
# ============================================================================


class FailureCategory(Enum):
    """Categories for classifying failures."""

    LLM_ERROR = "llm_error"  # Model refused, rate limit, etc.
    TOOL_ERROR = "tool_error"  # Tool execution failed
    STATE_ERROR = "state_error"  # Invalid state transition
    TIMEOUT = "timeout"  # Operation timed out
    VALIDATION = "validation"  # Input/output validation failed
    EXTERNAL_API = "external_api"  # External API failure
    AUTHENTICATION = "authentication"  # Auth/permission errors
    RESOURCE = "resource"  # Resource exhaustion (memory, etc.)
    NETWORK = "network"  # Network connectivity issues
    UNKNOWN = "unknown"  # Unclassified failure


class FailureSeverity(Enum):
    """Severity levels for failures."""

    LOW = "low"  # Minor issue, auto-recoverable
    MEDIUM = "medium"  # Significant issue, may need attention
    HIGH = "high"  # Critical issue, needs immediate attention
    CRITICAL = "critical"  # System-wide impact


# ============================================================================
# Failure Record
# ============================================================================


@dataclass
class FailureRecord:
    """Record of a single failure event.

    Attributes:
        failure_id: Unique identifier for this failure.
        trace_id: Associated trace identifier.
        category: Failure category.
        severity: Failure severity.
        agent_name: Agent that failed (if applicable).
        tool_name: Tool that failed (if applicable).
        error_message: Error message or description.
        error_type: Type/class of the error.
        stack_trace: Stack trace if available.
        context: Additional context about the failure.
        timestamp: When the failure occurred.
        recovered: Whether the system recovered from this failure.
        recovery_action: Action taken to recover (if any).
    """

    failure_id: str
    trace_id: str
    category: FailureCategory
    severity: FailureSeverity = FailureSeverity.MEDIUM
    agent_name: str | None = None
    tool_name: str | None = None
    error_message: str = ""
    error_type: str = ""
    stack_trace: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    recovered: bool = False
    recovery_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_id": self.failure_id,
            "trace_id": self.trace_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "recovered": self.recovered,
            "recovery_action": self.recovery_action,
        }

    def matches_pattern(self, pattern: "FailurePattern") -> bool:
        """Check if this failure matches a pattern.

        Args:
            pattern: Pattern to match against.

        Returns:
            True if the failure matches the pattern.
        """
        if pattern.category and self.category != pattern.category:
            return False
        if pattern.agent_name and self.agent_name != pattern.agent_name:
            return False
        if pattern.tool_name and self.tool_name != pattern.tool_name:
            return False
        if pattern.error_type and self.error_type != pattern.error_type:
            return False
        return not (
            pattern.error_message_contains
            and pattern.error_message_contains.lower() not in self.error_message.lower()
        )


# ============================================================================
# Failure Patterns
# ============================================================================


@dataclass
class FailurePattern:
    """Pattern for detecting similar failures.

    Attributes:
        pattern_id: Unique identifier for this pattern.
        name: Human-readable name for the pattern.
        description: Description of what this pattern represents.
        category: Category to match (or None for any).
        agent_name: Agent name to match (or None for any).
        tool_name: Tool name to match (or None for any).
        error_type: Error type to match (or None for any).
        error_message_contains: Substring to find in error message.
        suggested_fix: Suggested fix for this pattern.
        severity_override: Override severity for matching failures.
    """

    pattern_id: str
    name: str
    description: str = ""
    category: FailureCategory | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    error_type: str | None = None
    error_message_contains: str | None = None
    suggested_fix: str | None = None
    severity_override: FailureSeverity | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value if self.category else None,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "error_type": self.error_type,
            "error_message_contains": self.error_message_contains,
            "suggested_fix": self.suggested_fix,
            "severity_override": (self.severity_override.value if self.severity_override else None),
        }


# Standard failure patterns
STANDARD_PATTERNS = [
    FailurePattern(
        pattern_id="rate_limit",
        name="Rate Limit Exceeded",
        description="LLM API rate limit was exceeded",
        category=FailureCategory.LLM_ERROR,
        error_message_contains="rate limit",
        suggested_fix="Implement exponential backoff or reduce request frequency",
        severity_override=FailureSeverity.MEDIUM,
    ),
    FailurePattern(
        pattern_id="context_length",
        name="Context Length Exceeded",
        description="Input exceeded model's context window",
        category=FailureCategory.LLM_ERROR,
        error_message_contains="context length",
        suggested_fix="Reduce input size or use summarization",
        severity_override=FailureSeverity.MEDIUM,
    ),
    FailurePattern(
        pattern_id="api_timeout",
        name="API Timeout",
        description="External API call timed out",
        category=FailureCategory.TIMEOUT,
        error_message_contains="time",  # Matches "timeout", "timed out", etc.
        suggested_fix="Increase timeout or add retry logic",
        severity_override=FailureSeverity.LOW,
    ),
    FailurePattern(
        pattern_id="invalid_json",
        name="Invalid JSON Response",
        description="Failed to parse JSON from LLM response",
        category=FailureCategory.VALIDATION,
        error_message_contains="json",
        suggested_fix="Add JSON parsing retry with structured prompts",
        severity_override=FailureSeverity.MEDIUM,
    ),
    FailurePattern(
        pattern_id="auth_failure",
        name="Authentication Failure",
        description="API authentication failed",
        category=FailureCategory.AUTHENTICATION,
        error_message_contains="401",
        suggested_fix="Check API keys and credentials",
        severity_override=FailureSeverity.HIGH,
    ),
    FailurePattern(
        pattern_id="connection_error",
        name="Connection Error",
        description="Failed to connect to external service",
        category=FailureCategory.NETWORK,
        error_message_contains="connection",
        suggested_fix="Check network connectivity and service availability",
        severity_override=FailureSeverity.MEDIUM,
    ),
    FailurePattern(
        pattern_id="tool_not_found",
        name="Tool Not Found",
        description="Agent tried to use a non-existent tool",
        category=FailureCategory.TOOL_ERROR,
        error_message_contains="tool not found",
        suggested_fix="Verify tool registration and agent configuration",
        severity_override=FailureSeverity.HIGH,
    ),
    FailurePattern(
        pattern_id="state_transition",
        name="Invalid State Transition",
        description="Agent attempted invalid state transition",
        category=FailureCategory.STATE_ERROR,
        error_message_contains="state",
        suggested_fix="Review state machine logic and transitions",
        severity_override=FailureSeverity.HIGH,
    ),
]


# ============================================================================
# Failure Analysis
# ============================================================================


@dataclass
class FailureAnalysis:
    """Analysis result for a failure.

    Attributes:
        failure: The failure record being analyzed.
        matched_patterns: Patterns that matched this failure.
        root_cause: Identified root cause.
        suggested_fixes: Suggested fixes for the failure.
        similar_failures: IDs of similar failures.
        impact_assessment: Assessment of failure impact.
    """

    failure: FailureRecord
    matched_patterns: list[FailurePattern] = field(default_factory=list)
    root_cause: str | None = None
    suggested_fixes: list[str] = field(default_factory=list)
    similar_failures: list[str] = field(default_factory=list)
    impact_assessment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure": self.failure.to_dict(),
            "matched_patterns": [p.to_dict() for p in self.matched_patterns],
            "root_cause": self.root_cause,
            "suggested_fixes": self.suggested_fixes,
            "similar_failure_count": len(self.similar_failures),
            "similar_failures": self.similar_failures[:10],  # Limit to 10
            "impact_assessment": self.impact_assessment,
        }


# ============================================================================
# Failure Categorizer
# ============================================================================


class FailureCategorizer:
    """Categorizes failures based on error information.

    Uses error messages, types, and context to automatically
    assign appropriate failure categories.
    """

    # Keywords for categorization (order matters - more specific first)
    # Use list of tuples to maintain order
    CATEGORY_KEYWORDS: list[tuple[FailureCategory, list[str]]] = [
        # Authentication should be checked before validation (401/403 before "invalid")
        (
            FailureCategory.AUTHENTICATION,
            [
                "auth",
                "401",
                "403",
                "unauthorized",
                "forbidden",
                "permission",
                "credential",
            ],
        ),
        # Timeout before other errors
        (
            FailureCategory.TIMEOUT,
            [
                "timeout",
                "timed out",
                "deadline",
                "expired",
            ],
        ),
        # LLM errors
        (
            FailureCategory.LLM_ERROR,
            [
                "rate limit",
                "context length",
                "model",
                "completion",
                "anthropic",
                "openai",
                "claude",
                "gpt",
                "token",
                "prompt",
            ],
        ),
        # Tool errors
        (
            FailureCategory.TOOL_ERROR,
            [
                "tool",
                "function",
                "execute",
                "invocation",
                "call failed",
            ],
        ),
        # State errors
        (
            FailureCategory.STATE_ERROR,
            [
                "state",
                "transition",
                "invalid state",
                "workflow",
            ],
        ),
        # Validation (after auth since "invalid" is generic)
        (
            FailureCategory.VALIDATION,
            [
                "validation",
                "schema",
                "parse",
                "json",
                "format",
            ],
        ),
        # External API
        (
            FailureCategory.EXTERNAL_API,
            [
                "http",
                "request failed",
                "response error",
                "502",
                "503",
                "504",
            ],
        ),
        # Resource
        (
            FailureCategory.RESOURCE,
            [
                "memory",
                "disk",
                "resource",
                "exhausted",
                "out of",
                "limit exceeded",
            ],
        ),
        # Network
        (
            FailureCategory.NETWORK,
            [
                "connection",
                "network",
                "dns",
                "socket",
                "refused",
                "unreachable",
            ],
        ),
    ]

    def categorize(
        self,
        error_message: str,
        error_type: str = "",
        context: dict[str, Any] | None = None,
    ) -> FailureCategory:
        """Categorize a failure based on error information.

        Args:
            error_message: The error message.
            error_type: The error type/class name.
            context: Additional context.

        Returns:
            The failure category.
        """
        combined_text = f"{error_message} {error_type}".lower()

        # Check each category's keywords (order matters)
        for category, keywords in self.CATEGORY_KEYWORDS:
            for keyword in keywords:
                if keyword in combined_text:
                    return category

        # Check context for hints
        if context:
            if context.get("agent_name"):
                return FailureCategory.STATE_ERROR
            if context.get("tool_name"):
                return FailureCategory.TOOL_ERROR

        return FailureCategory.UNKNOWN

    def assess_severity(
        self,
        category: FailureCategory,
        _error_message: str,
        recovered: bool = False,
    ) -> FailureSeverity:
        """Assess the severity of a failure.

        Args:
            category: The failure category.
            error_message: The error message.
            recovered: Whether the system recovered.

        Returns:
            The failure severity.
        """
        # If recovered, lower severity
        if recovered:
            return FailureSeverity.LOW

        # Category-based severity
        severity_map = {
            FailureCategory.AUTHENTICATION: FailureSeverity.HIGH,
            FailureCategory.STATE_ERROR: FailureSeverity.HIGH,
            FailureCategory.RESOURCE: FailureSeverity.CRITICAL,
            FailureCategory.LLM_ERROR: FailureSeverity.MEDIUM,
            FailureCategory.TOOL_ERROR: FailureSeverity.MEDIUM,
            FailureCategory.EXTERNAL_API: FailureSeverity.MEDIUM,
            FailureCategory.TIMEOUT: FailureSeverity.LOW,
            FailureCategory.VALIDATION: FailureSeverity.LOW,
            FailureCategory.NETWORK: FailureSeverity.MEDIUM,
            FailureCategory.UNKNOWN: FailureSeverity.MEDIUM,
        }

        return severity_map.get(category, FailureSeverity.MEDIUM)


# ============================================================================
# Pattern Detector
# ============================================================================


class PatternDetector:
    """Detects failure patterns and groups similar failures.

    Identifies recurring failures and systemic issues based
    on pattern matching and frequency analysis.
    """

    def __init__(
        self,
        patterns: list[FailurePattern] | None = None,
    ) -> None:
        """Initialize the pattern detector.

        Args:
            patterns: Custom patterns to use (defaults to STANDARD_PATTERNS).
        """
        self.patterns = STANDARD_PATTERNS.copy() if patterns is None else patterns
        self._pattern_counts: Counter[str] = Counter()

    def add_pattern(self, pattern: FailurePattern) -> None:
        """Add a custom pattern.

        Args:
            pattern: Pattern to add.
        """
        self.patterns.append(pattern)

    def detect_patterns(
        self,
        failure: FailureRecord,
    ) -> list[FailurePattern]:
        """Detect patterns matching a failure.

        Args:
            failure: Failure to match.

        Returns:
            List of matching patterns.
        """
        matches = []
        for pattern in self.patterns:
            if failure.matches_pattern(pattern):
                matches.append(pattern)
                self._pattern_counts[pattern.pattern_id] += 1

        return matches

    def get_suggested_fixes(
        self,
        patterns: list[FailurePattern],
    ) -> list[str]:
        """Get suggested fixes from matching patterns.

        Args:
            patterns: Matching patterns.

        Returns:
            List of suggested fixes.
        """
        fixes = []
        for pattern in patterns:
            if pattern.suggested_fix and pattern.suggested_fix not in fixes:
                fixes.append(pattern.suggested_fix)
        return fixes

    def get_pattern_frequency(self) -> dict[str, int]:
        """Get frequency of pattern matches.

        Returns:
            Dictionary of pattern ID to match count.
        """
        return dict(self._pattern_counts)

    def get_top_patterns(self, n: int = 5) -> list[tuple[str, int]]:
        """Get the most frequently matched patterns.

        Args:
            n: Number of patterns to return.

        Returns:
            List of (pattern_id, count) tuples.
        """
        return self._pattern_counts.most_common(n)

    def reset_counts(self) -> None:
        """Reset pattern match counts."""
        self._pattern_counts.clear()


# ============================================================================
# Failure Tracker
# ============================================================================


class FailureTracker:
    """Tracks and manages failure records.

    Stores failures, provides querying capabilities, and
    enables failure analysis.
    """

    def __init__(
        self,
        categorizer: FailureCategorizer | None = None,
        pattern_detector: PatternDetector | None = None,
    ) -> None:
        """Initialize the failure tracker.

        Args:
            categorizer: Failure categorizer to use.
            pattern_detector: Pattern detector to use.
        """
        self.categorizer = categorizer or FailureCategorizer()
        self.pattern_detector = pattern_detector or PatternDetector()
        self._failures: list[FailureRecord] = []
        self._failure_counter = 0

    def record_failure(
        self,
        trace_id: str,
        error_message: str,
        *,
        error_type: str = "",
        agent_name: str | None = None,
        tool_name: str | None = None,
        stack_trace: str | None = None,
        context: dict[str, Any] | None = None,
        recovered: bool = False,
        recovery_action: str | None = None,
    ) -> FailureRecord:
        """Record a new failure.

        Args:
            trace_id: Associated trace ID.
            error_message: Error message.
            error_type: Error type/class name.
            agent_name: Agent that failed.
            tool_name: Tool that failed.
            stack_trace: Stack trace if available.
            context: Additional context.
            recovered: Whether the system recovered.
            recovery_action: Action taken to recover.

        Returns:
            The created failure record.
        """
        self._failure_counter += 1
        failure_id = f"failure-{self._failure_counter}"

        # Auto-categorize
        category = self.categorizer.categorize(
            error_message,
            error_type,
            context or {},
        )
        severity = self.categorizer.assess_severity(
            category,
            error_message,
            recovered,
        )

        failure = FailureRecord(
            failure_id=failure_id,
            trace_id=trace_id,
            category=category,
            severity=severity,
            agent_name=agent_name,
            tool_name=tool_name,
            error_message=error_message,
            error_type=error_type,
            stack_trace=stack_trace,
            context=context or {},
            recovered=recovered,
            recovery_action=recovery_action,
        )

        self._failures.append(failure)

        logger.info(
            "failure_recorded",
            failure_id=failure_id,
            category=category.value,
            severity=severity.value,
            agent=agent_name,
            tool=tool_name,
        )

        return failure

    def analyze_failure(
        self,
        failure: FailureRecord,
    ) -> FailureAnalysis:
        """Analyze a failure record.

        Args:
            failure: Failure to analyze.

        Returns:
            Failure analysis result.
        """
        # Detect matching patterns
        patterns = self.pattern_detector.detect_patterns(failure)

        # Get suggested fixes
        fixes = self.pattern_detector.get_suggested_fixes(patterns)

        # Find similar failures
        similar = self.find_similar_failures(failure)

        # Determine root cause
        root_cause = self._identify_root_cause(failure, patterns)

        # Assess impact
        impact = self._assess_impact(failure, similar)

        return FailureAnalysis(
            failure=failure,
            matched_patterns=patterns,
            root_cause=root_cause,
            suggested_fixes=fixes,
            similar_failures=[f.failure_id for f in similar],
            impact_assessment=impact,
        )

    def find_similar_failures(
        self,
        failure: FailureRecord,
        *,
        max_results: int = 10,
        time_window: timedelta | None = None,
    ) -> list[FailureRecord]:
        """Find failures similar to the given one.

        Args:
            failure: Failure to find similar to.
            max_results: Maximum number of results.
            time_window: Only consider failures within this window.

        Returns:
            List of similar failures.
        """
        similar = []
        cutoff = None
        if time_window:
            cutoff = datetime.now(UTC) - time_window

        for f in self._failures:
            if f.failure_id == failure.failure_id:
                continue
            if cutoff and f.timestamp < cutoff:
                continue

            # Check similarity
            score = self._similarity_score(failure, f)
            if score > 0.5:  # At least 50% similar
                similar.append((score, f))

        # Sort by similarity and return top results
        similar.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in similar[:max_results]]

    def get_failures(
        self,
        *,
        category: FailureCategory | None = None,
        severity: FailureSeverity | None = None,
        agent_name: str | None = None,
        tool_name: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[FailureRecord]:
        """Query failures with filters.

        Args:
            category: Filter by category.
            severity: Filter by severity.
            agent_name: Filter by agent name.
            tool_name: Filter by tool name.
            since: Filter by timestamp.
            limit: Maximum number of results.

        Returns:
            List of matching failures.
        """
        results = []

        for f in self._failures:
            if category and f.category != category:
                continue
            if severity and f.severity != severity:
                continue
            if agent_name and f.agent_name != agent_name:
                continue
            if tool_name and f.tool_name != tool_name:
                continue
            if since and f.timestamp < since:
                continue
            results.append(f)

        if limit:
            results = results[-limit:]

        return results

    def get_failure_by_id(self, failure_id: str) -> FailureRecord | None:
        """Get a failure by ID.

        Args:
            failure_id: Failure ID to find.

        Returns:
            Failure record or None.
        """
        for f in self._failures:
            if f.failure_id == failure_id:
                return f
        return None

    def get_failures_by_trace(self, trace_id: str) -> list[FailureRecord]:
        """Get all failures for a trace.

        Args:
            trace_id: Trace ID to find failures for.

        Returns:
            List of failures for the trace.
        """
        return [f for f in self._failures if f.trace_id == trace_id]

    def _similarity_score(
        self,
        f1: FailureRecord,
        f2: FailureRecord,
    ) -> float:
        """Calculate similarity score between two failures.

        Args:
            f1: First failure.
            f2: Second failure.

        Returns:
            Similarity score (0-1).
        """
        score = 0.0
        checks = 0

        # Category match
        checks += 1
        if f1.category == f2.category:
            score += 1

        # Agent match
        if f1.agent_name or f2.agent_name:
            checks += 1
            if f1.agent_name == f2.agent_name:
                score += 1

        # Tool match
        if f1.tool_name or f2.tool_name:
            checks += 1
            if f1.tool_name == f2.tool_name:
                score += 1

        # Error type match
        if f1.error_type or f2.error_type:
            checks += 1
            if f1.error_type == f2.error_type:
                score += 1

        return score / checks if checks > 0 else 0.0

    def _identify_root_cause(
        self,
        failure: FailureRecord,
        patterns: list[FailurePattern],
    ) -> str:
        """Identify the root cause of a failure.

        Args:
            failure: Failure to analyze.
            patterns: Matched patterns.

        Returns:
            Root cause description.
        """
        # Use pattern descriptions if available
        if patterns:
            return patterns[0].description

        # Generate based on category
        cause_map = {
            FailureCategory.LLM_ERROR: "LLM service error or configuration issue",
            FailureCategory.TOOL_ERROR: "Tool execution or configuration error",
            FailureCategory.STATE_ERROR: "Invalid workflow state or transition",
            FailureCategory.TIMEOUT: "Operation exceeded time limit",
            FailureCategory.VALIDATION: "Input or output validation failure",
            FailureCategory.EXTERNAL_API: "External API service issue",
            FailureCategory.AUTHENTICATION: "Authentication or authorization failure",
            FailureCategory.RESOURCE: "System resource exhaustion",
            FailureCategory.NETWORK: "Network connectivity issue",
            FailureCategory.UNKNOWN: "Unknown root cause - requires investigation",
        }

        return cause_map.get(failure.category, "Unknown")

    def _assess_impact(
        self,
        failure: FailureRecord,
        similar: list[FailureRecord],
    ) -> str:
        """Assess the impact of a failure.

        Args:
            failure: Failure to assess.
            similar: Similar failures found.

        Returns:
            Impact assessment description.
        """
        # Check for recurring pattern
        if len(similar) >= 5:
            return f"Systemic issue: {len(similar)} similar failures detected"

        # Severity-based assessment
        if failure.severity == FailureSeverity.CRITICAL:
            return "Critical failure requiring immediate attention"
        elif failure.severity == FailureSeverity.HIGH:
            return "High-impact failure affecting system reliability"
        elif failure.severity == FailureSeverity.MEDIUM:
            return "Moderate impact on user experience"
        else:
            return "Low impact, isolated incident"

    def clear(self) -> None:
        """Clear all failure records."""
        self._failures.clear()
        self._failure_counter = 0
        self.pattern_detector.reset_counts()


# ============================================================================
# Failure Alerts
# ============================================================================


class FailureAlertType(Enum):
    """Types of failure alerts."""

    SPIKE = "spike"  # Sudden increase in failures
    THRESHOLD = "threshold"  # Exceeded failure count threshold
    NEW_PATTERN = "new_pattern"  # New failure pattern detected
    RECURRING = "recurring"  # Recurring failure pattern


@dataclass
class FailureAlert:
    """A failure-related alert.

    Attributes:
        alert_id: Unique alert identifier.
        alert_type: Type of alert.
        message: Alert message.
        severity: Alert severity.
        failure_ids: Related failure IDs.
        pattern_id: Related pattern ID (if applicable).
        context: Additional context.
        timestamp: When the alert was triggered.
    """

    alert_id: str
    alert_type: FailureAlertType
    message: str
    severity: FailureSeverity
    failure_ids: list[str] = field(default_factory=list)
    pattern_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "message": self.message,
            "severity": self.severity.value,
            "failure_ids": self.failure_ids,
            "pattern_id": self.pattern_id,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FailureAlertThresholds:
    """Thresholds for failure alerts.

    Attributes:
        failures_per_minute: Failures per minute for spike alert.
        failures_per_hour: Failures per hour for threshold alert.
        recurring_count: Count for recurring pattern alert.
        critical_failures: Number of critical failures for alert.
    """

    failures_per_minute: int = 5
    failures_per_hour: int = 20
    recurring_count: int = 3
    critical_failures: int = 1


class FailureAlertManager:
    """Manages failure alerts.

    Monitors failure patterns and generates alerts when
    thresholds are exceeded or patterns are detected.
    """

    def __init__(
        self,
        thresholds: FailureAlertThresholds | None = None,
        on_alert: Callable[[FailureAlert], None] | None = None,
    ) -> None:
        """Initialize the alert manager.

        Args:
            thresholds: Alert thresholds.
            on_alert: Callback for alerts.
        """
        self.thresholds = thresholds or FailureAlertThresholds()
        self.on_alert = on_alert
        self._alerts: list[FailureAlert] = []
        self._alert_counter = 0
        self._recent_failures: list[tuple[datetime, str]] = []
        self._seen_patterns: set[str] = set()

    @property
    def alerts(self) -> list[FailureAlert]:
        """Get all triggered alerts."""
        return self._alerts.copy()

    def check_failure(
        self,
        failure: FailureRecord,
        patterns: list[FailurePattern],
    ) -> list[FailureAlert]:
        """Check a failure for alert conditions.

        Args:
            failure: Failure to check.
            patterns: Patterns matched by the failure.

        Returns:
            List of triggered alerts.
        """
        alerts = []
        now = datetime.now(UTC)

        # Track recent failure
        self._recent_failures.append((now, failure.failure_id))
        self._cleanup_recent_failures()

        # Check for spike
        spike_alert = self._check_spike(failure)
        if spike_alert:
            alerts.append(spike_alert)

        # Check for threshold
        threshold_alert = self._check_threshold(failure)
        if threshold_alert:
            alerts.append(threshold_alert)

        # Check for critical failure
        if failure.severity == FailureSeverity.CRITICAL:
            alert = self._create_alert(
                FailureAlertType.THRESHOLD,
                f"Critical failure: {failure.error_message[:100]}",
                FailureSeverity.CRITICAL,
                [failure.failure_id],
            )
            alerts.append(alert)

        # Check for new pattern
        for pattern in patterns:
            if pattern.pattern_id not in self._seen_patterns:
                self._seen_patterns.add(pattern.pattern_id)
                alert = self._create_alert(
                    FailureAlertType.NEW_PATTERN,
                    f"New failure pattern detected: {pattern.name}",
                    FailureSeverity.MEDIUM,
                    [failure.failure_id],
                    pattern_id=pattern.pattern_id,
                )
                alerts.append(alert)

        # Store and notify
        for alert in alerts:
            self._alerts.append(alert)
            if self.on_alert:
                self.on_alert(alert)
            logger.warning(
                "failure_alert_triggered",
                alert_type=alert.alert_type.value,
                message=alert.message,
            )

        return alerts

    def _check_spike(self, _failure: FailureRecord) -> FailureAlert | None:
        """Check for failure spike condition."""
        minute_ago = datetime.now(UTC) - timedelta(minutes=1)
        recent_count = sum(1 for ts, _ in self._recent_failures if ts >= minute_ago)

        if recent_count >= self.thresholds.failures_per_minute:
            failure_ids = [fid for ts, fid in self._recent_failures if ts >= minute_ago]
            return self._create_alert(
                FailureAlertType.SPIKE,
                f"Failure spike: {recent_count} failures in the last minute",
                FailureSeverity.HIGH,
                failure_ids,
            )
        return None

    def _check_threshold(self, _failure: FailureRecord) -> FailureAlert | None:
        """Check for threshold condition."""
        hour_ago = datetime.now(UTC) - timedelta(hours=1)
        hourly_count = sum(1 for ts, _ in self._recent_failures if ts >= hour_ago)

        if hourly_count >= self.thresholds.failures_per_hour:
            # Only alert once per hour
            recent_threshold_alerts = [
                a
                for a in self._alerts
                if a.alert_type == FailureAlertType.THRESHOLD and a.timestamp >= hour_ago
            ]
            if not recent_threshold_alerts:
                failure_ids = [fid for ts, fid in self._recent_failures if ts >= hour_ago]
                return self._create_alert(
                    FailureAlertType.THRESHOLD,
                    f"Failure threshold exceeded: {hourly_count} failures in the last hour",
                    FailureSeverity.HIGH,
                    failure_ids,
                )
        return None

    def _create_alert(
        self,
        alert_type: FailureAlertType,
        message: str,
        severity: FailureSeverity,
        failure_ids: list[str],
        *,
        pattern_id: str | None = None,
    ) -> FailureAlert:
        """Create a new alert."""
        self._alert_counter += 1
        return FailureAlert(
            alert_id=f"alert-{self._alert_counter}",
            alert_type=alert_type,
            message=message,
            severity=severity,
            failure_ids=failure_ids,
            pattern_id=pattern_id,
        )

    def _cleanup_recent_failures(self) -> None:
        """Clean up old failure records."""
        cutoff = datetime.now(UTC) - timedelta(hours=2)
        self._recent_failures = [(ts, fid) for ts, fid in self._recent_failures if ts >= cutoff]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()


# ============================================================================
# Failure Report
# ============================================================================


@dataclass
class FailureReport:
    """Aggregated failure report.

    Attributes:
        period_start: Start of the reporting period.
        period_end: End of the reporting period.
        total_failures: Total failure count.
        by_category: Failures by category.
        by_severity: Failures by severity.
        by_agent: Failures by agent.
        by_tool: Failures by tool.
        top_patterns: Most common failure patterns.
        recovery_rate: Percentage of failures that recovered.
        alerts_count: Number of alerts triggered.
    """

    period_start: datetime
    period_end: datetime
    total_failures: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)
    by_agent: dict[str, int] = field(default_factory=dict)
    by_tool: dict[str, int] = field(default_factory=dict)
    top_patterns: list[tuple[str, int]] = field(default_factory=list)
    recovery_rate: float = 0.0
    alerts_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_failures": self.total_failures,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "by_agent": self.by_agent,
            "by_tool": self.by_tool,
            "top_patterns": [{"pattern": p, "count": c} for p, c in self.top_patterns],
            "recovery_rate": self.recovery_rate,
            "alerts_count": self.alerts_count,
        }


def generate_failure_report(
    failures: list[FailureRecord],
    *,
    period_start: datetime | None = None,
    period_end: datetime | None = None,
    pattern_detector: PatternDetector | None = None,
    alerts: list[FailureAlert] | None = None,
) -> FailureReport:
    """Generate a failure report from failure records.

    Args:
        failures: List of failure records.
        period_start: Start of reporting period.
        period_end: End of reporting period.
        pattern_detector: Pattern detector for pattern stats.
        alerts: Alerts to include in count.

    Returns:
        Aggregated failure report.
    """
    now = datetime.now(UTC)

    if not failures:
        return FailureReport(
            period_start=period_start or now,
            period_end=period_end or now,
            alerts_count=len(alerts) if alerts else 0,
        )

    # Determine period
    if period_start is None:
        period_start = min(f.timestamp for f in failures)
    if period_end is None:
        period_end = max(f.timestamp for f in failures)

    # Aggregate data
    by_category: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    by_agent: Counter[str] = Counter()
    by_tool: Counter[str] = Counter()
    recovered_count = 0

    for f in failures:
        by_category[f.category.value] += 1
        by_severity[f.severity.value] += 1
        if f.agent_name:
            by_agent[f.agent_name] += 1
        if f.tool_name:
            by_tool[f.tool_name] += 1
        if f.recovered:
            recovered_count += 1

    # Get top patterns
    top_patterns = []
    if pattern_detector:
        top_patterns = pattern_detector.get_top_patterns(5)

    # Calculate recovery rate
    recovery_rate = (recovered_count / len(failures) * 100) if failures else 0.0

    return FailureReport(
        period_start=period_start,
        period_end=period_end,
        total_failures=len(failures),
        by_category=dict(by_category),
        by_severity=dict(by_severity),
        by_agent=dict(by_agent),
        by_tool=dict(by_tool),
        top_patterns=top_patterns,
        recovery_rate=recovery_rate,
        alerts_count=len(alerts) if alerts else 0,
    )


# ============================================================================
# Global Instance
# ============================================================================

_global_tracker: FailureTracker | None = None
_global_alert_manager: FailureAlertManager | None = None


def get_failure_tracker() -> FailureTracker:
    """Get the global failure tracker instance.

    Returns:
        Global FailureTracker instance.
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = FailureTracker()
    return _global_tracker


def get_failure_alert_manager() -> FailureAlertManager:
    """Get the global failure alert manager instance.

    Returns:
        Global FailureAlertManager instance.
    """
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = FailureAlertManager()
    return _global_alert_manager


def reset_failure_tracking() -> None:
    """Reset global failure tracking instances."""
    global _global_tracker, _global_alert_manager
    _global_tracker = None
    _global_alert_manager = None
