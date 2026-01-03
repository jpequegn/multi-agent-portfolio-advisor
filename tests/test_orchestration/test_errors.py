"""Tests for error handling and recovery module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestration.errors import (
    AGENT_FALLBACK_CONFIGS,
    AgentError,
    AgentFallbackConfig,
    ErrorRecoveryManager,
    FallbackType,
    PortfolioAdvisorError,
    RecoveryContext,
    RecoveryError,
    RetryConfig,
    RetryStrategy,
    StateError,
    ToolError,
    WorkflowTimeoutError,
    classify_error,
    execute_with_timeout,
    get_agent_fallback_config,
    get_recovery_manager,
    is_critical_error,
    reset_recovery_manager,
    with_retry,
)


# ============================================================================
# Exception Hierarchy Tests
# ============================================================================


class TestPortfolioAdvisorError:
    """Tests for base PortfolioAdvisorError."""

    def test_basic_creation(self) -> None:
        """Test basic error creation."""
        error = PortfolioAdvisorError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.recoverable is True
        assert error.details == {}
        assert error.trace_id is None

    def test_with_all_attributes(self) -> None:
        """Test error with all attributes."""
        error = PortfolioAdvisorError(
            "Test error",
            details={"key": "value"},
            recoverable=False,
            trace_id="trace-123",
        )
        assert error.details == {"key": "value"}
        assert error.recoverable is False
        assert error.trace_id == "trace-123"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        error = PortfolioAdvisorError(
            "Test error",
            details={"key": "value"},
            trace_id="trace-123",
        )
        result = error.to_dict()
        assert result["error_type"] == "PortfolioAdvisorError"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}
        assert result["trace_id"] == "trace-123"


class TestAgentError:
    """Tests for AgentError."""

    def test_creation_with_agent_name(self) -> None:
        """Test error with agent name."""
        error = AgentError(
            "Research failed",
            agent_name="research_agent",
            stage="data_fetch",
        )
        assert error.agent_name == "research_agent"
        assert error.stage == "data_fetch"

    def test_to_dict_includes_agent_info(self) -> None:
        """Test to_dict includes agent information."""
        error = AgentError(
            "Analysis failed",
            agent_name="analysis_agent",
            stage="risk_calculation",
        )
        result = error.to_dict()
        assert result["agent_name"] == "analysis_agent"
        assert result["stage"] == "risk_calculation"
        assert result["error_type"] == "AgentError"


class TestToolError:
    """Tests for ToolError."""

    def test_creation_with_tool_name(self) -> None:
        """Test error with tool name."""
        error = ToolError(
            "API call failed",
            tool_name="market_data",
            input_data={"symbol": "AAPL"},
        )
        assert error.tool_name == "market_data"
        assert error.input_data == {"symbol": "AAPL"}

    def test_to_dict_includes_tool_info(self) -> None:
        """Test to_dict includes tool information."""
        error = ToolError(
            "News fetch failed",
            tool_name="news_search",
        )
        result = error.to_dict()
        assert result["tool_name"] == "news_search"
        assert result["error_type"] == "ToolError"


class TestStateError:
    """Tests for StateError."""

    def test_creation_with_state_info(self) -> None:
        """Test error with state information."""
        error = StateError(
            "Invalid state",
            state_key="research",
            expected="dict",
            actual="None",
        )
        assert error.state_key == "research"
        assert error.expected == "dict"
        assert error.actual == "None"
        assert error.recoverable is False  # State errors are not recoverable by default

    def test_to_dict_includes_state_info(self) -> None:
        """Test to_dict includes state information."""
        error = StateError(
            "Missing key",
            state_key="analysis",
        )
        result = error.to_dict()
        assert result["state_key"] == "analysis"
        assert result["error_type"] == "StateError"


class TestWorkflowTimeoutError:
    """Tests for WorkflowTimeoutError."""

    def test_creation_with_timeout_info(self) -> None:
        """Test error with timeout information."""
        error = WorkflowTimeoutError(
            "Operation timed out",
            timeout_seconds=30.0,
            operation="market_data_fetch",
        )
        assert error.timeout_seconds == 30.0
        assert error.operation == "market_data_fetch"

    def test_to_dict_includes_timeout_info(self) -> None:
        """Test to_dict includes timeout information."""
        error = WorkflowTimeoutError(
            "Timeout",
            timeout_seconds=60.0,
            operation="analysis",
        )
        result = error.to_dict()
        assert result["timeout_seconds"] == 60.0
        assert result["operation"] == "analysis"


class TestRecoveryError:
    """Tests for RecoveryError."""

    def test_creation_with_original_error(self) -> None:
        """Test error with original error."""
        original = AgentError("Original", agent_name="test")
        error = RecoveryError(
            "Recovery failed",
            original_error=original,
            recovery_attempts=3,
        )
        assert error.original_error is original
        assert error.recovery_attempts == 3
        assert error.recoverable is False  # Recovery errors are never recoverable

    def test_to_dict_includes_recovery_info(self) -> None:
        """Test to_dict includes recovery information."""
        error = RecoveryError(
            "All retries exhausted",
            recovery_attempts=5,
        )
        result = error.to_dict()
        assert result["recovery_attempts"] == 5
        assert result["error_type"] == "RecoveryError"


# ============================================================================
# Retry Configuration Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.min_wait_seconds == 1.0
        assert config.max_wait_seconds == 30.0
        assert config.multiplier == 2.0

    def test_from_strategy_none(self) -> None:
        """Test creating config from NONE strategy."""
        config = RetryConfig.from_strategy(RetryStrategy.NONE)
        assert config.max_attempts == 1

    def test_from_strategy_quick(self) -> None:
        """Test creating config from QUICK strategy."""
        config = RetryConfig.from_strategy(RetryStrategy.QUICK)
        assert config.max_attempts == 2
        assert config.min_wait_seconds == 0.5
        assert config.max_wait_seconds == 5.0

    def test_from_strategy_standard(self) -> None:
        """Test creating config from STANDARD strategy."""
        config = RetryConfig.from_strategy(RetryStrategy.STANDARD)
        assert config.max_attempts == 3

    def test_from_strategy_persistent(self) -> None:
        """Test creating config from PERSISTENT strategy."""
        config = RetryConfig.from_strategy(RetryStrategy.PERSISTENT)
        assert config.max_attempts == 5
        assert config.max_wait_seconds == 60.0


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self) -> None:
        """Test successful call doesn't retry."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3))
        async def successful_fn() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_fn()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_agent_error(self) -> None:
        """Test retries on AgentError."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=3, min_wait_seconds=0.01, max_wait_seconds=0.1))
        async def failing_fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AgentError("Retry me", agent_name="test")
            return "success"

        result = await failing_fn()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries(self) -> None:
        """Test exhausts all retries and raises."""
        call_count = 0

        @with_retry(RetryConfig(max_attempts=2, min_wait_seconds=0.01, max_wait_seconds=0.1))
        async def always_failing() -> str:
            nonlocal call_count
            call_count += 1
            raise AgentError("Always fails", agent_name="test")

        with pytest.raises(AgentError):
            await always_failing()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_retry_strategy(self) -> None:
        """Test with RetryStrategy enum."""
        call_count = 0

        @with_retry(RetryStrategy.QUICK)
        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = await fn()
        assert result == "done"
        assert call_count == 1


# ============================================================================
# Fallback Configuration Tests
# ============================================================================


class TestAgentFallbackConfig:
    """Tests for AgentFallbackConfig."""

    def test_default_configs_exist(self) -> None:
        """Test default configs exist for known agents."""
        assert "research_agent" in AGENT_FALLBACK_CONFIGS
        assert "analysis_agent" in AGENT_FALLBACK_CONFIGS
        assert "recommendation_agent" in AGENT_FALLBACK_CONFIGS

    def test_get_agent_fallback_config_known(self) -> None:
        """Test getting config for known agent."""
        config = get_agent_fallback_config("research_agent")
        assert config.agent_name == "research_agent"
        assert config.fallback_type == FallbackType.CACHED_DATA

    def test_get_agent_fallback_config_unknown(self) -> None:
        """Test getting config for unknown agent returns default."""
        config = get_agent_fallback_config("unknown_agent")
        assert config.agent_name == "unknown_agent"
        assert config.fallback_type == FallbackType.ERROR


# ============================================================================
# Recovery Context Tests
# ============================================================================


class TestRecoveryContext:
    """Tests for RecoveryContext."""

    def test_creation(self) -> None:
        """Test context creation."""
        error = AgentError("Test", agent_name="test")
        context = RecoveryContext(original_error=error, agent_name="test_agent")
        assert context.original_error is error
        assert context.agent_name == "test_agent"
        assert context.attempt_count == 0
        assert context.can_retry is True

    def test_add_attempt(self) -> None:
        """Test adding recovery attempts."""
        error = AgentError("Test", agent_name="test")
        context = RecoveryContext(original_error=error, max_attempts=2)

        context.add_attempt("retry", success=False, error="Failed")
        assert context.attempt_count == 1
        assert context.can_retry is True

        context.add_attempt("fallback", success=True)
        assert context.attempt_count == 2
        assert context.can_retry is False

    def test_last_attempt(self) -> None:
        """Test getting last attempt."""
        error = AgentError("Test", agent_name="test")
        context = RecoveryContext(original_error=error)

        assert context.last_attempt is None

        context.add_attempt("retry", success=False)
        assert context.last_attempt is not None
        assert context.last_attempt.strategy == "retry"


# ============================================================================
# Error Recovery Manager Tests
# ============================================================================


class TestErrorRecoveryManager:
    """Tests for ErrorRecoveryManager."""

    def test_creation(self) -> None:
        """Test manager creation."""
        manager = ErrorRecoveryManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_recover_with_fallback_fn(self) -> None:
        """Test recovery with fallback function."""
        manager = ErrorRecoveryManager()
        error = AgentError("Failed", agent_name="research_agent")

        result = await manager.recover(
            error=error,
            agent_name="research_agent",
            fallback_fn=lambda: {"cached": True},
        )

        assert result == {"cached": True}

    @pytest.mark.asyncio
    async def test_recover_with_async_fallback(self) -> None:
        """Test recovery with async fallback function."""
        manager = ErrorRecoveryManager()
        error = AgentError("Failed", agent_name="research_agent")

        async def async_fallback() -> dict:
            return {"async_cached": True}

        result = await manager.recover(
            error=error,
            agent_name="research_agent",
            fallback_fn=async_fallback,
        )

        assert result == {"async_cached": True}

    @pytest.mark.asyncio
    async def test_recover_non_recoverable_error(self) -> None:
        """Test recovery fails for non-recoverable error."""
        manager = ErrorRecoveryManager()
        error = StateError("State error", recoverable=False)

        with pytest.raises(RecoveryError) as exc_info:
            await manager.recover(
                error=error,
                agent_name="test_agent",
                fallback_fn=lambda: "fallback",
            )

        assert "Non-recoverable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_recover_fallback_fails(self) -> None:
        """Test when fallback itself fails."""
        manager = ErrorRecoveryManager()
        error = AgentError("Failed", agent_name="test")

        def failing_fallback() -> None:
            raise ValueError("Fallback failed")

        with pytest.raises(RecoveryError) as exc_info:
            await manager.recover(
                error=error,
                agent_name="test",
                fallback_fn=failing_fallback,
            )

        assert "Fallback failed" in str(exc_info.value)


class TestGlobalRecoveryManager:
    """Tests for global recovery manager functions."""

    def setup_method(self) -> None:
        """Reset global manager before each test."""
        reset_recovery_manager()

    def test_get_recovery_manager_creates_instance(self) -> None:
        """Test get creates instance if none exists."""
        manager = get_recovery_manager()
        assert manager is not None
        assert isinstance(manager, ErrorRecoveryManager)

    def test_get_recovery_manager_returns_same_instance(self) -> None:
        """Test get returns same instance."""
        manager1 = get_recovery_manager()
        manager2 = get_recovery_manager()
        assert manager1 is manager2

    def test_reset_recovery_manager(self) -> None:
        """Test reset clears the manager."""
        manager1 = get_recovery_manager()
        reset_recovery_manager()
        manager2 = get_recovery_manager()
        assert manager1 is not manager2


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestExecuteWithTimeout:
    """Tests for execute_with_timeout."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self) -> None:
        """Test successful completion within timeout."""

        async def quick_fn() -> str:
            return "done"

        result = await execute_with_timeout(quick_fn(), timeout_seconds=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self) -> None:
        """Test raises WorkflowTimeoutError on timeout."""

        async def slow_fn() -> str:
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(WorkflowTimeoutError) as exc_info:
            await execute_with_timeout(
                slow_fn(),
                timeout_seconds=0.01,
                operation_name="slow_operation",
            )

        assert exc_info.value.timeout_seconds == 0.01
        assert exc_info.value.operation == "slow_operation"


class TestClassifyError:
    """Tests for classify_error."""

    def test_classify_agent_error(self) -> None:
        """Test classifying AgentError."""
        error = AgentError("Test", agent_name="test")
        category, recoverable = classify_error(error)
        assert category == "AgentError"
        assert recoverable is True

    def test_classify_state_error(self) -> None:
        """Test classifying StateError."""
        error = StateError("Test")
        category, recoverable = classify_error(error)
        assert category == "StateError"
        assert recoverable is False

    def test_classify_timeout_error(self) -> None:
        """Test classifying TimeoutError."""
        error = TimeoutError("Timeout")
        category, recoverable = classify_error(error)
        assert category == "timeout"
        assert recoverable is True

    def test_classify_connection_error(self) -> None:
        """Test classifying ConnectionError."""
        error = ConnectionError("Connection refused")
        category, recoverable = classify_error(error)
        assert category == "connection"
        assert recoverable is True

    def test_classify_unknown_error(self) -> None:
        """Test classifying unknown error."""
        error = RuntimeError("Unknown")
        category, recoverable = classify_error(error)
        assert category == "unknown"
        assert recoverable is False


class TestIsCriticalError:
    """Tests for is_critical_error."""

    def test_non_recoverable_is_critical(self) -> None:
        """Test non-recoverable errors are critical."""
        error = PortfolioAdvisorError("Error", recoverable=False)
        assert is_critical_error(error) is True

    def test_recoverable_is_not_critical(self) -> None:
        """Test recoverable errors are not critical."""
        error = AgentError("Error", agent_name="test", recoverable=True)
        assert is_critical_error(error) is False

    def test_state_error_is_critical(self) -> None:
        """Test StateError is always critical."""
        error = StateError("Error")
        assert is_critical_error(error) is True

    def test_recovery_error_is_critical(self) -> None:
        """Test RecoveryError is always critical."""
        error = RecoveryError("Error")
        assert is_critical_error(error) is True

    def test_critical_keyword_in_message(self) -> None:
        """Test error with 'critical' in message."""
        error = RuntimeError("CRITICAL: System failure")
        assert is_critical_error(error) is True

    def test_fatal_keyword_in_message(self) -> None:
        """Test error with 'fatal' in message."""
        error = RuntimeError("Fatal error occurred")
        assert is_critical_error(error) is True
