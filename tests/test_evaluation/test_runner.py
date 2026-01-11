"""Tests for the evaluation runner."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentState, BaseAgent
from src.evaluation.models import (
    ABTestConfig,
    DimensionScore,
    EvaluationCase,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
)
from src.evaluation.runner import EvaluationRunner, create_run_id


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, response: dict | None = None, error: Exception | None = None) -> None:
        """Initialize mock agent."""
        super().__init__(llm=MagicMock())
        self._response = response or {"result": "mock output"}
        self._error = error

    @property
    def name(self) -> str:
        return "mock_agent"

    @property
    def description(self) -> str:
        return "Mock agent for testing"

    @property
    def system_prompt(self) -> str:
        return "You are a mock agent."

    async def invoke(self, state: AgentState) -> AgentState:
        if self._error:
            raise self._error

        state.context.update(self._response)
        return state


class TestCreateRunId:
    """Tests for run ID generation."""

    def test_format(self) -> None:
        """Test run ID format."""
        run_id = create_run_id()
        assert run_id.startswith("eval-")
        assert len(run_id) == 17  # "eval-" + 12 hex chars

    def test_uniqueness(self) -> None:
        """Test that run IDs are unique."""
        ids = {create_run_id() for _ in range(100)}
        assert len(ids) == 100


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    def test_init(self) -> None:
        """Test runner initialization."""
        runner = EvaluationRunner()
        assert runner.llm is not None
        assert runner.agent_factory is None

    def test_init_with_factory(self) -> None:
        """Test runner initialization with factory."""

        def factory() -> MockAgent:
            return MockAgent()

        runner = EvaluationRunner(agent_factory=factory)
        assert runner.agent_factory is not None

    @pytest.mark.asyncio
    async def test_run_no_factory_raises(self) -> None:
        """Test that running without factory raises error."""
        runner = EvaluationRunner()
        cases = [EvaluationCase(id="test", name="Test", input={})]
        config = EvaluationRunConfig(run_id="test-run")

        with pytest.raises(ValueError) as exc_info:
            await runner.run_evaluation(cases, config)
        assert "No agent factory" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_single_case(self) -> None:
        """Test running a single evaluation case."""
        mock_response = {"recommendation": "buy", "confidence": 0.8}

        def agent_factory() -> MockAgent:
            return MockAgent(response=mock_response)

        # Mock the LLM judge
        with patch.object(
            EvaluationRunner,
            "_evaluate_single_case",
            new_callable=AsyncMock,
        ) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                case_id="case-1",
                run_id="run-1",
                model="test-model",
                scores=[
                    DimensionScore(
                        dimension=EvaluationDimension.ACCURACY,
                        score=0.9,
                    )
                ],
                aggregate_score=0.9,
                passed=True,
            )

            runner = EvaluationRunner(agent_factory=agent_factory)

            cases = [
                EvaluationCase(
                    id="case-1",
                    name="Test Case",
                    input={"query": "analyze portfolio"},
                )
            ]
            config = EvaluationRunConfig(run_id="run-1", parallel_workers=1)

            summary = await runner.run_evaluation(cases, config)

            assert summary.total_cases == 1
            assert summary.passed_cases == 1
            assert summary.failed_cases == 0
            assert summary.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_filters_by_tags(self) -> None:
        """Test that cases are filtered by tags."""

        def agent_factory() -> MockAgent:
            return MockAgent()

        with patch.object(
            EvaluationRunner,
            "_evaluate_single_case",
            new_callable=AsyncMock,
        ) as mock_eval:
            mock_eval.return_value = EvaluationResult(
                case_id="case-1",
                run_id="run-1",
                model="test",
                passed=True,
            )

            runner = EvaluationRunner(agent_factory=agent_factory)

            cases = [
                EvaluationCase(id="case-1", name="Case 1", input={}, tags=["portfolio"]),
                EvaluationCase(id="case-2", name="Case 2", input={}, tags=["research"]),
                EvaluationCase(id="case-3", name="Case 3", input={}, tags=["portfolio", "urgent"]),
            ]
            config = EvaluationRunConfig(
                run_id="run-1",
                tags_filter=["portfolio"],
            )

            _ = await runner.run_evaluation(cases, config)

            # Should only run cases with "portfolio" tag
            assert mock_eval.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_case_exceptions(self) -> None:
        """Test that exceptions in cases are handled gracefully."""

        def agent_factory() -> MockAgent:
            return MockAgent(error=RuntimeError("Agent failed"))

        runner = EvaluationRunner(agent_factory=agent_factory)

        cases = [EvaluationCase(id="case-1", name="Test", input={})]
        config = EvaluationRunConfig(run_id="run-1", timeout_seconds=10)

        # Mock the LLM to avoid actual API calls
        with patch.object(runner, "llm"):
            summary = await runner.run_evaluation(cases, config)

        assert summary.total_cases == 1
        assert summary.passed_cases == 0
        # The exception is caught and recorded in the result's errors list
        assert len(summary.results) == 1
        assert len(summary.results[0].errors) > 0
        assert "Agent failed" in summary.results[0].errors[0]

    def test_calculate_aggregate_score(self) -> None:
        """Test aggregate score calculation."""
        runner = EvaluationRunner()

        scores = [
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=0.8,
                confidence=1.0,
            ),
            DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=0.6,
                confidence=0.5,
            ),
        ]

        aggregate = runner._calculate_aggregate_score(scores)
        # (0.8 * 1.0 + 0.6 * 0.5) / (1.0 + 0.5) = 1.1 / 1.5 = 0.733
        assert abs(aggregate - 0.733) < 0.01

    def test_calculate_aggregate_empty(self) -> None:
        """Test aggregate with empty scores."""
        runner = EvaluationRunner()
        assert runner._calculate_aggregate_score([]) == 0.0

    def test_check_thresholds_pass(self) -> None:
        """Test threshold checking - passing case."""
        runner = EvaluationRunner()

        scores = [
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=0.8,
            ),
            DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=0.9,
            ),
        ]

        thresholds = {
            "accuracy": 0.7,
            "relevance": 0.7,
            "aggregate": 0.7,
        }

        assert runner._check_thresholds(scores, 0.85, thresholds) is True

    def test_check_thresholds_fail_dimension(self) -> None:
        """Test threshold checking - failing dimension."""
        runner = EvaluationRunner()

        scores = [
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=0.5,  # Below threshold
            ),
        ]

        thresholds = {"accuracy": 0.7, "aggregate": 0.5}

        assert runner._check_thresholds(scores, 0.5, thresholds) is False

    def test_check_thresholds_fail_aggregate(self) -> None:
        """Test threshold checking - failing aggregate."""
        runner = EvaluationRunner()

        scores = [
            DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=0.8,
            ),
        ]

        thresholds = {"accuracy": 0.7, "aggregate": 0.9}  # High aggregate threshold

        assert runner._check_thresholds(scores, 0.8, thresholds) is False

    def test_estimate_cost_sonnet(self) -> None:
        """Test cost estimation for Sonnet model."""
        runner = EvaluationRunner()
        cost = runner._estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )
        # Input: 1000/1M * 3.0 = 0.003
        # Output: 500/1M * 15.0 = 0.0075
        # Total: 0.0105
        assert abs(cost - 0.0105) < 0.001

    def test_estimate_cost_haiku(self) -> None:
        """Test cost estimation for Haiku model."""
        runner = EvaluationRunner()
        cost = runner._estimate_cost(
            "claude-3-haiku-20240307",
            input_tokens=1000,
            output_tokens=500,
        )
        # Input: 1000/1M * 0.25 = 0.00025
        # Output: 500/1M * 1.25 = 0.000625
        # Total: 0.000875
        assert abs(cost - 0.000875) < 0.0001

    def test_create_summary(self) -> None:
        """Test summary creation."""
        runner = EvaluationRunner()
        config = EvaluationRunConfig(run_id="test-run")

        results = [
            EvaluationResult(
                case_id="case-1",
                run_id="test-run",
                model="test",
                scores=[
                    DimensionScore(
                        dimension=EvaluationDimension.ACCURACY,
                        score=0.9,
                    )
                ],
                aggregate_score=0.9,
                passed=True,
                latency_ms=100,
                cost_usd=0.01,
            ),
            EvaluationResult(
                case_id="case-2",
                run_id="test-run",
                model="test",
                scores=[
                    DimensionScore(
                        dimension=EvaluationDimension.ACCURACY,
                        score=0.7,
                    )
                ],
                aggregate_score=0.7,
                passed=True,
                latency_ms=150,
                cost_usd=0.015,
            ),
        ]

        start_time = datetime.utcnow()
        summary = runner._create_summary(config, results, start_time, error_count=0)

        assert summary.total_cases == 2
        assert summary.passed_cases == 2
        assert summary.failed_cases == 0
        assert summary.pass_rate == 1.0
        assert summary.total_latency_ms == 250
        assert summary.mean_latency_ms == 125.0
        assert summary.total_cost_usd == 0.025
        assert "accuracy" in summary.mean_scores


class TestEvaluationRunnerABTest:
    """Tests for A/B testing functionality."""

    @pytest.mark.asyncio
    async def test_run_ab_test(self) -> None:
        """Test running an A/B test."""

        def control_factory() -> MockAgent:
            return MockAgent(response={"variant": "control"})

        def treatment_factory() -> MockAgent:
            return MockAgent(response={"variant": "treatment"})

        with patch.object(
            EvaluationRunner,
            "run_evaluation",
            new_callable=AsyncMock,
        ) as mock_run:
            # Return different summaries for control and treatment
            control_config = EvaluationRunConfig(run_id="control")
            treatment_config = EvaluationRunConfig(run_id="treatment")

            mock_run.side_effect = [
                EvaluationRunSummary(
                    run_id="control",
                    config=control_config,
                    total_cases=10,
                    passed_cases=7,
                    failed_cases=3,
                    pass_rate=0.7,
                    mean_scores={"accuracy": 0.75, "aggregate": 0.75},
                    std_scores={"accuracy": 0.1, "aggregate": 0.1},
                ),
                EvaluationRunSummary(
                    run_id="treatment",
                    config=treatment_config,
                    total_cases=10,
                    passed_cases=9,
                    failed_cases=1,
                    pass_rate=0.9,
                    mean_scores={"accuracy": 0.85, "aggregate": 0.85},
                    std_scores={"accuracy": 0.1, "aggregate": 0.1},
                ),
            ]

            runner = EvaluationRunner()

            cases = [EvaluationCase(id="case-1", name="Test", input={})]
            ab_config = ABTestConfig(
                name="Test A/B",
                control=control_config,
                treatment=treatment_config,
                min_samples=10,
            )

            result = await runner.run_ab_test(
                cases=cases,
                config=ab_config,
                control_agent_factory=control_factory,
                treatment_agent_factory=treatment_factory,
            )

            assert result.control_summary.run_id == "control"
            assert result.treatment_summary.run_id == "treatment"
            # With these scores, treatment should be better
            assert result.recommendation  # Should have a recommendation


class TestEvaluationRunnerRegressionTest:
    """Tests for regression testing functionality."""

    @pytest.mark.asyncio
    async def test_run_regression_test(self) -> None:
        """Test running a regression test."""

        def agent_factory() -> MockAgent:
            return MockAgent()

        baseline_config = EvaluationRunConfig(run_id="baseline")
        current_config = EvaluationRunConfig(run_id="current")

        baseline_summary = EvaluationRunSummary(
            run_id="baseline",
            config=baseline_config,
            total_cases=50,
            passed_cases=45,
            failed_cases=5,
            pass_rate=0.9,
            mean_scores={"accuracy": 0.85, "aggregate": 0.85},
            std_scores={"accuracy": 0.1, "aggregate": 0.1},
        )

        with patch.object(
            EvaluationRunner,
            "run_evaluation",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = EvaluationRunSummary(
                run_id="current",
                config=current_config,
                total_cases=50,
                passed_cases=40,
                failed_cases=10,
                pass_rate=0.8,
                mean_scores={"accuracy": 0.80, "aggregate": 0.80},
                std_scores={"accuracy": 0.1, "aggregate": 0.1},
            )

            runner = EvaluationRunner(agent_factory=agent_factory)

            cases = [EvaluationCase(id="case-1", name="Test", input={})]

            result = await runner.run_regression_test(
                cases=cases,
                current_config=current_config,
                baseline_summary=baseline_summary,
                agent_factory=agent_factory,
                regression_threshold=0.03,
            )

            assert "has_regression" in result
            assert "regressions" in result
            assert "improvements" in result
            assert "current_summary" in result
