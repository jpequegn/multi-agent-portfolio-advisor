"""Evaluation runner with parallel execution.

This module provides the main runner for executing evaluations
across multiple test cases in parallel.
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from anthropic import Anthropic

from src.agents.base import AgentState, BaseAgent
from src.evaluation.judges import GroundTruthJudge, LLMJudge, SafetyJudge
from src.evaluation.models import (
    ABTestConfig,
    ABTestResult,
    DimensionScore,
    EvaluationCase,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRunConfig,
    EvaluationRunSummary,
)
from src.evaluation.statistics import StatisticalAnalyzer

logger = structlog.get_logger(__name__)


class EvaluationRunner:
    """Runner for executing evaluation suites in parallel.

    Coordinates test case execution, judging, and result aggregation.
    """

    def __init__(
        self,
        llm: Anthropic | None = None,
        agent_factory: Callable[[], BaseAgent] | None = None,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            llm: Optional Anthropic client for judging.
            agent_factory: Factory function to create agent instances.
        """
        self.llm = llm or Anthropic()
        self.agent_factory = agent_factory
        self._logger = logger.bind(component="evaluation_runner")
        self._analyzer = StatisticalAnalyzer()

    async def run_evaluation(
        self,
        cases: list[EvaluationCase],
        config: EvaluationRunConfig,
        agent_factory: Callable[[], BaseAgent] | None = None,
    ) -> EvaluationRunSummary:
        """Run evaluation on a set of test cases.

        Args:
            cases: List of evaluation cases to run.
            config: Configuration for this run.
            agent_factory: Optional factory to override instance factory.

        Returns:
            Summary of evaluation results.
        """
        factory = agent_factory or self.agent_factory
        if factory is None:
            raise ValueError("No agent factory provided")

        start_time = datetime.utcnow()
        self._logger.info(
            "evaluation_run_started",
            run_id=config.run_id,
            total_cases=len(cases),
            parallel_workers=config.parallel_workers,
        )

        # Filter cases by tags if specified
        if config.tags_filter:
            cases = [c for c in cases if any(t in c.tags for t in config.tags_filter)]
            self._logger.info(
                "cases_filtered_by_tags",
                original_count=len(cases),
                filtered_count=len(cases),
                tags=config.tags_filter,
            )

        # Create semaphore for parallel execution
        semaphore = asyncio.Semaphore(config.parallel_workers)

        # Create judges
        llm_judge = LLMJudge(llm=self.llm, model=config.judge_model)
        ground_truth_judge = GroundTruthJudge()
        safety_judge = SafetyJudge()

        # Run evaluations in parallel
        async def run_single_case(case: EvaluationCase) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_single_case(
                    case=case,
                    config=config,
                    agent_factory=factory,
                    llm_judge=llm_judge,
                    ground_truth_judge=ground_truth_judge,
                    safety_judge=safety_judge,
                )

        results = await asyncio.gather(
            *[run_single_case(case) for case in cases],
            return_exceptions=True,
        )

        # Process results
        successful_results: list[EvaluationResult] = []
        error_count = 0

        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                self._logger.error(
                    "case_evaluation_exception",
                    case_id=cases[i].id,
                    error=str(result),
                )
                # Create error result
                error_result = EvaluationResult(
                    case_id=cases[i].id,
                    run_id=config.run_id,
                    model=config.model,
                    prompt_version=config.prompt_version,
                    passed=False,
                    errors=[str(result)],
                )
                successful_results.append(error_result)
                error_count += 1
            else:
                # result is EvaluationResult after the isinstance check
                successful_results.append(result)

        # Calculate summary statistics
        summary = self._create_summary(
            config=config,
            results=successful_results,
            start_time=start_time,
            error_count=error_count,
        )

        self._logger.info(
            "evaluation_run_completed",
            run_id=config.run_id,
            total_cases=summary.total_cases,
            passed=summary.passed_cases,
            failed=summary.failed_cases,
            errors=summary.error_cases,
            pass_rate=summary.pass_rate,
            duration_seconds=summary.duration_seconds,
        )

        return summary

    async def _evaluate_single_case(
        self,
        case: EvaluationCase,
        config: EvaluationRunConfig,
        agent_factory: Callable[[], BaseAgent],
        llm_judge: LLMJudge,
        ground_truth_judge: GroundTruthJudge,
        safety_judge: SafetyJudge,
    ) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            case: The test case to evaluate.
            config: Run configuration.
            agent_factory: Factory to create agent.
            llm_judge: LLM-based judge.
            ground_truth_judge: Ground truth comparison judge.
            safety_judge: Safety evaluation judge.

        Returns:
            Evaluation result for the case.
        """
        self._logger.debug("evaluating_case", case_id=case.id, case_name=case.name)

        start_time = time.monotonic()
        errors: list[str] = []

        try:
            # Create agent and run
            agent = agent_factory()
            agent_state = AgentState(
                messages=[],
                context=case.input,
                errors=[],
            )

            # Execute with timeout
            result_state = await asyncio.wait_for(
                agent(agent_state),
                timeout=config.timeout_seconds,
            )

            latency_ms = int((time.monotonic() - start_time) * 1000)

            # Extract output
            output = result_state.context
            errors.extend(result_state.errors)

        except TimeoutError:
            self._logger.warning(
                "case_timeout",
                case_id=case.id,
                timeout_seconds=config.timeout_seconds,
            )
            return EvaluationResult(
                case_id=case.id,
                run_id=config.run_id,
                model=config.model,
                prompt_version=config.prompt_version,
                passed=False,
                errors=[f"Timeout after {config.timeout_seconds}s"],
            )
        except Exception as e:
            self._logger.error(
                "case_execution_error",
                case_id=case.id,
                error=str(e),
            )
            return EvaluationResult(
                case_id=case.id,
                run_id=config.run_id,
                model=config.model,
                prompt_version=config.prompt_version,
                passed=False,
                errors=[f"Execution error: {e!s}"],
            )

        # Evaluate the output
        scores: list[DimensionScore] = []
        overall_reasoning = ""

        try:
            # Use LLM judge for most dimensions
            llm_dimensions = [
                d for d in case.evaluation_criteria
                if d not in [EvaluationDimension.SAFETY]
            ]

            if llm_dimensions:
                # Create a modified case with only LLM dimensions
                llm_case = EvaluationCase(
                    id=case.id,
                    name=case.name,
                    input=case.input,
                    expected_output=case.expected_output,
                    evaluation_criteria=llm_dimensions,
                    tags=case.tags,
                )
                llm_scores, llm_reasoning, _ = await llm_judge.evaluate(llm_case, output)
                scores.extend(llm_scores)
                overall_reasoning = llm_reasoning

            # Use safety judge for safety dimension
            if EvaluationDimension.SAFETY in case.evaluation_criteria:
                safety_score, safety_reasoning = safety_judge.evaluate(output)
                scores.append(
                    DimensionScore(
                        dimension=EvaluationDimension.SAFETY,
                        score=safety_score,
                        reasoning=safety_reasoning,
                        confidence=0.9,
                    )
                )

            # If we have expected output and accuracy is a criterion, augment with ground truth
            if case.expected_output and EvaluationDimension.ACCURACY in case.evaluation_criteria:
                gt_score, gt_reasoning = ground_truth_judge.evaluate(
                    case.expected_output, output
                )
                # Adjust accuracy score based on ground truth
                for s in scores:
                    if s.dimension == EvaluationDimension.ACCURACY:
                        # Average LLM and ground truth scores
                        s.score = (s.score + gt_score) / 2
                        s.reasoning += f" Ground truth comparison: {gt_reasoning}"

        except Exception as e:
            self._logger.error(
                "evaluation_error",
                case_id=case.id,
                error=str(e),
            )
            errors.append(f"Evaluation error: {e!s}")

        # Calculate aggregate score
        aggregate_score = self._calculate_aggregate_score(scores)

        # Determine pass/fail
        passed = self._check_thresholds(scores, aggregate_score, config.score_thresholds)

        # Estimate cost (rough approximation based on model)
        input_tokens = len(str(case.input)) // 4  # Very rough estimate
        output_tokens = len(str(output)) // 4
        cost_usd = self._estimate_cost(config.model, input_tokens, output_tokens)

        result = EvaluationResult(
            case_id=case.id,
            run_id=config.run_id,
            model=config.model,
            prompt_version=config.prompt_version,
            scores=scores,
            aggregate_score=aggregate_score,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output=output,
            evaluator_reasoning=overall_reasoning,
            passed=passed,
            errors=errors,
        )

        self._logger.debug(
            "case_evaluated",
            case_id=case.id,
            aggregate_score=aggregate_score,
            passed=passed,
            latency_ms=latency_ms,
        )

        return result

    def _calculate_aggregate_score(self, scores: list[DimensionScore]) -> float:
        """Calculate weighted aggregate score.

        Args:
            scores: List of dimension scores.

        Returns:
            Weighted average score.
        """
        if not scores:
            return 0.0

        # Weight by confidence
        total_weight = sum(s.confidence for s in scores)
        if total_weight == 0:
            return sum(s.score for s in scores) / len(scores)

        return sum(s.score * s.confidence for s in scores) / total_weight

    def _check_thresholds(
        self,
        scores: list[DimensionScore],
        aggregate: float,
        thresholds: dict[str, float],
    ) -> bool:
        """Check if scores meet configured thresholds.

        Args:
            scores: List of dimension scores.
            aggregate: Aggregate score.
            thresholds: Threshold configuration.

        Returns:
            True if all thresholds are met.
        """
        # Check aggregate threshold
        if aggregate < thresholds.get("aggregate", 0.0):
            return False

        # Check individual dimension thresholds
        for score in scores:
            threshold = thresholds.get(score.dimension.value, 0.0)
            if score.score < threshold:
                return False

        return True

    def _estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Estimate cost for API call.

        Args:
            model: Model identifier.
            input_tokens: Input token count.
            output_tokens: Output token count.

        Returns:
            Estimated cost in USD.
        """
        # Approximate costs per 1M tokens (as of 2024)
        costs = {
            "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
            "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        }

        model_costs = costs.get(model, {"input": 3.0, "output": 15.0})

        input_cost = (input_tokens / 1_000_000) * model_costs["input"]
        output_cost = (output_tokens / 1_000_000) * model_costs["output"]

        return input_cost + output_cost

    def _create_summary(
        self,
        config: EvaluationRunConfig,
        results: list[EvaluationResult],
        start_time: datetime,
        error_count: int,
    ) -> EvaluationRunSummary:
        """Create summary from evaluation results.

        Args:
            config: Run configuration.
            results: List of evaluation results.
            start_time: When the run started.
            error_count: Number of cases that errored.

        Returns:
            Summary statistics for the run.
        """
        completed_at = datetime.utcnow()
        duration = (completed_at - start_time).total_seconds()

        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.passed and not r.errors)
        failed_cases = sum(1 for r in results if not r.passed and not r.errors)

        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0

        # Calculate dimension statistics
        stats = self._analyzer.calculate_run_statistics(results)

        mean_scores = {dim: s["mean"] for dim, s in stats.items()}
        std_scores = {dim: s["std"] for dim, s in stats.items()}
        min_scores = {dim: s["min"] for dim, s in stats.items()}
        max_scores = {dim: s["max"] for dim, s in stats.items()}

        # Performance totals
        total_latency = sum(r.latency_ms for r in results)
        total_cost = sum(r.cost_usd for r in results)
        total_input = sum(r.input_tokens for r in results)
        total_output = sum(r.output_tokens for r in results)

        return EvaluationRunSummary(
            run_id=config.run_id,
            config=config,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            error_cases=error_count,
            pass_rate=pass_rate,
            mean_scores=mean_scores,
            std_scores=std_scores,
            min_scores=min_scores,
            max_scores=max_scores,
            total_latency_ms=total_latency,
            mean_latency_ms=total_latency / total_cases if total_cases > 0 else 0.0,
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            started_at=start_time,
            completed_at=completed_at,
            duration_seconds=duration,
            results=results,
        )

    async def run_ab_test(
        self,
        cases: list[EvaluationCase],
        config: ABTestConfig,
        control_agent_factory: Callable[[], BaseAgent],
        treatment_agent_factory: Callable[[], BaseAgent],
    ) -> ABTestResult:
        """Run an A/B test comparing two variants.

        Args:
            cases: Test cases to run on both variants.
            config: A/B test configuration.
            control_agent_factory: Factory for control variant agent.
            treatment_agent_factory: Factory for treatment variant agent.

        Returns:
            A/B test results with statistical analysis.
        """
        self._logger.info(
            "ab_test_started",
            test_name=config.name,
            total_cases=len(cases),
        )

        # Run control
        self._logger.info("running_control_variant")
        control_summary = await self.run_evaluation(
            cases=cases,
            config=config.control,
            agent_factory=control_agent_factory,
        )

        # Run treatment
        self._logger.info("running_treatment_variant")
        treatment_summary = await self.run_evaluation(
            cases=cases,
            config=config.treatment,
            agent_factory=treatment_agent_factory,
        )

        # Analyze results
        result = self._analyzer.analyze_ab_test(
            config=config,
            control_summary=control_summary,
            treatment_summary=treatment_summary,
        )

        self._logger.info(
            "ab_test_completed",
            test_name=config.name,
            winner=result.winner,
            is_significant=result.is_significant,
            p_value=result.p_value,
        )

        return result

    async def run_regression_test(
        self,
        cases: list[EvaluationCase],
        current_config: EvaluationRunConfig,
        baseline_summary: EvaluationRunSummary,
        agent_factory: Callable[[], BaseAgent],
        regression_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Run regression test against a baseline.

        Args:
            cases: Test cases to run.
            current_config: Configuration for current run.
            baseline_summary: Previous baseline results.
            agent_factory: Factory to create agent.
            regression_threshold: Score drop threshold.

        Returns:
            Regression analysis results.
        """
        self._logger.info(
            "regression_test_started",
            baseline_run=baseline_summary.run_id,
            current_run=current_config.run_id,
        )

        # Run current evaluation
        current_summary = await self.run_evaluation(
            cases=cases,
            config=current_config,
            agent_factory=agent_factory,
        )

        # Detect regressions
        analysis = self._analyzer.detect_regression(
            baseline=baseline_summary,
            current=current_summary,
            threshold=regression_threshold,
        )

        analysis["current_summary"] = current_summary

        self._logger.info(
            "regression_test_completed",
            has_regression=analysis["has_regression"],
            regression_count=len(analysis["regressions"]),
            improvement_count=len(analysis["improvements"]),
        )

        return analysis


def create_run_id() -> str:
    """Generate a unique run ID.

    Returns:
        UUID-based run identifier.
    """
    return f"eval-{uuid.uuid4().hex[:12]}"
