"""Evaluation data models and schemas.

This module defines the core data structures for the evaluation framework,
including test cases, results, and evaluation metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationDimension(str, Enum):
    """Dimensions on which agent responses are evaluated."""

    ACCURACY = "accuracy"  # Factual correctness
    RELEVANCE = "relevance"  # Response relevance to query
    COMPLETENESS = "completeness"  # Coverage of requirements
    CONSISTENCY = "consistency"  # Reproducibility across runs
    SAFETY = "safety"  # Harmful content detection
    ACTIONABILITY = "actionability"  # Can user act on the response


class EvaluationType(str, Enum):
    """Types of evaluation runs."""

    GOLDEN_SET = "golden_set"  # Fixed test cases with known answers
    AB_TEST = "ab_test"  # Compare prompt/model variants
    REGRESSION = "regression"  # Catch quality degradation
    PRODUCTION_SAMPLE = "production_sample"  # Evaluate random production requests


class EvaluationCase(BaseModel):
    """A single evaluation test case.

    Represents a test scenario with input, expected output (optional),
    and criteria for evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique identifier for the test case")
    name: str = Field(..., description="Human-readable name for the test case")
    description: str = Field(default="", description="Detailed description of what this tests")
    input: dict[str, Any] = Field(
        ..., description="Input data (portfolio, query, context, etc.)"
    )
    expected_output: dict[str, Any] | None = Field(
        default=None, description="Expected output for golden set evaluation"
    )
    evaluation_criteria: list[EvaluationDimension] = Field(
        default_factory=lambda: [
            EvaluationDimension.ACCURACY,
            EvaluationDimension.RELEVANCE,
            EvaluationDimension.COMPLETENESS,
            EvaluationDimension.ACTIONABILITY,
        ],
        description="Dimensions to evaluate on",
    )
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    evaluation_type: EvaluationType = Field(
        default=EvaluationType.GOLDEN_SET, description="Type of evaluation"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: EvaluationDimension
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    reasoning: str = Field(default="", description="Explanation for the score")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in the score"
    )


class EvaluationResult(BaseModel):
    """Result from evaluating a single test case.

    Contains scores across dimensions, timing, cost, and the actual output.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., description="ID of the evaluated test case")
    run_id: str = Field(..., description="ID of the evaluation run")
    model: str = Field(..., description="Model used for generation")
    prompt_version: str = Field(default="default", description="Version of the prompt")
    scores: list[DimensionScore] = Field(
        default_factory=list, description="Scores per dimension"
    )
    aggregate_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weighted average score"
    )
    latency_ms: int = Field(default=0, ge=0, description="Response latency in milliseconds")
    cost_usd: float = Field(default=0.0, ge=0.0, description="Cost in USD")
    input_tokens: int = Field(default=0, ge=0, description="Input token count")
    output_tokens: int = Field(default=0, ge=0, description="Output token count")
    output: dict[str, Any] = Field(
        default_factory=dict, description="Actual agent output"
    )
    evaluator_reasoning: str = Field(
        default="", description="Overall reasoning from evaluator"
    )
    passed: bool = Field(default=True, description="Whether the case passed thresholds")
    errors: list[str] = Field(default_factory=list, description="Any errors during evaluation")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When evaluation was performed"
    )

    def get_score(self, dimension: EvaluationDimension) -> float | None:
        """Get the score for a specific dimension."""
        for ds in self.scores:
            if ds.dimension == dimension:
                return ds.score
        return None


class EvaluationRunConfig(BaseModel):
    """Configuration for an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Unique identifier for this run")
    name: str = Field(default="", description="Human-readable name")
    description: str = Field(default="", description="Description of this run")
    evaluation_type: EvaluationType = Field(
        default=EvaluationType.GOLDEN_SET, description="Type of evaluation"
    )
    model: str = Field(
        default="claude-sonnet-4-20250514", description="Model to use for agent"
    )
    judge_model: str = Field(
        default="claude-sonnet-4-20250514", description="Model to use for judging"
    )
    prompt_version: str = Field(default="default", description="Prompt version identifier")
    parallel_workers: int = Field(
        default=5, ge=1, le=20, description="Number of parallel workers"
    )
    timeout_seconds: int = Field(
        default=120, ge=10, le=600, description="Timeout per case in seconds"
    )
    score_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "accuracy": 0.7,
            "relevance": 0.7,
            "completeness": 0.6,
            "actionability": 0.6,
            "aggregate": 0.65,
        },
        description="Minimum scores to pass",
    )
    tags_filter: list[str] | None = Field(
        default=None, description="Only run cases with these tags"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional run metadata"
    )


class EvaluationRunSummary(BaseModel):
    """Summary statistics for an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    config: EvaluationRunConfig
    total_cases: int = Field(default=0, ge=0)
    passed_cases: int = Field(default=0, ge=0)
    failed_cases: int = Field(default=0, ge=0)
    error_cases: int = Field(default=0, ge=0)
    pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Aggregate scores by dimension
    mean_scores: dict[str, float] = Field(default_factory=dict)
    std_scores: dict[str, float] = Field(default_factory=dict)
    min_scores: dict[str, float] = Field(default_factory=dict)
    max_scores: dict[str, float] = Field(default_factory=dict)

    # Performance metrics
    total_latency_ms: int = Field(default=0, ge=0)
    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)
    duration_seconds: float = Field(default=0.0, ge=0.0)

    results: list[EvaluationResult] = Field(default_factory=list)


class ABTestConfig(BaseModel):
    """Configuration for A/B testing between variants."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the A/B test")
    description: str = Field(default="", description="Description of what we're testing")
    control: EvaluationRunConfig = Field(..., description="Control variant configuration")
    treatment: EvaluationRunConfig = Field(..., description="Treatment variant configuration")
    min_samples: int = Field(
        default=30, ge=10, description="Minimum samples for statistical significance"
    )
    confidence_level: float = Field(
        default=0.95, ge=0.8, le=0.99, description="Required confidence level"
    )


class ABTestResult(BaseModel):
    """Results from an A/B test comparison."""

    model_config = ConfigDict(extra="forbid")

    config: ABTestConfig
    control_summary: EvaluationRunSummary
    treatment_summary: EvaluationRunSummary

    # Statistical results per dimension
    dimension_comparisons: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-dimension statistical comparisons",
    )

    # Overall verdict
    winner: str | None = Field(
        default=None, description="'control', 'treatment', or None if inconclusive"
    )
    is_significant: bool = Field(default=False, description="Whether difference is significant")
    p_value: float | None = Field(default=None, description="Overall p-value")
    effect_size: float | None = Field(default=None, description="Cohen's d effect size")
    recommendation: str = Field(default="", description="Human-readable recommendation")
