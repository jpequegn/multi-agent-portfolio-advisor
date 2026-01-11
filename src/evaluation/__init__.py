"""Model evaluation suite for agent quality assessment.

This module provides tools for evaluating agent responses across
multiple quality dimensions using LLM-as-judge and statistical testing.
"""

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
    EvaluationType,
)
from src.evaluation.runner import EvaluationRunner, create_run_id
from src.evaluation.statistics import (
    StatisticalAnalyzer,
    calculate_cohens_d,
    calculate_mean,
    calculate_p_value,
    calculate_std,
)

__all__ = [
    # Models
    "ABTestConfig",
    "ABTestResult",
    "DimensionScore",
    "EvaluationCase",
    "EvaluationDimension",
    "EvaluationResult",
    "EvaluationRunConfig",
    "EvaluationRunSummary",
    "EvaluationType",
    # Judges
    "GroundTruthJudge",
    "LLMJudge",
    "SafetyJudge",
    # Runner
    "EvaluationRunner",
    "create_run_id",
    # Statistics
    "StatisticalAnalyzer",
    "calculate_cohens_d",
    "calculate_mean",
    "calculate_p_value",
    "calculate_std",
]
