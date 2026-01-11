"""Command-line interface for evaluation suite.

This module provides CLI commands for running evaluations,
comparing results, and generating reports.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.evaluation.models import (
    EvaluationCase,
    EvaluationRunConfig,
)
from src.evaluation.runner import EvaluationRunner, create_run_id
from src.evaluation.statistics import StatisticalAnalyzer
from src.evaluation.storage import EvaluationStorage

logger = structlog.get_logger(__name__)

# Default evaluation cases for quick runs
DEFAULT_QUICK_CASES = [
    EvaluationCase(
        id="quick-portfolio-analysis",
        name="Basic Portfolio Analysis",
        input={
            "query": "Analyze my portfolio",
            "portfolio": {"AAPL": 50, "GOOGL": 30, "MSFT": 20},
        },
        tags=["portfolio", "quick"],
    ),
    EvaluationCase(
        id="quick-risk-assessment",
        name="Basic Risk Assessment",
        input={
            "query": "What are the risks in my portfolio?",
            "portfolio": {"AAPL": 50, "GOOGL": 30, "MSFT": 20},
        },
        tags=["risk", "quick"],
    ),
    EvaluationCase(
        id="quick-recommendation",
        name="Basic Recommendation",
        input={
            "query": "Should I rebalance my portfolio?",
            "portfolio": {"AAPL": 50, "GOOGL": 30, "MSFT": 20},
            "risk_tolerance": "moderate",
        },
        tags=["recommendation", "quick"],
    ),
]


def load_cases_from_file(path: Path) -> list[EvaluationCase]:
    """Load evaluation cases from a JSON file.

    Args:
        path: Path to JSON file containing cases.

    Returns:
        List of evaluation cases.
    """
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data:
        cases.append(EvaluationCase.model_validate(item))

    return cases


def save_results(results: dict[str, Any], path: Path) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Results dictionary to save.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


async def run_evaluation_command(args: argparse.Namespace) -> int:
    """Execute the 'run' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Load cases
    if args.cases_file:
        cases = load_cases_from_file(Path(args.cases_file))
    elif args.full:
        # Load full test suite from default location
        cases_path = Path("tests/evaluation_cases.json")
        if cases_path.exists():
            cases = load_cases_from_file(cases_path)
        else:
            logger.warning("full_cases_not_found", path=str(cases_path))
            cases = DEFAULT_QUICK_CASES
    else:
        # Quick evaluation with default cases
        cases = DEFAULT_QUICK_CASES

    # Filter by tags if specified
    if args.tags:
        tag_set = set(args.tags.split(","))
        cases = [c for c in cases if any(t in tag_set for t in c.tags)]

    if not cases:
        logger.error("no_cases_to_evaluate")
        return 1

    # Create configuration
    run_id = args.run_id or create_run_id()
    config = EvaluationRunConfig(
        run_id=run_id,
        name=f"CLI Run {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        model=args.model,
        parallel_workers=args.workers,
        score_thresholds={"aggregate": args.threshold},
    )

    logger.info(
        "starting_evaluation",
        run_id=run_id,
        case_count=len(cases),
        model=args.model,
    )

    # Create a mock agent factory for CLI testing
    # In production, this would be replaced with actual agent factory
    from src.agents.base import AgentState, BaseAgent

    def mock_agent_factory() -> BaseAgent:
        from unittest.mock import MagicMock

        class MockCLIAgent(BaseAgent):
            def __init__(self) -> None:
                super().__init__(llm=MagicMock())

            @property
            def name(self) -> str:
                return "cli_mock_agent"

            @property
            def description(self) -> str:
                return "Mock agent for CLI evaluation"

            @property
            def system_prompt(self) -> str:
                return "You are a mock portfolio advisor."

            async def invoke(self, state: AgentState) -> AgentState:
                # Return mock response for testing
                state.context.update({
                    "recommendation": "hold",
                    "confidence": 0.75,
                    "reasoning": "Based on current market conditions...",
                    "risk_level": "moderate",
                })
                return state

        return MockCLIAgent()

    # Run evaluation
    runner = EvaluationRunner(agent_factory=mock_agent_factory)

    try:
        summary = await runner.run_evaluation(cases, config)
    except Exception as e:
        logger.error("evaluation_failed", error=str(e))
        return 1

    # Store results
    storage = EvaluationStorage()
    await storage.save_summary(summary)

    # Prepare output
    results = {
        "run_id": summary.run_id,
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "failed_cases": summary.failed_cases,
        "error_cases": summary.error_cases,
        "pass_rate": summary.pass_rate,
        "mean_scores": summary.mean_scores,
        "std_scores": summary.std_scores,
        "total_cost_usd": summary.total_cost_usd,
        "duration_seconds": summary.duration_seconds,
        "threshold": args.threshold,
        "passed_threshold": summary.pass_rate >= args.threshold,
    }

    # Save output
    if args.output:
        save_results(results, Path(args.output))
        logger.info("results_saved", path=args.output)

    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Run ID: {summary.run_id}")
    print(f"Total Cases: {summary.total_cases}")
    print(f"Passed: {summary.passed_cases}")
    print(f"Failed: {summary.failed_cases}")
    print(f"Pass Rate: {summary.pass_rate:.2%}")
    print(f"Duration: {summary.duration_seconds:.1f}s")
    print(f"\nAggregate Score: {summary.mean_scores.get('aggregate', 0):.3f}")
    print(f"Threshold: {args.threshold}")
    print(f"{'='*50}\n")

    # Return exit code based on threshold
    if summary.pass_rate < args.threshold:
        logger.warning(
            "evaluation_below_threshold",
            pass_rate=summary.pass_rate,
            threshold=args.threshold,
        )
        return 1

    return 0


async def run_regression_command(args: argparse.Namespace) -> int:
    """Execute the 'regression' command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for regression detected).
    """
    # Load current results
    if not Path(args.current).exists():
        logger.error("current_results_not_found", path=args.current)
        return 1

    with open(args.current) as f:
        current_data = json.load(f)

    # Get baseline from storage
    storage = EvaluationStorage()

    if args.baseline:
        baseline_summary = await storage.get_summary(args.baseline)
    else:
        baseline_summary = await storage.get_baseline_summary()

    if not baseline_summary:
        logger.warning("no_baseline_found")
        # No baseline to compare against, pass by default
        results = {
            "has_regression": False,
            "regressions": [],
            "improvements": [],
            "message": "No baseline available for comparison",
        }
        if args.output:
            save_results(results, Path(args.output))
        return 0

    # Compare scores
    analyzer = StatisticalAnalyzer()

    # Build current summary for comparison
    from src.evaluation.models import EvaluationRunConfig, EvaluationRunSummary

    current_config = EvaluationRunConfig(run_id=current_data.get("run_id", "current"))
    current_summary = EvaluationRunSummary(
        run_id=current_data.get("run_id", "current"),
        config=current_config,
        total_cases=current_data.get("total_cases", 0),
        passed_cases=current_data.get("passed_cases", 0),
        failed_cases=current_data.get("failed_cases", 0),
        pass_rate=current_data.get("pass_rate", 0),
        mean_scores=current_data.get("mean_scores", {}),
        std_scores=current_data.get("std_scores", {}),
    )

    # Detect regressions
    analysis = analyzer.detect_regression(
        baseline=baseline_summary,
        current=current_summary,
        threshold=args.threshold,
    )

    # Prepare output
    results = {
        "has_regression": analysis["has_regression"],
        "regressions": analysis["regressions"],
        "improvements": analysis["improvements"],
        "baseline_run_id": baseline_summary.run_id,
        "current_run_id": current_summary.run_id,
        "threshold": args.threshold,
    }

    if args.output:
        save_results(results, Path(args.output))
        logger.info("regression_report_saved", path=args.output)

    # Print summary
    print(f"\n{'='*50}")
    print("REGRESSION ANALYSIS")
    print(f"{'='*50}")
    print(f"Baseline: {baseline_summary.run_id}")
    print(f"Current: {current_summary.run_id}")
    print(f"Threshold: {args.threshold}")

    if analysis["regressions"]:
        print(f"\n⚠️  REGRESSIONS DETECTED: {len(analysis['regressions'])}")
        for reg in analysis["regressions"]:
            print(f"  - {reg['dimension']}: {reg['baseline_score']:.3f} → {reg['current_score']:.3f} (drop: {reg['drop']:.3f})")

    if analysis["improvements"]:
        print(f"\n✅ IMPROVEMENTS: {len(analysis['improvements'])}")
        for imp in analysis["improvements"]:
            print(f"  - {imp['dimension']}: {imp['baseline_score']:.3f} → {imp['current_score']:.3f} (gain: {imp['gain']:.3f})")

    print(f"{'='*50}\n")

    return 1 if analysis["has_regression"] else 0


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Model Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run evaluation suite")
    run_parser.add_argument(
        "--run-id",
        help="Unique run identifier",
    )
    run_parser.add_argument(
        "--cases-file",
        help="Path to JSON file with evaluation cases",
    )
    run_parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation suite",
    )
    run_parser.add_argument(
        "--tags",
        help="Comma-separated tags to filter cases",
    )
    run_parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use for evaluation",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Minimum pass rate threshold",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        help="Output file for results JSON",
    )

    # Regression command
    reg_parser = subparsers.add_parser("regression", help="Check for quality regression")
    reg_parser.add_argument(
        "--current",
        required=True,
        help="Path to current evaluation results JSON",
    )
    reg_parser.add_argument(
        "--baseline",
        help="Baseline run ID to compare against (uses latest if not specified)",
    )
    reg_parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Minimum score drop to consider a regression",
    )
    reg_parser.add_argument(
        "--output",
        "-o",
        help="Output file for regression report JSON",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run appropriate command
    if args.command == "run":
        return asyncio.run(run_evaluation_command(args))
    elif args.command == "regression":
        return asyncio.run(run_regression_command(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())
