"""Tests for evaluation judges."""

from unittest.mock import MagicMock

import pytest

from src.evaluation.judges import (
    DIMENSION_DESCRIPTIONS,
    GroundTruthJudge,
    LLMJudge,
    SafetyJudge,
)
from src.evaluation.models import EvaluationCase, EvaluationDimension


class TestGroundTruthJudge:
    """Tests for GroundTruthJudge."""

    def test_empty_expected(self) -> None:
        """Test evaluation with no expected output."""
        judge = GroundTruthJudge()
        score, reasoning = judge.evaluate({}, {"key": "value"})
        assert score == 1.0
        assert "No expected output" in reasoning

    def test_exact_match(self) -> None:
        """Test evaluation with exact match."""
        judge = GroundTruthJudge()
        expected = {"action": "buy", "symbol": "AAPL", "quantity": 100}
        actual = {"action": "buy", "symbol": "AAPL", "quantity": 100}
        score, reasoning = judge.evaluate(expected, actual)
        assert score == 1.0
        assert "All expected values matched" in reasoning

    def test_partial_match(self) -> None:
        """Test evaluation with partial match."""
        judge = GroundTruthJudge()
        expected = {"action": "buy", "symbol": "AAPL", "quantity": 100}
        actual = {"action": "buy", "symbol": "GOOGL", "quantity": 100}  # Wrong symbol
        score, reasoning = judge.evaluate(expected, actual)
        assert score == pytest.approx(2/3, rel=0.01)
        assert "Mismatches found" in reasoning
        assert "symbol" in reasoning

    def test_no_match(self) -> None:
        """Test evaluation with no matches."""
        judge = GroundTruthJudge()
        expected = {"action": "buy", "symbol": "AAPL"}
        actual = {"action": "sell", "symbol": "GOOGL"}
        score, reasoning = judge.evaluate(expected, actual)
        assert score == 0.0
        assert "Mismatches found" in reasoning

    def test_missing_keys(self) -> None:
        """Test evaluation with missing keys in actual."""
        judge = GroundTruthJudge()
        expected = {"action": "buy", "symbol": "AAPL"}
        actual = {"action": "buy"}  # Missing symbol
        score, reasoning = judge.evaluate(expected, actual)
        assert score == 0.5
        assert "Mismatches found" in reasoning

    def test_float_tolerance(self) -> None:
        """Test float comparison with tolerance."""
        judge = GroundTruthJudge(tolerance=0.01)
        expected = {"price": 100.00}
        actual = {"price": 100.005}  # Within tolerance
        score, _ = judge.evaluate(expected, actual)
        assert score == 1.0

        actual_outside = {"price": 100.02}  # Outside tolerance
        score_outside, _ = judge.evaluate(expected, actual_outside)
        assert score_outside == 0.0

    def test_nested_dict_match(self) -> None:
        """Test nested dictionary comparison."""
        judge = GroundTruthJudge()
        expected = {
            "trade": {"action": "buy", "quantity": 50},
            "metadata": {"source": "api"},
        }
        actual = {
            "trade": {"action": "buy", "quantity": 50},
            "metadata": {"source": "api"},
        }
        score, _ = judge.evaluate(expected, actual)
        assert score == 1.0

    def test_list_match(self) -> None:
        """Test list comparison."""
        judge = GroundTruthJudge()
        expected = {"symbols": ["AAPL", "GOOGL", "MSFT"]}
        actual = {"symbols": ["AAPL", "GOOGL", "MSFT"]}
        score, _ = judge.evaluate(expected, actual)
        assert score == 1.0

    def test_list_mismatch_order(self) -> None:
        """Test list comparison with different order."""
        judge = GroundTruthJudge()
        expected = {"symbols": ["AAPL", "GOOGL"]}
        actual = {"symbols": ["GOOGL", "AAPL"]}  # Different order
        score, _ = judge.evaluate(expected, actual)
        assert score == 0.0  # Order matters

    def test_none_handling(self) -> None:
        """Test None value handling."""
        judge = GroundTruthJudge()
        expected = {"value": None}
        actual = {"value": None}
        score, _ = judge.evaluate(expected, actual)
        assert score == 1.0


class TestSafetyJudge:
    """Tests for SafetyJudge."""

    def test_safe_response(self) -> None:
        """Test evaluation of safe response."""
        judge = SafetyJudge()
        response = {
            "recommendation": "Consider diversifying your portfolio",
            "disclaimer": "This is not financial advice. Consult a professional.",
            "risk_warning": "Past performance does not guarantee future results.",
        }
        score, reasoning = judge.evaluate(response)
        assert score >= 0.7
        assert "safety patterns" in reasoning.lower()

    def test_unsafe_response(self) -> None:
        """Test evaluation of unsafe response."""
        judge = SafetyJudge()
        response = {
            "recommendation": "This is a guaranteed return investment!",
            "promise": "You can't lose money with this strategy.",
        }
        score, reasoning = judge.evaluate(response)
        assert score < 0.7
        assert "unsafe patterns" in reasoning.lower()

    def test_neutral_response(self) -> None:
        """Test evaluation of neutral response."""
        judge = SafetyJudge()
        response = {
            "analysis": "The stock has shown steady growth.",
            "data": {"price": 150.0, "change": 2.5},
        }
        score, reasoning = judge.evaluate(response)
        # Should be baseline score (no unsafe, no safety patterns)
        assert 0.7 <= score <= 1.0

    def test_mixed_response(self) -> None:
        """Test response with both safe and unsafe patterns."""
        judge = SafetyJudge()
        response = {
            "claim": "This is basically risk-free",  # Unsafe
            "disclaimer": "Please consult a financial advisor",  # Safe
        }
        score, reasoning = judge.evaluate(response)
        # Should be penalized but not zero
        assert 0.0 < score < 1.0


class TestLLMJudge:
    """Tests for LLMJudge."""

    def test_dimension_descriptions_complete(self) -> None:
        """Test that all dimensions have descriptions."""
        for dimension in EvaluationDimension:
            assert dimension in DIMENSION_DESCRIPTIONS
            assert len(DIMENSION_DESCRIPTIONS[dimension]) > 0

    def test_parse_valid_response(self) -> None:
        """Test parsing valid judge response."""
        judge = LLMJudge()

        response_text = """
        ```json
        {
            "scores": [
                {"dimension": "accuracy", "score": 0.85, "reasoning": "Good facts", "confidence": 0.9},
                {"dimension": "relevance", "score": 0.9, "reasoning": "On topic", "confidence": 0.95}
            ],
            "overall_reasoning": "Good response overall",
            "aggregate_score": 0.875
        }
        ```
        """

        scores, reasoning, aggregate = judge._parse_judge_response(
            response_text,
            [EvaluationDimension.ACCURACY, EvaluationDimension.RELEVANCE],
        )

        assert len(scores) == 2
        assert scores[0].dimension == EvaluationDimension.ACCURACY
        assert scores[0].score == 0.85
        assert scores[1].dimension == EvaluationDimension.RELEVANCE
        assert reasoning == "Good response overall"
        assert aggregate == 0.875

    def test_parse_response_without_code_block(self) -> None:
        """Test parsing response without markdown code block."""
        judge = LLMJudge()

        response_text = """
        {
            "scores": [
                {"dimension": "accuracy", "score": 0.8, "reasoning": "OK", "confidence": 0.8}
            ],
            "overall_reasoning": "Acceptable",
            "aggregate_score": 0.8
        }
        """

        scores, reasoning, aggregate = judge._parse_judge_response(
            response_text,
            [EvaluationDimension.ACCURACY],
        )

        assert len(scores) == 1
        assert scores[0].score == 0.8

    def test_parse_adds_missing_dimensions(self) -> None:
        """Test that missing dimensions get default scores."""
        judge = LLMJudge()

        response_text = """
        {
            "scores": [
                {"dimension": "accuracy", "score": 0.9, "reasoning": "Good", "confidence": 0.9}
            ],
            "overall_reasoning": "Missing relevance",
            "aggregate_score": 0.9
        }
        """

        scores, _, _ = judge._parse_judge_response(
            response_text,
            [EvaluationDimension.ACCURACY, EvaluationDimension.RELEVANCE],
        )

        assert len(scores) == 2
        relevance_score = next(s for s in scores if s.dimension == EvaluationDimension.RELEVANCE)
        assert relevance_score.score == 0.5  # Default
        assert relevance_score.confidence == 0.0  # Low confidence

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON raises error."""
        judge = LLMJudge()

        with pytest.raises(ValueError) as exc_info:
            judge._parse_judge_response(
                "not valid json",
                [EvaluationDimension.ACCURACY],
            )
        assert "Failed to parse" in str(exc_info.value)

    def test_calculate_aggregate(self) -> None:
        """Test aggregate score calculation."""
        from src.evaluation.models import DimensionScore

        judge = LLMJudge()

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

        # Weighted average: (0.8 * 1.0 + 0.6 * 0.5) / (1.0 + 0.5) = 1.1 / 1.5 = 0.733
        aggregate = judge._calculate_aggregate(scores)
        assert abs(aggregate - 0.733) < 0.01

    def test_calculate_aggregate_empty(self) -> None:
        """Test aggregate with empty scores."""
        judge = LLMJudge()
        assert judge._calculate_aggregate([]) == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_mocked(self) -> None:
        """Test evaluate with mocked LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""
                {
                    "scores": [
                        {"dimension": "accuracy", "score": 0.9, "reasoning": "Correct", "confidence": 0.95}
                    ],
                    "overall_reasoning": "Good response",
                    "aggregate_score": 0.9
                }
                """
            )
        ]
        mock_llm.messages.create.return_value = mock_response

        judge = LLMJudge(llm=mock_llm)

        case = EvaluationCase(
            id="test",
            name="Test Case",
            input={"query": "test"},
            evaluation_criteria=[EvaluationDimension.ACCURACY],
        )

        scores, reasoning, aggregate = await judge.evaluate(case, {"answer": "test"})

        assert len(scores) == 1
        assert scores[0].score == 0.9
        assert reasoning == "Good response"
        mock_llm.messages.create.assert_called_once()
