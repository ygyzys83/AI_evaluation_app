import pytest
from unittest.mock import patch, MagicMock
from evaluate_results import grade_entry, _extract_json


# ── _extract_json tests ───────────────────────────────────────────────────────
# These test the cleaning function directly — no mocking needed
# because _extract_json is pure deterministic logic.

class TestExtractJson:

    def test_plain_json_passes_through_unchanged(self):
        raw = '{"verdict": "PASS", "similarity_score": 4, "reasoning": "Correct."}'
        assert _extract_json(raw) == raw

    def test_strips_json_code_fence(self):
        raw = '```json\n{"verdict": "PASS", "similarity_score": 4, "reasoning": "Correct."}\n```'
        result = _extract_json(raw)
        assert "```" not in result
        assert '"verdict"' in result

    def test_strips_plain_code_fence(self):
        raw = '```\n{"verdict": "FAIL", "similarity_score": 2, "reasoning": "Wrong."}\n```'
        result = _extract_json(raw)
        assert "```" not in result

    def test_handles_extra_whitespace(self):
        raw = '   {"verdict": "PASS", "similarity_score": 5, "reasoning": "Perfect."}   '
        result = _extract_json(raw)
        assert result.startswith("{")


# ── grade_entry tests ─────────────────────────────────────────────────────────
# These mock ollama.generate so no model needs to be running.

# A reusable sample entry — what a row from multi_model_results.json looks like
SAMPLE_ENTRY = {
    "id": 1,
    "model_used": "qwen2.5",
    "provider": "ollama",
    "question": "Which team won the Magic vs Pistons game on April 19, 2026?",
    "ground_truth": "The Magic won 112-101.",
    "llm_answer": "The Magic won the game 112-101.",
    "latency": 1.23,
    "char_count": 31,
    "status": "SUCCESS"
}


def make_mock_response(verdict: str, score: int, reasoning: str) -> MagicMock:
    """Helper: builds a fake ollama.generate() return value."""
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: (
        f'{{"verdict": "{verdict}", "similarity_score": {score}, "reasoning": "{reasoning}"}}'
        if key == "response" else None
    )
    return mock


class TestGradeEntry:

    def test_pass_verdict_is_stored_correctly(self):
        entry = SAMPLE_ENTRY.copy()
        mock_response = make_mock_response("PASS", 5, "Answer matches ground truth exactly.")

        with patch("evaluate_results.ollama.generate", return_value=mock_response):
            result = grade_entry(entry)

        assert result["grade"] == "PASS"
        assert result["similarity_score"] == 5
        assert result["reasoning"] == "Answer matches ground truth exactly."

    def test_fail_verdict_is_stored_correctly(self):
        entry = SAMPLE_ENTRY.copy()
        entry["llm_answer"] = "The Pistons won."
        mock_response = make_mock_response("FAIL", 1, "Answer names the wrong team.")

        with patch("evaluate_results.ollama.generate", return_value=mock_response):
            result = grade_entry(entry)

        assert result["grade"] == "FAIL"
        assert result["similarity_score"] == 1

    def test_empty_response_returns_error_grade(self):
        """Model returns empty string — should not crash, should return ERROR grade."""
        entry = SAMPLE_ENTRY.copy()
        mock = MagicMock()
        mock.__getitem__ = lambda self, key: "" if key == "response" else None

        with patch("evaluate_results.ollama.generate", return_value=mock):
            result = grade_entry(entry)

        assert result is not None
        assert result["grade"] == "ERROR"
        assert result["similarity_score"] == 0

    def test_invalid_verdict_value_returns_error_grade(self):
        """Model returns a verdict value outside PASS/FAIL — Pydantic should catch it."""
        entry = SAMPLE_ENTRY.copy()
        mock = MagicMock()
        mock.__getitem__ = lambda self, key: (
            '{"verdict": "MAYBE", "similarity_score": 3, "reasoning": "Not sure."}'
            if key == "response" else None
        )

        with patch("evaluate_results.ollama.generate", return_value=mock):
            result = grade_entry(entry)

        assert result["grade"] == "ERROR"

    def test_similarity_score_out_of_range_returns_error_grade(self):
        """Model returns similarity_score of 7 — Pydantic ge/le constraints catch it."""
        entry = SAMPLE_ENTRY.copy()
        mock = MagicMock()
        mock.__getitem__ = lambda self, key: (
            '{"verdict": "PASS", "similarity_score": 7, "reasoning": "Great."}'
            if key == "response" else None
        )

        with patch("evaluate_results.ollama.generate", return_value=mock):
            result = grade_entry(entry)

        assert result["grade"] == "ERROR"

    def test_markdown_fences_are_handled(self):
        """Model wraps JSON in markdown fences — should still parse correctly."""
        entry = SAMPLE_ENTRY.copy()
        mock = MagicMock()
        mock.__getitem__ = lambda self, key: (
            '```json\n{"verdict": "PASS", "similarity_score": 4, "reasoning": "Correct."}\n```'
            if key == "response" else None
        )

        with patch("evaluate_results.ollama.generate", return_value=mock):
            result = grade_entry(entry)

        assert result["grade"] == "PASS"
        assert result["similarity_score"] == 4

    def test_original_entry_fields_are_preserved(self):
        """Grading should add fields, not overwrite existing ones."""
        entry = SAMPLE_ENTRY.copy()
        mock_response = make_mock_response("PASS", 5, "Correct.")

        with patch("evaluate_results.ollama.generate", return_value=mock_response):
            result = grade_entry(entry)

        assert result["model_used"] == "qwen2.5"
        assert result["latency"] == 1.23
        assert result["question"] == SAMPLE_ENTRY["question"]