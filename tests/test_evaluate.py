"""
tests/test_evaluate.py — Unit tests for evaluate.py pure functions.

Covers the JSON extraction logic that is critical for the LLM judge pipeline.
No network calls, no Ollama, no OpenRouter.
"""
import pytest
from evaluate import extract_json


class TestExtractJson:
    """
    extract_json() must handle every messy response format that
    Nemotron / Qwen3 / DeepSeek can produce.
    """

    def test_clean_json(self):
        result = extract_json('{"faithful": true}')
        assert result == {"faithful": True}

    def test_json_with_leading_whitespace(self):
        result = extract_json('   \n{"relevancy": 0.8}\n')
        assert result.get("relevancy") == pytest.approx(0.8)

    def test_json_inside_markdown_fence(self):
        raw = '```json\n{"precision": 0.5}\n```'
        result = extract_json(raw)
        assert result.get("precision") == pytest.approx(0.5)

    def test_json_after_reasoning_text(self):
        """Nemotron often produces reasoning text before the JSON object."""
        raw = "Let me think about this... The answer is clearly faithful.\n{\"faithful\": true}"
        result = extract_json(raw)
        assert result.get("faithful") is True

    def test_think_tags_stripped(self):
        """DeepSeek and Qwen3 wrap reasoning in <think>...</think>."""
        raw = "<think>This is my reasoning step.</think>{\"relevancy\": 0.9}"
        result = extract_json(raw)
        assert result.get("relevancy") == pytest.approx(0.9)

    def test_multiple_json_objects_returns_last(self):
        """When there are multiple JSON blobs, the LAST one is the answer."""
        raw = '{"debug": "intermediate"} some text {"precision": 0.7}'
        result = extract_json(raw)
        assert result.get("precision") == pytest.approx(0.7)

    def test_totally_invalid_returns_empty_dict(self):
        result = extract_json("ERROR — model timed out")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = extract_json("")
        assert result == {}

    def test_nested_json_object(self):
        """Nested objects should parse correctly."""
        raw = '{"scores": {"f": 0.9, "r": 0.8}, "faithful": true}'
        result = extract_json(raw)
        assert result.get("faithful") is True

    def test_faithful_false(self):
        result = extract_json('{"faithful": false}')
        assert result.get("faithful") is False

    def test_relevancy_zero(self):
        result = extract_json('{"relevancy": 0.0}')
        assert result.get("relevancy") == pytest.approx(0.0)

    def test_relevancy_one(self):
        result = extract_json('{"relevancy": 1.0}')
        assert result.get("relevancy") == pytest.approx(1.0)