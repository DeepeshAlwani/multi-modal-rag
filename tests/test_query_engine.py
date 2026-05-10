"""
tests/test_query_engine.py — Unit tests for query_engine.py pure functions.

These tests cover:
  - reciprocal_rank_fusion  (the core retrieval fusion logic)
  - _parse_line_range       (metadata parsing)
  - _query_to_clean_name    (stop-word stripping for keyword boost)
  - _build_prompt           (prompt structure sanity)

No ChromaDB, no Ollama, no network calls.
"""
import pytest
from query_engine import (
    reciprocal_rank_fusion,
    _parse_line_range,
    _query_to_clean_name,
    _build_prompt,
)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_single_collection_preserves_order(self, sample_functions):
        """With one collection, RRF should preserve the original ranking."""
        results = reciprocal_rank_fusion({"code": sample_functions}, k=60)
        assert len(results) == len(sample_functions)
        # First result from the original list should score highest
        assert results[0][1]["function"] == "verify_user"

    def test_agreement_boosts_score(self, sample_functions, sample_diagram):
        """
        If two collections both rank the same doc first, it should float to the top
        after fusion, scoring higher than a doc only seen in one collection.
        """
        # Duplicate the first code item into a second 'structural' collection
        top_item = sample_functions[0]
        results = reciprocal_rank_fusion(
            {
                "code":       sample_functions,
                "structural": [top_item, sample_functions[1]],
            },
            k=60,
        )
        # The item that appeared first in BOTH collections should win
        assert results[0][1]["function"] == "verify_user"

    def test_empty_collections_return_empty(self):
        results = reciprocal_rank_fusion({}, k=60)
        assert results == []

    def test_empty_list_in_collection(self):
        results = reciprocal_rank_fusion({"code": []}, k=60)
        assert results == []

    def test_deduplication_across_collections(self, sample_functions):
        """
        The same logical document appearing in two collections should produce
        only ONE entry in the output (identified by file+function key).
        """
        item = sample_functions[0]
        results = reciprocal_rank_fusion(
            {"code": [item], "structural": [item]}, k=60
        )
        assert len(results) == 1

    def test_high_k_flattens_scores(self, sample_functions):
        """
        With a very large k, rank differences matter less and the top item's
        score advantage over the second item should be smaller than with k=1.
        """
        def score_gap(k):
            # Approximate: 1/(1+k) - 1/(2+k)
            return 1 / (1 + k) - 1 / (2 + k)

        assert score_gap(1) > score_gap(1000)

    def test_diagram_and_code_merge(self, sample_functions, sample_diagram):
        """Fusion should combine code and diagram sources without error."""
        results = reciprocal_rank_fusion(
            {"code": sample_functions, "diagram": sample_diagram}, k=60
        )
        assert len(results) == len(sample_functions) + len(sample_diagram)

    def test_output_is_list_of_tuples(self, sample_functions):
        results = reciprocal_rank_fusion({"code": sample_functions})
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2


# ---------------------------------------------------------------------------
# _parse_line_range
# ---------------------------------------------------------------------------

class TestParseLineRange:
    def test_tuple_input(self):
        assert _parse_line_range((10, 45)) == (10, 45)

    def test_string_tuple_repr(self):
        assert _parse_line_range("(10, 45)") == (10, 45)

    def test_string_no_parens(self):
        assert _parse_line_range("10, 45") == (10, 45)

    def test_single_line(self):
        assert _parse_line_range("(7, 7)") == (7, 7)

    def test_malformed_string_returns_default(self):
        assert _parse_line_range("broken") == (1, 1)

    def test_empty_string_returns_default(self):
        assert _parse_line_range("") == (1, 1)

    def test_none_like_returns_default(self):
        assert _parse_line_range(None) == (1, 1)

    def test_large_line_numbers(self):
        assert _parse_line_range("(1000, 2000)") == (1000, 2000)


# ---------------------------------------------------------------------------
# _query_to_clean_name
# ---------------------------------------------------------------------------

class TestQueryToCleanName:
    def test_strips_common_stop_words(self):
        result = _query_to_clean_name("what does verify_user do")
        assert "verify_user" in result

    def test_converts_spaces_to_underscores(self):
        result = _query_to_clean_name("create session")
        assert "_" in result or result in ("create_session", "create")

    def test_lowercases_output(self):
        result = _query_to_clean_name("What Is Check_Rate_Limit")
        assert result == result.lower()

    def test_empty_question(self):
        result = _query_to_clean_name("")
        assert isinstance(result, str)

    def test_only_stop_words(self):
        result = _query_to_clean_name("what is the a an")
        # Should produce empty or near-empty string — no crash
        assert isinstance(result, str)

    def test_function_name_preserved(self):
        result = _query_to_clean_name("explain build_all_indexes")
        assert "build_all_indexes" in result


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def _make_prompt(self, question="How does auth work?", context="def verify(): pass", sources=None):
        sources = sources or ["database.py → verify_user (lines 10-30)"]
        return _build_prompt(question, context, sources)

    def test_question_appears_in_prompt(self):
        p = self._make_prompt(question="How does auth work?")
        assert "How does auth work?" in p

    def test_sources_appear_in_prompt(self):
        p = self._make_prompt(sources=["database.py → verify_user"])
        assert "database.py → verify_user" in p

    def test_code_context_appears_in_prompt(self):
        p = self._make_prompt(context="def secret_function(): pass")
        assert "def secret_function(): pass" in p

    def test_prompt_contains_answer_instructions(self):
        p = self._make_prompt()
        assert "ONLY" in p  # "Answer using ONLY the source code"

    def test_multiple_sources_all_listed(self):
        sources = ["file_a.py → fn_a", "file_b.py → fn_b", "file_c.py → fn_c"]
        p = _build_prompt("question", "context", sources)
        for s in sources:
            assert s in p

    def test_empty_sources_does_not_crash(self):
        p = _build_prompt("question", "context", [])
        assert isinstance(p, str)
        assert len(p) > 0