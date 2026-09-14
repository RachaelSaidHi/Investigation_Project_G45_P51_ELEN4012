"""
Tests for the metric computation functionality
"""

import pytest

from src.asr.metrics import WERResult, compute_wer

class TestCorrectTranscription:
 
    def test_identical_text_gives_zero_wer(self):
        result = compute_wer("the patient is stable", "the patient is stable")
 
        assert result.wer == 0.0
        assert result.substitutions == 0
        assert result.deletions == 0
        assert result.insertions == 0
        assert result.hits == 4
        assert result.n_reference_words == 4
 
    def test_single_word_reference(self):
        result = compute_wer("stable", "stable")
 
        assert result.wer == 0.0
        assert result.n_reference_words == 1
 
    def test_extra_whitespace_does_not_create_errors(self):
        """
        Words are split on whitespace, so irregular spacing is ignored.
        """
 
        result = compute_wer("the patient is       stable", "the  patient   is stable")
 
        assert result.wer == 0.0
        assert result.n_reference_words == 4


class TestErrorCounts:
 
    def test_substitution_is_counted(self):
        result = compute_wer("the patient is stable", "the patient is unstable")
 
        assert result.substitutions == 1
        assert result.deletions == 0
        assert result.insertions == 0
        assert result.n_reference_words == 4
        assert result.hits == 3
        assert result.wer == pytest.approx(1 / 4)
 
    def test_deletion_is_counted(self):
        result = compute_wer("the patient is stable", "the patient stable")
 
        assert result.deletions == 1
        assert result.substitutions == 0
        assert result.insertions == 0
        assert result.n_reference_words == 4
        assert result.hits == 3
        assert result.wer == pytest.approx(1 / 4)
 
    def test_insertion_is_counted(self):
        result = compute_wer("the patient is stable", "the patient is very stable")
 
        assert result.insertions == 1
        assert result.substitutions == 0
        assert result.deletions == 0
        assert result.n_reference_words == 4
        assert result.hits == 4
        assert result.wer == pytest.approx(1 / 4)
 
    def test_all_three_error_types_together(self):
        # jiwer aligns positionally here, so two->xxx, four->five and
        # five->six are all counted as substitutions.
        result = compute_wer("one two three four five", "one xxx three five six")
 
        assert result.substitutions == 3
        assert result.deletions == 0
        assert result.insertions == 0
        assert result.n_reference_words == 5
        assert result.hits == 2
        assert result.wer == pytest.approx(3 / 5)
 
    def test_hits_are_counted(self):
        result = compute_wer("one two three four", "one two three five")
 
        assert result.hits == 3


# run using python -m pytest tests/test_metrics.py -v




