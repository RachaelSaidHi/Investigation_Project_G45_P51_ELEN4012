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
 
    def test_multiple_substitutions_together(self):
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

    def test_all_three_error_types_together(self):
        """
        Deletions and insertions only appear when the alignment is cheaper than substituting. 
        Here the anchors either side force one of each (S, I, D) rather than four substitutions.
        """

        # reference:  alpha bravo charlie delta echo            foxtrot
        # hypothesis: alpha       charlie delta epsilon golf    foxtrot
        #               H     D      H      H     S      I         H
        # Aligning positionally would cost 4 substitutions

        result = compute_wer(
            "alpha bravo charlie delta echo foxtrot",
            "alpha charlie delta epsilon golf foxtrot",
        )

        assert result.deletions == 1
        assert result.substitutions == 1
        assert result.insertions == 1
        assert result.hits == 4
        assert result.n_reference_words == 6
        assert result.wer == pytest.approx(3 / 6)


class TestReferenceWordCount:
 
    def test_n_comes_from_the_reference_not_the_hypothesis(self):
        """
        N must be the reference length. If insertions were included, corpus WER could not be summed across samples correctly.
        """
 
        result = compute_wer("one two three", "one two three four five")
 
        assert result.n_reference_words == 3
 
    def test_n_counts_deleted_words(self):
        """
        Deleted words were in the reference, so they still count towards N.
        """
 
        result = compute_wer("one two three four", "one two")
 
        assert result.n_reference_words == 4

    def test_n_on_a_long_reference(self):
        reference = " ".join(["word"] * 50)

        result = compute_wer(reference, reference)

        assert result.n_reference_words == 50
        assert result.hits == 50

    def test_n_matches_the_reference_word_count(self):
        """
        N is derived from the alignment (H + D + S) rather than counted directly.
        This checks that the dervied result does match the counted reference words, which they should for any reference that has already been normalised to single-spaced words.
        """

        reference = "the patient is stable and alert"

        result = compute_wer(reference, "the patient is unstable")

        assert result.n_reference_words == len(reference.split())


# run using python -m pytest tests/test_metrics.py -v




