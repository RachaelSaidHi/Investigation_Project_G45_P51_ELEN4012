"""
Tests for the metric computation functionality
"""

import pytest
from dataclasses import FrozenInstanceError

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
        """
        For this pair, the minimum-edit alignment contains three substitutions and no insertions or deletions.
        """

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

    def test_wer_matches_error_count_formula(self):
        result = compute_wer("alpha bravo charlie delta echo foxtrot", "alpha charlie delta epsilon golf foxtrot")

        expected_wer = (result.substitutions + result.deletions + result.insertions) / result.n_reference_words

        assert result.wer == pytest.approx(expected_wer)

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
        This checks that the derived result does match the counted reference words, which they should for any reference that has already been normalised to single-spaced words.
        """

        reference = "the patient is stable and alert"

        result = compute_wer(reference, "the patient is unstable")

        assert result.n_reference_words == len(reference.split())

    def test_reference_word_count_matches_alignment_components(self):
        result = compute_wer("one two three four five", "one six three",)

        assert result.n_reference_words == (result.hits + result.substitutions + result.deletions)


class TestBoundaryValues:
 
    def test_wer_can_exceed_one(self):
        """
        Insertions are not limited by the reference length, so a per-sample WER above 1.0 is valid and must not be clipped.
        """
 
        result = compute_wer("one", "one two three four five")

        assert result.hits == 1
        assert result.substitutions == 0
        assert result.deletions == 0
        assert result.insertions == 4
        assert result.wer > 1.0
        assert result.wer == pytest.approx(4.0)
 
    def test_completely_wrong_transcription_gives_wer_of_one(self):
        result = compute_wer("one two three", "four five six")
 
        assert result.substitutions == 3
        assert result.wer == pytest.approx(1.0)
 
    def test_empty_hypothesis_is_all_deletions(self):
        """
        An empty hypothesis is a real transcription outcome, not an error. Dropping these samples would remove the worst results from exactly the accents expected to perform worst.
        """
 
        result = compute_wer("the patient is stable", "")
 
        assert result.deletions == 4
        assert result.substitutions == 0
        assert result.insertions == 0
        assert result.wer == pytest.approx(1.0)
 
    def test_whitespace_only_hypothesis_is_all_deletions(self):
        result = compute_wer("the patient is stable", "   ")
 
        assert result.deletions == 4
        assert result.wer == pytest.approx(1.0)


class TestInvalidInput:
 
    def test_none_reference_fails(self):
        with pytest.raises(ValueError, match = "Reference cannot be None"):
            compute_wer(None, "something")
 
    def test_none_hypothesis_fails(self):
        with pytest.raises(ValueError, match = "Hypothesis cannot be None"):
            compute_wer("something", None)
 
    def test_empty_reference_fails(self):
        """
        WER divides by the reference length, so an empty reference has no defined value.
        """
 
        with pytest.raises(ValueError, match = "Reference transcript cannot be empty"):
            compute_wer("", "something")
 
    def test_whitespace_only_reference_fails(self):
        with pytest.raises(ValueError, match = "Reference transcript cannot be empty"):
            compute_wer("   ", "something")


class TestContract:
 
    def test_result_is_immutable(self):
        result = compute_wer("one two", "one two")
 
        with pytest.raises(FrozenInstanceError):
            result.wer = 0.5
 
    def test_returns_a_wer_result(self):
        result = compute_wer("one two", "one two")
 
        assert isinstance(result, WERResult)
 
    def test_counts_are_integers(self):
        result = compute_wer("one two three", "one four")
 
        assert isinstance(result.substitutions, int)
        assert isinstance(result.deletions, int)
        assert isinstance(result.insertions, int)
        assert isinstance(result.hits, int)
        assert isinstance(result.n_reference_words, int)
 
    def test_wer_is_a_float(self):
        result = compute_wer("one two", "one two")
 
        assert isinstance(result.wer, float)
 
    def test_case_differences_are_counted_as_errors(self):
        """
        compute_wer expects already-normalised text, where lowercasing is part of normalization.
        """

        result = compute_wer("the patient", "The patient")

        assert result.substitutions == 1

    def test_number_words_and_digits_are_counted_as_errors(self):
        """
        "six" and "6" only match after normalization. This is why a having only a lowercase and punctuation stripping normalizer could inflate WER.
        """

        result = compute_wer("six spoons", "6 spoons")

        assert result.substitutions == 1

    def test_punctuation_is_counted_as_errors(self):
        """
        Punctuation symbols are stripped by normalization, not here.
        """

        result = compute_wer("stable alert", "stable, alert.")

        assert result.substitutions == 2

    def test_spoken_punctuation_words_are_counted_as_errors(self):
        """
        The AfriSpeech punctuation correction happens in normalization. Without it, a spoken comma is an insertion.
        """

        result = compute_wer("stable alert", "stable comma alert")

        assert result.insertions == 1


# run using python -m pytest tests/test_metrics.py -v




