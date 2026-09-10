"""
Tests for the normaliser
"""

import pytest

from src.asr.normalize import (
    AFRISPEECH_PUNCTUATION_TERMS,
    normalize_text,
    remove_duplicate_whitespace,
    remove_exact_phrases,
)


def strip_punctuation(text):
    """
    Run the phrase removal with the AfriSpeech terms.
    """

    return remove_exact_phrases(
        text,
        AFRISPEECH_PUNCTUATION_TERMS,
    )


class TestRemoveExactPhrases:

    def test_single_word_phrase_is_removed(self):
        text, removed = strip_punctuation("hello comma i am here")
 
        assert text == "hello i am here"
        assert removed == 1
 
    def test_multi_word_phrase_is_removed(self):
        text, removed = strip_punctuation("i am here full stop")
 
        assert text == "i am here"
        assert removed == 1
 
    def test_phrase_inside_longer_word_is_not_removed(self):
        text, removed = strip_punctuation("patient had a colonoscopy")
 
        assert text == "patient had a colonoscopy"
        assert removed == 0
 
    def test_adjacent_phrases_are_both_removed(self):
        text, removed = strip_punctuation("hello comma comma there")
 
        assert text == "hello there"
        assert removed == 2
 
    def test_repeated_non_adjacent_phrases_are_counted(self):
        text, removed = strip_punctuation("a comma b comma c")
 
        assert text == "a b c"
        assert removed == 2
 
    def test_longer_phrase_matched_before_shorter_one(self):
        text, removed = strip_punctuation("results semi colon next")
 
        assert text == "results next"
        assert removed == 1
 
    def test_phrase_at_start_of_text(self):
        text, removed = strip_punctuation("comma hello there")
 
        assert text == "hello there"
        assert removed == 1
 
    def test_text_that_is_entirely_a_phrase(self):
        text, removed = strip_punctuation("full stop")
 
        assert text == ""
        assert removed == 1
 
    def test_text_without_phrases_is_unchanged(self):
        text, removed = strip_punctuation("the patient is stable")
 
        assert text == "the patient is stable"
        assert removed == 0
 
    def test_empty_text_is_handled(self):
        text, removed = strip_punctuation("")
 
        assert text == ""
        assert removed == 0


class TestRemoveDuplicateWhitespace:
    def test_repeated_whitespace_is_collapsed(self):
        assert remove_duplicate_whitespace("a   b  c") == "a b c"
 
    def test_leading_and_trailing_whitespace_is_removed(self):
        assert remove_duplicate_whitespace("  a b  ") == "a b"
 
    def test_empty_text_is_handled(self):
        assert remove_duplicate_whitespace("   ") == ""


class TestNormalizeText:
    def test_spoken_and_written_punctuation_normalise_identically(self):
        """
        The three ways Whisper can render a spoken comma must all match.
        """
 
        written, _ = normalize_text("stable, alert", "afrispeech")
        spoken, _ = normalize_text("stable comma alert", "afrispeech")
        both, _ = normalize_text("stable, comma, alert", "afrispeech")
 
        assert written == spoken
        assert spoken == both
        assert "comma" not in written
 
    def test_number_words_and_digits_normalise_identically(self):
        words, _ = normalize_text("six spoons", "afrispeech")
        digits, _ = normalize_text("6 spoons", "afrispeech")
 
        assert words == digits
 
    def test_british_spelling_is_normalised(self):
        """
        Canary for the spelling mapping. Fails if the mapping is not loaded.
        """
 
        text, _ = normalize_text("the colour of the centre", "afrispeech")
 
        assert "color" in text
        assert "center" in text

    def test_whisper_normaliser_handles_case_and_contractions(self):
        """
        Tests the Stage 1 behaviours the pipeline depends on, so a library change that removed them can be caught.
        """

        text, _ = normalize_text("She DOESN'T have Six spoons", "afrispeech")

        assert text == "she does not have 6 spoons"
 
    def test_homograph_is_removed_from_both_sides(self):
        """
        A real word matching a phrase is removed symmetrically, so it can never create an error.
        """
 
        reference, _ = normalize_text("inflammation of the colon", "afrispeech")
        hypothesis, _ = normalize_text("inflammation of the colon", "afrispeech")

        assert reference == hypothesis == "inflammation of the"
 
    def test_punctuation_is_not_removed_for_other_datasets(self):
        text, removed = normalize_text("stable comma alert", "speech_accent_archive")
 
        assert "comma" in text
        assert removed == 0
 
    def test_dataset_name_is_case_and_whitespace_insensitive(self):
        lower, _ = normalize_text("stable comma alert", "afrispeech")
        messy, _ = normalize_text("stable comma alert", "  AfriSpeech ")
 
        assert lower == messy
 
    def test_none_text_fails(self):
        with pytest.raises(ValueError, match="Cannot normalize None text"):
            normalize_text(None, "afrispeech")
 
    def test_unknown_dataset_fails(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            normalize_text("stable alert", "afrispeach")