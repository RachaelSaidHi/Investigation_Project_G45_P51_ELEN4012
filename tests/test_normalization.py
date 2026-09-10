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


def test_single_word_phrase_is_removed():
    text, removed = strip_punctuation("hello comma i am here")

    assert text == "hello i am here"
    assert removed == 1


def test_multi_word_phrase_is_removed():
    text, removed = strip_punctuation("i am here full stop")

    assert text == "i am here"
    assert removed == 1


def test_phrase_inside_longer_word_is_not_removed():
    text, removed = strip_punctuation("patient had a colonoscopy")

    assert text == "patient had a colonoscopy"
    assert removed == 0


def test_adjacent_phrases_are_both_removed():
    text, removed = strip_punctuation("hello comma comma there")

    assert text == "hello there"
    assert removed == 2


def test_longer_phrase_matched_before_shorter_one():
    text, removed = strip_punctuation("results semi colon next")

    assert text == "results next"
    assert removed == 1


def test_text_without_phrases_is_unchanged():
    text, removed = strip_punctuation("the patient is stable")

    assert text == "the patient is stable"
    assert removed == 0


def test_empty_text_is_handled():
    text, removed = strip_punctuation("")

    assert text == ""
    assert removed == 0


def test_duplicate_whitespace_is_collapsed():
    assert remove_duplicate_whitespace("  a   b  c ") == "a b c"


def test_spoken_and_written_punctuation_normalise_identically():
    """
    The three ways Whisper can render a spoken comma must all match.
    """

    written, _ = normalize_text("stable, alert", "afrispeech")
    spoken, _ = normalize_text("stable comma alert", "afrispeech")
    both, _ = normalize_text("stable, comma, alert", "afrispeech")

    assert written == spoken
    assert spoken == both
    assert "comma" not in written


def test_number_words_and_digits_normalise_identically():
    words, _ = normalize_text("six spoons", "afrispeech")
    digits, _ = normalize_text("6 spoons", "afrispeech")

    assert words == digits


def test_british_spelling_is_normalised():
    """
    Canary for the spelling mapping. Fails if the mapping is not loaded.
    """

    text, _ = normalize_text("the colour of the centre", "afrispeech")

    assert "color" in text
    assert "center" in text


def test_homograph_is_removed_from_both_sides():
    """
    A real word matching a phrase is removed symmetrically, so it can never create an error.
    """

    reference, _ = normalize_text("inflammation of the colon", "afrispeech")
    hypothesis, _ = normalize_text("inflammation of the colon", "afrispeech")

    assert reference == hypothesis


def test_punctuation_is_not_removed_for_other_datasets():
    text, removed = normalize_text("stable comma alert", "speech_accent_archive")

    assert "comma" in text
    assert removed == 0


def test_dataset_name_is_case_and_whitespace_insensitive():
    lower, _ = normalize_text("stable comma alert", "afrispeech")
    messy, _ = normalize_text("stable comma alert", "  AfriSpeech ")

    assert lower == messy


def test_none_text_fails():
    with pytest.raises(ValueError, match="Cannot normalize None text"):
        normalize_text(None, "afrispeech")


def test_unknown_dataset_fails():
    with pytest.raises(ValueError, match="Unknown dataset"):
        normalize_text("stable alert", "afrispeach")