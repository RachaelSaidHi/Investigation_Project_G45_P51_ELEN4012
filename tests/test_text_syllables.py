"""
Tests for transcript syllable counting used by the reference speech rate.

Method-selection tests replace the CMU dictionary with a small fake one, so
they do not depend on the installed cmudict version. A few sanity checks use
the real dictionary for common words.
"""

import pytest

import src.dsp.text_syllables as text_syllables
from src.dsp.text_syllables import (
    _estimated_syllables,
    _normalise,
    count_text_syllables,
    word_syllables,
)


# CMU-style pronunciations: vowel phonemes end in a stress digit.
FAKE_DICTIONARY = {
    "please": [["P", "L", "IY1", "Z"]],
    "call": [["K", "AO1", "L"]],
    "stella": [["S", "T", "EH1", "L", "AH0"]],
    "banana": [
        ["B", "AH0", "N", "AE1", "N", "AH0"],
        ["B", "AH0", "N", "AA1", "N"],
    ],
    "doctor": [["D", "AA1", "K", "T", "ER0"]],
    "mg": [["EH1", "M", "JH", "IY1"]],
}


@pytest.fixture
def fake_dictionary(monkeypatch):
    """Replace the CMU dictionary with FAKE_DICTIONARY."""

    monkeypatch.setattr(
        text_syllables,
        "_pronunciations",
        lambda: FAKE_DICTIONARY,
    )


class TestEstimatedSyllables:

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("zorb", 1),
            ("zorbaf", 2),
            ("flarbe", 1),       # silent final e removed
            ("flarble", 2),      # -le is pronounced
            ("glee", 1),         # -ee is not silent
            ("glarye", 2),       # -ye is not silent
            ("queue", 1),        # one vowel group
            ("strz", 1),         # never below one
            ("e", 1),
        ],
    )
    def test_vowel_group_rule(self, word, expected):
        assert _estimated_syllables(word) == expected

    def test_y_counts_as_a_vowel(self):
        assert _estimated_syllables("xyzzyq") == 2


class TestWordSyllables:

    def test_dictionary_word_counts_stress_digits(self, fake_dictionary):
        assert word_syllables("stella") == (2, "dictionary")

    def test_first_pronunciation_is_used(self, fake_dictionary):
        assert word_syllables("banana") == (3, "dictionary")

    def test_possessive_falls_back_to_base_word(self, fake_dictionary):
        assert word_syllables("doctor's") == (2, "dictionary")

    def test_dictionary_abbreviation_is_not_spelled(self, fake_dictionary):
        assert word_syllables("mg") == (2, "dictionary")

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("sbp", 3),
            ("bp", 2),
            ("hcl", 3),
            ("w", 3),            # "double-u"
            ("www", 9),
            ("tbsp", 4),
        ],
    )
    def test_words_without_vowels_are_spelled(self, fake_dictionary, word, expected):
        assert word_syllables(word) == (expected, "spelled")

    def test_unknown_word_with_vowels_is_estimated(self, fake_dictionary):
        assert word_syllables("ibuprofen") == (4, "estimated")

    def test_unknown_possessive_is_estimated(self, fake_dictionary):
        count, method = word_syllables("zorbaf's")

        assert method == "estimated"
        assert count >= 1


class TestNormalise:

    def test_text_is_lowercased(self):
        assert _normalise("Please CALL") == "please call"

    def test_integers_are_written_out(self):
        assert _normalise("42").split() == ["forty", "two"]

    def test_hyphens_from_numbers_become_word_breaks(self):
        assert "-" not in _normalise("42")

    def test_decimals_are_written_out(self):
        assert _normalise("3.5").split() == ["three", "point", "five"]

    def test_percent_and_ampersand_are_written_out(self):
        assert _normalise("50%").split() == ["fifty", "percent"]
        assert _normalise("a & b").split() == ["a", "and", "b"]

    def test_letters_and_digits_are_separated(self):
        assert _normalise("fio2").split() == ["fio", "two"]
        assert _normalise("61cc").split() == ["sixty", "one", "cc"]

    def test_hyphenated_words_are_split(self):
        assert _normalise("follow-up").split() == ["follow", "up"]


class TestCountTextSyllables:

    def test_result_has_expected_keys(self, fake_dictionary):
        result = count_text_syllables("please call stella")

        assert set(result) == {
            "syllables",
            "words",
            "dictionary_words",
            "spelled_words",
            "estimated_words",
        }

    def test_dictionary_words_are_summed(self, fake_dictionary):
        result = count_text_syllables("Please call Stella.")

        assert result == {
            "syllables": 4,
            "words": 3,
            "dictionary_words": 3,
            "spelled_words": 0,
            "estimated_words": 0,
        }

    def test_methods_are_counted_separately(self, fake_dictionary):
        result = count_text_syllables("please check sbp zorbaf")

        # "check" is not in the fake dictionary, so it is estimated.
        assert result["words"] == 4
        assert result["dictionary_words"] == 1
        assert result["spelled_words"] == 1
        assert result["estimated_words"] == 2
        assert result["syllables"] == 1 + 1 + 3 + 2

    def test_method_counts_add_up_to_word_count(self, fake_dictionary):
        result = count_text_syllables("Doctor's 2 mg, sbp & zorbaf!")

        assert (
            result["dictionary_words"]
            + result["spelled_words"]
            + result["estimated_words"]
            == result["words"]
        )

    def test_punctuation_is_ignored(self, fake_dictionary):
        with_punctuation = count_text_syllables("please, call: stella!")
        without = count_text_syllables("please call stella")

        assert with_punctuation == without

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "!!! ... ,,,",
        ],
    )
    def test_text_without_words_counts_zero(self, text):
        result = count_text_syllables(text)

        assert result["words"] == 0
        assert result["syllables"] == 0

    def test_possessive_is_one_word(self, fake_dictionary):
        result = count_text_syllables("the doctor's")

        assert result["words"] == 2


class TestRealDictionary:

    def test_common_words_are_found_in_cmu_dictionary(self):
        result = count_text_syllables("Please call Stella")

        assert result["syllables"] == 4
        assert result["dictionary_words"] == 3

    def test_numbers_are_counted_from_written_form(self):
        # "one hundred and eighty"
        result = count_text_syllables("180")

        assert result["words"] == 4
        assert result["syllables"] == 6

    def test_dictionary_is_loaded_once(self):
        text_syllables._pronunciations.cache_clear()

        count_text_syllables("please")
        count_text_syllables("call")

        assert text_syllables._pronunciations.cache_info().misses == 1


# run using:
# python -m pytest tests/test_text_syllables.py -v
