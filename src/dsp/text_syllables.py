"""
Count the syllables in a transcript.

Used to compute speaking rate from the text a speaker read rather than from
acoustic syllable detection. For the Speech Accent Archive, where everyone
reads the same passage, this gives the true syllable count. For AfriSpeech it
is an estimate, because speakers may expand abbreviations or misread words.

Each word is counted in one of three ways, in order of preference:
- dictionary: looked up in the CMU Pronouncing Dictionary (vowel phonemes
  carry a stress digit, so syllables = phonemes ending in a digit);
- spelled: words with no vowels (e.g. "sbp", "mg") are assumed to be read
  letter by letter ("w" = 3 syllables, other letters 1);
- estimated: other unknown words (names, rare medical terms) use a simple
  vowel-group rule.

Numbers are written out first ("180" -> "one hundred and eighty") and "%"
becomes "percent". Punctuation marks are ignored, even though some AfriSpeech
speakers read punctuation aloud ("comma"); this is a known limitation.

The counts of spelled and estimated words are returned so that samples whose
count relies heavily on guesses can be flagged or excluded.
"""

import re
from functools import lru_cache

import cmudict 
from num2words import num2words


VOWELS = set("aeiouy")


@lru_cache(maxsize=1)
def _pronunciations() -> dict:
    """
    Load the CMU Pronouncing Dictionary once.
    """

    return cmudict.dict()


def _dictionary_syllables(word: str) -> int | None:
    """
    Syllables from the CMU dictionary (first pronunciation), or None if absent.
    """

    pronunciations = _pronunciations().get(word)

    if not pronunciations:
        return None

    return sum(phoneme[-1].isdigit() for phoneme in pronunciations[0])


def _estimated_syllables(word: str) -> int:
    """
    Rough syllable count for an unknown word: groups of vowels, minus a silent final e.
    """

    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)

    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1

    return max(count, 1)


def word_syllables(word: str) -> tuple[int, str]:
    """
    Return (syllable count, method) for one lowercase word.

    method is "dictionary", "spelled" or "estimated".
    """

    count = _dictionary_syllables(word)

    if count is None and word.endswith("'s"):
        count = _dictionary_syllables(word[:-2])

    if count is not None:
        return count, "dictionary"

    if not VOWELS & set(word):
        return sum(3 if letter == "w" else 1 for letter in word), "spelled"

    return _estimated_syllables(word), "estimated"


def _normalise(text: str) -> str:
    """
    Lowercase, write out numbers and symbols, and separate letters from digits.
    """

    text = text.lower()
    text = text.replace("%", " percent ").replace("&", " and ")

    # "fio2" -> "fio 2", "61cc" -> "61 cc"
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)

    def spell_number(match: re.Match) -> str:
        return f" {num2words(float(match.group()) if '.' in match.group() else int(match.group()))} "

    text = re.sub(r"\d+(?:\.\d+)?", spell_number, text)

    # num2words produces hyphens ("forty-two"); treat them as word breaks
    return text.replace("-", " ")


def count_text_syllables(text: str) -> dict[str, int]:
    """
    Count syllables in a transcript.

    Returns a dict with the total syllables and how many words were counted
    by each method.
    """

    words = re.findall(r"[a-z]+(?:'[a-z]+)?", _normalise(text))

    result = {
        "syllables": 0,
        "words": len(words),
        "dictionary_words": 0,
        "spelled_words": 0,
        "estimated_words": 0,
    }

    for word in words:
        count, method = word_syllables(word)
        result["syllables"] += count
        result[f"{method}_words"] += 1

    return result