from jiwer import process_words
from dataclasses import dataclass

# the dataclass is frozen so that once the metrics have been computed, they cannot be changed
@dataclass(frozen=True)
class WERResult:
    """
    Stores results from comparing the normalized reference transcript with the normalized hypothesis transcript.
    """

    wer: float              # (S + D + I)/N
    substitutions: int      # words that are incorrecly replaced with another word
    deletions: int          # words that are ommited 
    insertions: int         # words that are incorrectly added
    hits: int               # correctly recognised words
    n_reference_words: int    # total words in the reference transcript


def compute_wer(reference: str, hypothesis: str,) -> WERResult:
    """
    Compute Word Error Rate and the associated error counts for pre-normalized text (the function does not perform normalization)

    WER = (S + D + I) / N

    where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of words in the reference transcript
    """

    if reference is None:
        raise ValueError("Reference cannot be None.")

    if hypothesis is None:
        raise ValueError("Hypothesis cannot be None.")

    if not reference.strip():
        raise ValueError("Reference transcript cannot be empty.")

    # TODO: decide whether to keep this or not
    # could this be good to have as all deletions, extremely bad transcription
    # if not hypothesis.strip():
    #     raise ValueError("Hypothsis transcript cannot be empty.")

    result = process_words(reference, hypothesis)

    # the reference words (N) do not include insertions since they did not occur in the reference transcript
    reference_words = (result.hits+ result.substitutions+ result.deletions)

    return WERResult(
        wer = float(result.wer),
        substitutions = int(result.substitutions),
        deletions = int(result.deletions),
        insertions = int(result.insertions),
        hits = int(result.hits),
        n_reference_words = int(reference_words),
    )

