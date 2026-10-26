"""
Reference-transcript speech-rate and pause features.

Uses the ground-truth reference transcript for the syllable count and the
project-wide shared speech mask for timing:

- first-to-last speech span,
- active-speech duration,
- pause timing

Whisper's transcription is not used either: its errors would feed into the
rate, making any correlation with WER circular.

Speaking rate:
    reference syllables / first-to-last speech span

Articulation rate:
    reference syllables / total speech-active time

    
The shared speech mask removes leading/trailing non-speech from the timing
span while retaining pauses that occur between detected speech regions.
Pauses are every gap between speech-mask intervals, so the shortest pause
counted is set by the mask's max_gap_s.
"""

import numpy as np

from src.dsp.speech_mask import speech_intervals
from src.dsp.text_syllables import count_text_syllables


FEATURE_NAME = "speech_rate_reference"
FEATURE_VERSION = "1"


SPEECH_EXCLUSION_MESSAGES = (
    "Insufficient speech",
    "Audio contains only digital silence",
)


class NoUsableSpeech(ValueError):
    """
    Deterministic exclusion because no usable speech was detected.
    """


def feature_columns(config: dict) -> list[str]:
    return [
        "rate_ref_syllables",
        "rate_ref_guessed_words",
        "rate_ref_speech_span_s",
        "rate_ref_active_speech_s",
        "rate_ref_speaking_syl_per_s",
        "rate_ref_articulation_syl_per_s",
        "rate_ref_active_speech_fraction",
        "rate_pause_count",
        "rate_pauses_per_min",
        "rate_total_pause_s",
        "rate_mean_pause_s",
        "rate_pause_fraction",
    ]


def _shared_intervals(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[tuple[float, float]]:
    """
    Return shared speech intervals, mapping deterministic exclusions.
    """

    # Calls the speech mask's function to get speech intervals for
    # stretches of speech
    try:
        return speech_intervals(
            audio=audio,
            sample_rate=sample_rate,
            config=config["speech_mask"],
        )
    except ValueError as error:
        if str(error).startswith(SPEECH_EXCLUSION_MESSAGES):
            raise NoUsableSpeech(str(error)) from error
        raise


def _timing_features(
    intervals: list[tuple[float, float]],
) -> tuple[float, float, np.ndarray]:
    """
    Return speech span, active-speech duration and internal pause durations.
    """

    if not intervals:
        raise NoUsableSpeech("No speech intervals were detected.")

    # start of first speech interval to the end of the last
    # cuts silence before and after speech but keeps pauses
    speech_span_s = intervals[-1][1] - intervals[0][0]
    active_speech_s = sum(
        end_s - start_s
        for start_s, end_s in intervals
    )

    if speech_span_s <= 0:
        raise NoUsableSpeech("Detected speech span is zero.")

    if active_speech_s <= 0:
        raise NoUsableSpeech("Detected active-speech duration is zero.")

    # The gaps between intervals
    # The shortest pause is set by the speech mask's max_gap_s
    pauses = np.asarray(
        [
            intervals[i + 1][0] - intervals[i][1]
            for i in range(len(intervals) - 1)
        ],
        dtype=float,
    )

    # Numerical guard only. Shared intervals should already be ordered and
    # non-overlapping (should never happen)
    if np.any(pauses < 0):
        raise ValueError("Speech intervals overlap or are out of order.")

    return float(speech_span_s), float(active_speech_s), pauses


def extract(
    audio,
    sample_rate,
    row,
    config,
) -> dict[str, float]:
    settings = config["feature"]

    # Gets speech intervals
    intervals = _shared_intervals(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
    )

    reference_column = settings["reference_column"]

    if reference_column not in row:
        raise KeyError(
            f"Reference transcript column {reference_column!r} "
            "is missing from the manifest row."
        )

    raw_reference = str(row[reference_column]).strip()

    if not raw_reference:
        raise ValueError("Reference transcript is empty.")


    # Extracts the number of syllables from the reference
    counts = count_text_syllables(raw_reference)
    syllables = counts["syllables"]

    # Rejects transcripts where words are less than the minimum in config
    if counts["words"] < settings["min_reference_words"]:
        raise ValueError(f"Only {counts["words"]} reference words found (minimum 3).")


    speech_span_s, active_speech_s, pauses = _timing_features(intervals)

    speaking_rate = syllables / speech_span_s           # syllables per second
    articulation_rate = syllables / active_speech_s

    active_fraction = active_speech_s / speech_span_s
    pause_fraction = 1.0 - active_fraction

    pause_count = len(pauses)
    total_pause_s = float(pauses.sum()) if pause_count else 0.0
    mean_pause_s = float(pauses.mean()) if pause_count else 0.0
    pauses_per_min = 60.0 * pause_count / speech_span_s

    # Sanity checks that should always hold if the shared intervals are valid.
    if articulation_rate + 1e-12 < speaking_rate:
        raise ValueError(
            "Articulation rate is lower than speaking rate; "
            "speech timing is inconsistent."
        )

    if not 0.0 <= active_fraction <= 1.0 + 1e-9:
        raise ValueError(
            f"Active-speech fraction is outside [0, 1]: {active_fraction}."
        )

    return {
        "rate_ref_syllables": float(syllables),

        # can flag rows where syllable count had guesses
        "rate_ref_guessed_words": float(
            counts["spelled_words"] + counts["estimated_words"]
        ),

        "rate_ref_speech_span_s": float(speech_span_s),
        "rate_ref_active_speech_s": float(active_speech_s),
        "rate_ref_speaking_syl_per_s": float(speaking_rate),
        "rate_ref_articulation_syl_per_s": float(articulation_rate),
        "rate_ref_active_speech_fraction": float(active_fraction),
        "rate_pause_count": float(pause_count),
        "rate_pauses_per_min": float(pauses_per_min),
        "rate_total_pause_s": float(total_pause_s),
        "rate_mean_pause_s": float(mean_pause_s),
        "rate_pause_fraction": float(pause_fraction),
    }
