"""Tests for the reference-transcript speech-rate and pause features.

Most timing tests replace the shared speech mask with fixed intervals so the
expected rates can be calculated by hand. End-to-end tests run the real mask
on synthetic syllable trains.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.dsp.features.speech_rate_reference as speech_rate_reference
from src.common.audio_io import SAMPLE_RATE
from src.dsp.features.speech_rate_reference import (
    NoUsableSpeech,
    _timing_features,
)


CONFIG_DIR = Path("configs/dsp")


# Test settings only, matching the structure of configs/dsp/speech_mask.json.
TEST_SPEECH_MASK_CONFIG = {
    "frame_length_s": 0.025,
    "hop_length_s": 0.010,
    "reference_percentile": 95,
    "threshold_db": 30,
    "max_gap_s": 0.10,
    "min_speech_run_s": 0.05,
    "min_total_speech_s": 1.0,
}

# Test settings only, matching configs/dsp/speech_rate_reference.json.
TEST_FEATURE_CONFIG = {
    "reference_column": "reference_transcript",
    "min_reference_words": 3,
}

# Three speech intervals: span 4.0 s, active speech 3.0 s, two 0.5 s pauses.
FIXED_INTERVALS = [
    (0.5, 1.5),
    (2.0, 3.0),
    (3.5, 4.5),
]


def make_config(**feature_overrides) -> dict:
    """Return the combined config passed to extract(), with overrides."""

    feature = dict(TEST_FEATURE_CONFIG)
    feature.update(feature_overrides)

    return {
        "feature": feature,
        "speech_mask": dict(TEST_SPEECH_MASK_CONFIG),
    }


def make_row(transcript="please call stella") -> dict:
    """Return a minimal manifest row."""

    return {
        "sample_id": "saa_test",
        "reference_transcript": transcript,
    }


def make_syllable_counts(syllables=12, words=6, spelled=0, estimated=0) -> dict:
    """Return a count_text_syllables()-style result."""

    return {
        "syllables": syllables,
        "words": words,
        "dictionary_words": words - spelled - estimated,
        "spelled_words": spelled,
        "estimated_words": estimated,
    }


def make_voiced_tone(
    duration_s: float,
    amplitude: float = 0.1,
    f0_hz: float = 150,
) -> np.ndarray:
    """Create a harmonic tone standing in for voiced speech."""

    time = (
        np.arange(int(SAMPLE_RATE * duration_s))
        / SAMPLE_RATE
    )

    signal = sum(
        np.sin(2 * np.pi * f0_hz * harmonic * time) / harmonic
        for harmonic in range(1, 6)
    )

    return (amplitude * signal).astype(np.float32)


def make_silence(
    duration_s: float,
) -> np.ndarray:
    """Create exact digital silence."""

    return np.zeros(
        int(SAMPLE_RATE * duration_s),
        dtype=np.float32,
    )


def make_syllable_train(
    n_syllables: int,
    syllable_s: float = 0.15,
    gap_s: float = 0.10,
    edge_s: float = 0.5,
) -> np.ndarray:
    """Create separated voiced bursts with silence before and after them."""

    window = np.hanning(int(SAMPLE_RATE * syllable_s)).astype(np.float32)

    pieces = [make_silence(edge_s)]

    for _ in range(n_syllables):
        pieces.append(make_voiced_tone(syllable_s) * window)
        pieces.append(make_silence(gap_s))

    pieces.append(make_silence(edge_s))

    return np.concatenate(pieces)


@pytest.fixture
def fixed_intervals(monkeypatch):
    """Replace the shared speech mask with FIXED_INTERVALS."""

    monkeypatch.setattr(
        speech_rate_reference,
        "speech_intervals",
        lambda audio, sample_rate, config: list(FIXED_INTERVALS),
    )


@pytest.fixture
def fixed_syllables(monkeypatch):
    """Replace the syllable counter with a fixed 12-syllable, 6-word count."""

    monkeypatch.setattr(
        speech_rate_reference,
        "count_text_syllables",
        lambda text: make_syllable_counts(),
    )


class TestModuleInterface:

    def test_module_declares_name_and_version(self):
        assert speech_rate_reference.FEATURE_NAME == "speech_rate_reference"
        assert isinstance(speech_rate_reference.FEATURE_VERSION, str)

    def test_feature_columns_are_unique(self):
        columns = speech_rate_reference.feature_columns(make_config())

        assert len(columns) == len(set(columns))

    def test_no_usable_speech_is_a_value_error(self):
        assert issubclass(NoUsableSpeech, ValueError)

    def test_project_config_has_every_setting_used(self):
        with (CONFIG_DIR / "speech_rate_reference.json").open(encoding="utf-8") as file:
            project_config = json.load(file)

        assert set(TEST_FEATURE_CONFIG) <= set(project_config)


class TestTimingFeatures:

    def test_span_active_and_pauses(self):
        span, active, pauses = _timing_features(FIXED_INTERVALS)

        assert span == pytest.approx(4.0)
        assert active == pytest.approx(3.0)
        np.testing.assert_allclose(pauses, [0.5, 0.5])

    def test_single_interval_has_no_pauses(self):
        span, active, pauses = _timing_features([(1.0, 3.0)])

        assert span == pytest.approx(2.0)
        assert active == pytest.approx(2.0)
        assert len(pauses) == 0

    def test_empty_intervals_are_no_usable_speech(self):
        with pytest.raises(NoUsableSpeech, match="No speech intervals"):
            _timing_features([])

    def test_zero_length_interval_is_no_usable_speech(self):
        with pytest.raises(NoUsableSpeech, match="span is zero"):
            _timing_features([(1.0, 1.0)])

    def test_overlapping_intervals_raise_error(self):
        with pytest.raises(ValueError, match="overlap or are out of order"):
            _timing_features([(0.0, 2.0), (1.5, 3.0)])

    def test_overlap_is_not_a_speech_exclusion(self):
        with pytest.raises(ValueError) as error:
            _timing_features([(0.0, 2.0), (1.5, 3.0)])

        assert not isinstance(error.value, NoUsableSpeech)


class TestExtractRates:

    def test_returns_exactly_the_declared_columns(self, fixed_intervals, fixed_syllables):
        config = make_config()

        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row(),
            config,
        )

        assert list(values) == speech_rate_reference.feature_columns(config)
        assert all(isinstance(value, float) for value in values.values())

    def test_rates_are_calculated_from_span_and_active_time(
        self,
        fixed_intervals,
        fixed_syllables,
    ):
        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert values["rate_ref_syllables"] == 12.0
        assert values["rate_ref_speech_span_s"] == pytest.approx(4.0)
        assert values["rate_ref_active_speech_s"] == pytest.approx(3.0)
        assert values["rate_ref_speaking_syl_per_s"] == pytest.approx(12 / 4.0)
        assert values["rate_ref_articulation_syl_per_s"] == pytest.approx(12 / 3.0)
        assert values["rate_ref_active_speech_fraction"] == pytest.approx(0.75)

    def test_pause_features(self, fixed_intervals, fixed_syllables):
        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert values["rate_pause_count"] == 2.0
        assert values["rate_total_pause_s"] == pytest.approx(1.0)
        assert values["rate_mean_pause_s"] == pytest.approx(0.5)
        assert values["rate_pauses_per_min"] == pytest.approx(60 * 2 / 4.0)
        assert values["rate_pause_fraction"] == pytest.approx(0.25)

    def test_active_and_pause_fractions_sum_to_one(self, fixed_intervals, fixed_syllables):
        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert (
            values["rate_ref_active_speech_fraction"]
            + values["rate_pause_fraction"]
        ) == pytest.approx(1.0)

    def test_single_interval_has_zero_pause_features(self, monkeypatch, fixed_syllables):
        monkeypatch.setattr(
            speech_rate_reference,
            "speech_intervals",
            lambda audio, sample_rate, config: [(0.5, 3.5)],
        )

        values = speech_rate_reference.extract(
            make_silence(4.0),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert values["rate_pause_count"] == 0.0
        assert values["rate_total_pause_s"] == 0.0
        assert values["rate_mean_pause_s"] == 0.0
        assert values["rate_pauses_per_min"] == 0.0
        assert values["rate_pause_fraction"] == pytest.approx(0.0)
        assert values["rate_ref_speaking_syl_per_s"] == pytest.approx(
            values["rate_ref_articulation_syl_per_s"]
        )

    def test_guessed_words_count_spelled_and_estimated_words(
        self,
        monkeypatch,
        fixed_intervals,
    ):
        monkeypatch.setattr(
            speech_rate_reference,
            "count_text_syllables",
            lambda text: make_syllable_counts(words=6, spelled=1, estimated=2),
        )

        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert values["rate_ref_guessed_words"] == 3.0

    def test_real_syllable_count_is_used(self, fixed_intervals):
        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row("Please call Stella."),
            make_config(),
        )

        assert values["rate_ref_syllables"] == 4.0
        assert values["rate_ref_guessed_words"] == 0.0


class TestReferenceTranscript:

    def test_configured_reference_column_is_used(self, fixed_intervals, monkeypatch):
        received = {}

        def fake_count(text):
            received["text"] = text
            return make_syllable_counts()

        monkeypatch.setattr(speech_rate_reference, "count_text_syllables", fake_count)

        row = {
            "reference_transcript": "not used",
            "corrected_transcript": "  the corrected text  ",
        }

        speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            row,
            make_config(reference_column="corrected_transcript"),
        )

        # The transcript is stripped before counting.
        assert received["text"] == "the corrected text"

    def test_pandas_series_row_is_accepted(self, fixed_intervals, fixed_syllables):
        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            pd.Series(make_row()),
            make_config(),
        )

        assert values["rate_ref_syllables"] == 12.0

    def test_missing_reference_column_raises_error(self, fixed_intervals):
        with pytest.raises(
            KeyError,
            match="'reference_transcript' is missing",
        ):
            speech_rate_reference.extract(
                make_silence(5.0),
                SAMPLE_RATE,
                {"sample_id": "saa_test"},
                make_config(),
            )

    @pytest.mark.parametrize(
        "transcript",
        [
            "",
            "   ",
        ],
    )
    def test_empty_reference_raises_error(self, fixed_intervals, transcript):
        with pytest.raises(ValueError, match="Reference transcript is empty"):
            speech_rate_reference.extract(
                make_silence(5.0),
                SAMPLE_RATE,
                make_row(transcript),
                make_config(),
            )

    def test_too_few_reference_words_raise_error(self, fixed_intervals):
        with pytest.raises(ValueError, match="Only 2 reference words found"):
            speech_rate_reference.extract(
                make_silence(5.0),
                SAMPLE_RATE,
                make_row("call stella"),
                make_config(),
            )

    def test_minimum_word_count_comes_from_config(self, fixed_intervals):
        values = speech_rate_reference.extract(
            make_silence(5.0),
            SAMPLE_RATE,
            make_row("call stella"),
            make_config(min_reference_words=2),
        )

        assert values["rate_ref_syllables"] == 3.0


class TestSpeechExclusions:

    def test_digital_silence_is_no_usable_speech(self):
        with pytest.raises(NoUsableSpeech, match="digital silence"):
            speech_rate_reference.extract(
                make_silence(3.0),
                SAMPLE_RATE,
                make_row(),
                make_config(),
            )

    def test_too_little_speech_is_no_usable_speech(self):
        audio = np.concatenate(
            [
                make_silence(1.0),
                make_voiced_tone(0.3),
                make_silence(1.0),
            ]
        )

        with pytest.raises(NoUsableSpeech, match="Insufficient speech"):
            speech_rate_reference.extract(
                audio,
                SAMPLE_RATE,
                make_row(),
                make_config(),
            )

    def test_other_speech_mask_errors_are_not_exclusions(self, monkeypatch):
        def broken_intervals(audio, sample_rate, config):
            raise ValueError("threshold_db must be greater than zero.")

        monkeypatch.setattr(
            speech_rate_reference,
            "speech_intervals",
            broken_intervals,
        )

        with pytest.raises(ValueError, match="threshold_db") as error:
            speech_rate_reference.extract(
                make_silence(3.0),
                SAMPLE_RATE,
                make_row(),
                make_config(),
            )

        assert not isinstance(error.value, NoUsableSpeech)

    def test_speech_is_checked_before_transcript(self):
        # A silent clip with no transcript reports the speech problem, so
        # deterministic exclusions are identified first.
        with pytest.raises(NoUsableSpeech):
            speech_rate_reference.extract(
                make_silence(3.0),
                SAMPLE_RATE,
                {},
                make_config(),
            )


class TestEndToEnd:

    @pytest.mark.parametrize(
        "n_syllables",
        [
            8,
            12,
        ],
    )
    def test_pauses_are_found_between_separated_bursts(self, n_syllables):
        values = speech_rate_reference.extract(
            make_syllable_train(n_syllables),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        # The 0.10 s gaps are longer than the mask's bridged gaps, so each
        # burst becomes its own interval.
        assert values["rate_pause_count"] == n_syllables - 1
        assert 0.0 < values["rate_pause_fraction"] < 1.0

    def test_edge_silence_is_excluded_from_speech_span(self):
        short_edges = speech_rate_reference.extract(
            make_syllable_train(8, edge_s=0.3),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        long_edges = speech_rate_reference.extract(
            make_syllable_train(8, edge_s=2.0),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert long_edges["rate_ref_speech_span_s"] == pytest.approx(
            short_edges["rate_ref_speech_span_s"],
            abs=0.02,
        )
        assert long_edges["rate_ref_speaking_syl_per_s"] == pytest.approx(
            short_edges["rate_ref_speaking_syl_per_s"],
            rel=0.02,
        )

    def test_articulation_rate_is_at_least_speaking_rate(self):
        values = speech_rate_reference.extract(
            make_syllable_train(10),
            SAMPLE_RATE,
            make_row(),
            make_config(),
        )

        assert (
            values["rate_ref_articulation_syl_per_s"]
            >= values["rate_ref_speaking_syl_per_s"]
        )


# run using:
# python -m pytest tests/test_speech_rate_reference.py -v
