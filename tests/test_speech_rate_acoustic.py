"""Tests for the transcript-free acoustic speech-rate features.

The syllable-nucleus rules are tested on hand-made intensity contours with a
fake pitch object. End-to-end tests run Praat on synthetic audio: voiced
harmonic bursts stand in for syllables and noise bursts for unvoiced sounds.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import src.dsp.features.speech_rate_acoustic as speech_rate_acoustic
import src.dsp.features.speech_rate_reference as speech_rate_reference
from src.common.audio_io import SAMPLE_RATE
from src.dsp.features.speech_rate_acoustic import (
    NoUsableSpeech,
    TooFewSyllables,
    _candidate_intensity_peaks,
    _time_in_intervals,
    syllable_nuclei,
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

# Test settings only, matching configs/dsp/speech_rate_acoustic.json
# (the de Jong & Wempe v2 script settings).
TEST_FEATURE_CONFIG = {
    "intensity_min_pitch_hz": 50.0,
    "intensity_time_step_s": None,
    "intensity_reference_quantile": 0.99,
    "silence_threshold_db": -25.0,
    "min_dip_db": 2.0,
    "pitch_time_step_s": 0.02,
    "pitch_floor_hz": 30.0,
    "pitch_ceiling_hz": 450.0,
    "pitch_max_candidates": 4,
    "pitch_very_accurate": False,
    "pitch_silence_threshold": 0.03,
    "pitch_voicing_threshold": 0.25,
    "pitch_octave_cost": 0.01,
    "pitch_octave_jump_cost": 0.35,
    "pitch_voiced_unvoiced_cost": 0.25,
    "min_syllables": 3,
}


class FakePitch:
    """Stand-in for parselmouth.Pitch returning fixed values by time."""

    def __init__(self, unvoiced_times=(), default_hz=150.0, unvoiced_value=np.nan):
        self.unvoiced_times = set(unvoiced_times)
        self.default_hz = default_hz
        self.unvoiced_value = unvoiced_value

    def get_value_at_time(self, time_s):
        if time_s in self.unvoiced_times:
            return self.unvoiced_value

        return self.default_hz


def make_config(**feature_overrides) -> dict:
    """Return the combined config passed to extract(), with overrides."""

    feature = dict(TEST_FEATURE_CONFIG)
    feature.update(feature_overrides)

    return {
        "feature": feature,
        "speech_mask": dict(TEST_SPEECH_MASK_CONFIG),
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


def make_noise_burst(
    duration_s: float,
    std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Create a reproducible Gaussian noise burst (unvoiced)."""

    rng = np.random.default_rng(seed)

    return (
        std * rng.standard_normal(int(SAMPLE_RATE * duration_s))
    ).astype(np.float32)


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
    voiced: bool = True,
) -> np.ndarray:
    """Create separated syllable-like bursts with silence before and after."""

    window = np.hanning(int(SAMPLE_RATE * syllable_s)).astype(np.float32)

    pieces = [make_silence(edge_s)]

    for index in range(n_syllables):
        if voiced:
            burst = make_voiced_tone(syllable_s)
        else:
            burst = make_noise_burst(syllable_s, seed=index)

        pieces.append(burst * window)
        pieces.append(make_silence(gap_s))

    pieces.append(make_silence(edge_s))

    return np.concatenate(pieces)


class TestModuleInterface:

    def test_module_declares_name_and_version(self):
        assert speech_rate_acoustic.FEATURE_NAME == "speech_rate_acoustic"
        assert isinstance(speech_rate_acoustic.FEATURE_VERSION, str)

    def test_exclusions_are_value_errors(self):
        assert issubclass(NoUsableSpeech, ValueError)
        assert issubclass(TooFewSyllables, ValueError)

    def test_feature_columns_are_unique(self):
        columns = speech_rate_acoustic.feature_columns(make_config())

        assert len(columns) == len(set(columns))

    def test_columns_do_not_clash_with_reference_rate_columns(self):
        # Both feature families are joined on sample_id at analysis time.
        acoustic = set(speech_rate_acoustic.feature_columns(make_config()))
        reference = set(speech_rate_reference.feature_columns({}))

        assert acoustic.isdisjoint(reference)

    def test_project_config_has_every_setting_used(self):
        with (CONFIG_DIR / "speech_rate_acoustic.json").open(encoding="utf-8") as file:
            project_config = json.load(file)

        assert set(TEST_FEATURE_CONFIG) <= set(project_config)


class TestTimeInIntervals:

    @pytest.mark.parametrize(
        ("time_s", "expected"),
        [
            (0.5, True),        # start boundary is inclusive
            (1.0, True),
            (1.5, True),        # end boundary is inclusive
            (1.75, False),      # between intervals
            (2.5, True),
            (0.1, False),
            (5.0, False),
        ],
    )
    def test_membership(self, time_s, expected):
        intervals = [(0.5, 1.5), (2.0, 3.0)]

        assert _time_in_intervals(time_s, intervals) is expected

    def test_no_intervals(self):
        assert _time_in_intervals(1.0, []) is False


class TestCandidateIntensityPeaks:

    def test_local_maxima_above_threshold_are_returned(self):
        values = np.array([0, 10, 0, 20, 0, 5, 0], dtype=float)

        peaks = _candidate_intensity_peaks(values, threshold_db=7)

        assert peaks.tolist() == [1, 3]

    def test_threshold_is_strict(self):
        values = np.array([0, 10, 0], dtype=float)

        assert _candidate_intensity_peaks(values, threshold_db=10).tolist() == []

    def test_plateau_counts_once_at_its_first_frame(self):
        # Higher than the previous frame and at least as high as the next.
        values = np.array([0, 5, 5, 5, 0], dtype=float)

        assert _candidate_intensity_peaks(values, threshold_db=0).tolist() == [1]

    def test_edge_frames_are_never_peaks(self):
        values = np.array([10, 0, 0, 10], dtype=float)

        assert _candidate_intensity_peaks(values, threshold_db=0).tolist() == []

    @pytest.mark.parametrize(
        "values",
        [
            np.array([], dtype=float),
            np.array([1.0]),
            np.array([1.0, 2.0]),
        ],
    )
    def test_fewer_than_three_frames_returns_no_peaks(self, values):
        peaks = _candidate_intensity_peaks(values, threshold_db=0)

        assert peaks.tolist() == []

    def test_two_dimensional_input_raises_error(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            _candidate_intensity_peaks(
                np.zeros((2, 5)),
                threshold_db=0,
            )


class TestSyllableNuclei:

    # Three clear peaks with deep dips between them, one frame per 0.1 s.
    VALUES = np.array([40, 70, 40, 72, 40, 71, 40], dtype=float)
    TIMES = np.arange(7) * 0.1
    ALL_SPEECH = [(0.0, 1.0)]

    def nuclei(self, **overrides):
        """Run syllable_nuclei() with default test inputs and overrides."""

        arguments = {
            "values": self.VALUES,
            "times": self.TIMES,
            "threshold_db": 50.0,
            "min_dip_db": 2.0,
            "intervals": self.ALL_SPEECH,
            "pitch": FakePitch(),
        }

        arguments.update(overrides)

        return syllable_nuclei(**arguments)

    def test_clear_voiced_peaks_are_accepted(self):
        np.testing.assert_allclose(self.nuclei(), [0.1, 0.3, 0.5])

    def test_peaks_below_threshold_are_rejected(self):
        np.testing.assert_allclose(
            self.nuclei(threshold_db=70.5),
            [0.3, 0.5],
        )

    def test_no_candidate_peaks_returns_empty_array(self):
        result = self.nuclei(threshold_db=100.0)

        assert result.size == 0
        assert result.dtype == float

    def test_small_dip_merges_peaks_of_one_syllable(self):
        # The second peak is only 1 dB above the dip after the first one.
        values = np.array([40, 70, 69, 70, 40], dtype=float)

        result = self.nuclei(
            values=values,
            times=np.arange(5) * 0.1,
        )

        np.testing.assert_allclose(result, [0.1])

    def test_dip_must_exceed_min_dip(self):
        # Dip of exactly min_dip_db is not enough.
        values = np.array([40, 70, 68, 70, 40], dtype=float)

        result = self.nuclei(
            values=values,
            times=np.arange(5) * 0.1,
            min_dip_db=2.0,
        )

        np.testing.assert_allclose(result, [0.1])

    def test_dip_is_measured_from_previous_candidate_peak(self):
        # The third peak's dip is measured from the rejected second peak,
        # not from the last accepted nucleus.
        values = np.array([40, 70, 69, 70, 60, 75, 40], dtype=float)

        result = self.nuclei(
            values=values,
            times=np.arange(7) * 0.1,
        )

        np.testing.assert_allclose(result, [0.1, 0.5])

    def test_first_peak_dip_is_measured_from_start(self):
        # Only 1 dB rise from the start of the contour to the first peak.
        values = np.array([69, 70, 40, 72, 40], dtype=float)

        result = self.nuclei(
            values=values,
            times=np.arange(5) * 0.1,
        )

        np.testing.assert_allclose(result, [0.3])

    def test_peaks_outside_speech_intervals_are_rejected(self):
        result = self.nuclei(intervals=[(0.25, 0.35)])

        np.testing.assert_allclose(result, [0.3])

    @pytest.mark.parametrize(
        "unvoiced_value",
        [
            np.nan,
            None,
            np.inf,
        ],
    )
    def test_unvoiced_peaks_are_rejected(self, unvoiced_value):
        pitch = FakePitch(
            unvoiced_times={float(self.TIMES[3])},
            unvoiced_value=unvoiced_value,
        )

        np.testing.assert_allclose(
            self.nuclei(pitch=pitch),
            [0.1, 0.5],
        )


class TestExtract:

    @pytest.mark.parametrize(
        "n_syllables",
        [
            8,
            12,
            16,
        ],
    )
    def test_voiced_bursts_are_counted_as_syllables(self, n_syllables):
        values = speech_rate_acoustic.extract(
            make_syllable_train(n_syllables),
            SAMPLE_RATE,
            {},
            make_config(),
        )

        assert values["rate_ac_n_syllables"] == float(n_syllables)

    def test_returns_exactly_the_declared_columns(self):
        config = make_config()

        values = speech_rate_acoustic.extract(
            make_syllable_train(8),
            SAMPLE_RATE,
            {},
            config,
        )

        assert list(values) == speech_rate_acoustic.feature_columns(config)
        assert all(isinstance(value, float) for value in values.values())
        assert all(np.isfinite(value) for value in values.values())

    def test_rates_are_consistent_with_timing(self):
        values = speech_rate_acoustic.extract(
            make_syllable_train(8),
            SAMPLE_RATE,
            {},
            make_config(),
        )

        n = values["rate_ac_n_syllables"]
        span = values["rate_ac_speech_span_s"]
        active = values["rate_ac_active_speech_s"]

        assert values["rate_ac_speaking_syl_per_s"] == pytest.approx(n / span)
        assert values["rate_ac_articulation_syl_per_s"] == pytest.approx(n / active)
        assert values["rate_ac_active_speech_fraction"] == pytest.approx(active / span)
        assert 0.0 < values["rate_ac_active_speech_fraction"] <= 1.0
        assert (
            values["rate_ac_articulation_syl_per_s"]
            >= values["rate_ac_speaking_syl_per_s"]
        )

    def test_speech_span_matches_reference_rate_module(self):
        # Both modules take their timing from the same shared speech mask.
        audio = make_syllable_train(8)

        acoustic = speech_rate_acoustic.extract(
            audio,
            SAMPLE_RATE,
            {},
            make_config(),
        )

        reference = speech_rate_reference.extract(
            audio,
            SAMPLE_RATE,
            {"reference_transcript": "please call stella"},
            {
                "feature": {
                    "reference_column": "reference_transcript",
                    "min_reference_words": 3,
                },
                "speech_mask": dict(TEST_SPEECH_MASK_CONFIG),
            },
        )

        assert acoustic["rate_ac_speech_span_s"] == pytest.approx(
            reference["rate_ref_speech_span_s"]
        )
        assert acoustic["rate_ac_active_speech_s"] == pytest.approx(
            reference["rate_ref_active_speech_s"]
        )

    def test_edge_silence_does_not_lower_speaking_rate(self):
        short_edges = speech_rate_acoustic.extract(
            make_syllable_train(8, edge_s=0.3),
            SAMPLE_RATE,
            {},
            make_config(),
        )

        long_edges = speech_rate_acoustic.extract(
            make_syllable_train(8, edge_s=2.0),
            SAMPLE_RATE,
            {},
            make_config(),
        )

        assert long_edges["rate_ac_n_syllables"] == short_edges["rate_ac_n_syllables"]
        assert long_edges["rate_ac_speaking_syl_per_s"] == pytest.approx(
            short_edges["rate_ac_speaking_syl_per_s"],
            rel=0.02,
        )

    def test_extraction_is_deterministic(self):
        audio = make_syllable_train(8)

        first = speech_rate_acoustic.extract(audio, SAMPLE_RATE, {}, make_config())
        second = speech_rate_acoustic.extract(audio, SAMPLE_RATE, {}, make_config())

        assert first == second


class TestExclusions:

    def test_unvoiced_bursts_are_too_few_syllables(self):
        with pytest.raises(TooFewSyllables, match="Only 0 syllable nuclei"):
            speech_rate_acoustic.extract(
                make_syllable_train(10, voiced=False),
                SAMPLE_RATE,
                {},
                make_config(),
            )

    def test_minimum_syllables_comes_from_config(self):
        # Eight bursts are enough speech for the mask but fewer than the
        # configured minimum number of nuclei.
        with pytest.raises(TooFewSyllables, match=r"Only 8 .*minimum 9"):
            speech_rate_acoustic.extract(
                make_syllable_train(8),
                SAMPLE_RATE,
                {},
                make_config(min_syllables=9),
            )

    def test_digital_silence_is_no_usable_speech(self):
        with pytest.raises(NoUsableSpeech, match="digital silence"):
            speech_rate_acoustic.extract(
                make_silence(3.0),
                SAMPLE_RATE,
                {},
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
            speech_rate_acoustic.extract(audio, SAMPLE_RATE, {}, make_config())

    def test_other_speech_mask_errors_are_not_exclusions(self, monkeypatch):
        def broken_intervals(audio, sample_rate, config):
            raise ValueError("max_gap_s must not be negative.")

        monkeypatch.setattr(
            speech_rate_acoustic,
            "speech_intervals",
            broken_intervals,
        )

        with pytest.raises(ValueError, match="max_gap_s") as error:
            speech_rate_acoustic.extract(
                make_syllable_train(8),
                SAMPLE_RATE,
                {},
                make_config(),
            )

        assert not isinstance(error.value, NoUsableSpeech)

    def test_zero_length_speech_span_is_no_usable_speech(self, monkeypatch):
        monkeypatch.setattr(
            speech_rate_acoustic,
            "speech_intervals",
            lambda audio, sample_rate, config: [(1.0, 1.0)],
        )

        with pytest.raises(NoUsableSpeech, match="span is zero"):
            speech_rate_acoustic.extract(
                make_syllable_train(8),
                SAMPLE_RATE,
                {},
                make_config(),
            )


# run using:
# python -m pytest tests/test_speech_rate_acoustic.py -v
