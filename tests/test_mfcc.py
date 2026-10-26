"""Tests for the speech-masked MFCC feature module."""

import json
from pathlib import Path

import numpy as np
import pytest

import src.dsp.features.mfcc as mfcc
from src.common.audio_io import SAMPLE_RATE


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

# Test settings only, matching the structure of configs/dsp/mfcc.json.
TEST_MFCC_CONFIG = {
    "n_fft": 400,
    "win_length": 400,
    "hop_length": 160,
    "window": "hamming",
    "center": True,
    "n_mels": 40,
    "fmin": 0,
    "fmax": 8000,
    "n_mfcc": 13,
    "dct_type": 2,
    "norm": "ortho",
    "lifter": 0,
    "power": 2.0,
}


def make_config(**feature_overrides) -> dict:
    """Return the combined config passed to extract(), with overrides."""

    feature = dict(TEST_MFCC_CONFIG)
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


def make_silence(
    duration_s: float,
) -> np.ndarray:
    """Create exact digital silence."""

    return np.zeros(
        int(SAMPLE_RATE * duration_s),
        dtype=np.float32,
    )


def make_speech_with_edges(
    speech_s: float = 2.0,
    edge_s: float = 0.5,
) -> np.ndarray:
    """Create a voiced tone with silence before and after it."""

    return np.concatenate(
        [
            make_silence(edge_s),
            make_voiced_tone(speech_s),
            make_silence(edge_s),
        ]
    )


class TestModuleInterface:

    def test_module_declares_name_and_version(self):
        assert mfcc.FEATURE_NAME == "mfcc"
        assert isinstance(mfcc.FEATURE_VERSION, str)

    def test_no_usable_speech_is_a_value_error(self):
        assert issubclass(mfcc.NoUsableSpeech, ValueError)

    def test_project_config_has_every_setting_used(self):
        with (CONFIG_DIR / "mfcc.json").open(encoding="utf-8") as file:
            project_config = json.load(file)

        assert set(TEST_MFCC_CONFIG) <= set(project_config)


class TestFeatureColumns:

    def test_mean_and_std_for_every_coefficient(self):
        columns = mfcc.feature_columns(make_config())

        assert len(columns) == 2 * TEST_MFCC_CONFIG["n_mfcc"]
        assert columns[:4] == [
            "mfcc_00_mean",
            "mfcc_00_std",
            "mfcc_01_mean",
            "mfcc_01_std",
        ]
        assert columns[-1] == "mfcc_12_std"

    def test_columns_follow_n_mfcc(self):
        columns = mfcc.feature_columns(make_config(n_mfcc=20))

        assert len(columns) == 40
        assert "mfcc_19_mean" in columns

    def test_columns_are_unique(self):
        columns = mfcc.feature_columns(make_config())

        assert len(columns) == len(set(columns))


class TestConfigChecks:

    @pytest.mark.parametrize(
        "center",
        [
            "true",
            1,
            None,
        ],
    )
    def test_non_boolean_center_raises_type_error(self, center):
        with pytest.raises(
            TypeError,
            match="'center' must be true/false",
        ):
            mfcc.extract(
                make_speech_with_edges(),
                SAMPLE_RATE,
                {},
                make_config(center=center),
            )

    @pytest.mark.parametrize(
        "power",
        [
            1.0,
            3.0,
        ],
    )
    def test_power_other_than_two_raises_error(self, power):
        with pytest.raises(
            ValueError,
            match="power must be 2.0",
        ):
            mfcc.extract(
                make_speech_with_edges(),
                SAMPLE_RATE,
                {},
                make_config(power=power),
            )

    def test_integer_power_two_is_accepted(self):
        values = mfcc.extract(
            make_speech_with_edges(),
            SAMPLE_RATE,
            {},
            make_config(power=2),
        )

        assert len(values) == 2 * TEST_MFCC_CONFIG["n_mfcc"]


class TestExtract:

    def test_returns_exactly_the_declared_columns(self):
        config = make_config()

        values = mfcc.extract(
            make_speech_with_edges(),
            SAMPLE_RATE,
            {},
            config,
        )

        assert list(values) == mfcc.feature_columns(config)

    def test_values_are_finite_python_floats(self):
        values = mfcc.extract(
            make_speech_with_edges(),
            SAMPLE_RATE,
            {},
            make_config(),
        )

        assert all(isinstance(value, float) for value in values.values())
        assert all(np.isfinite(value) for value in values.values())

    def test_standard_deviations_are_non_negative(self):
        values = mfcc.extract(
            make_speech_with_edges(),
            SAMPLE_RATE,
            {},
            make_config(),
        )

        assert all(
            value >= 0
            for column, value in values.items()
            if column.endswith("_std")
        )

    def test_extraction_is_deterministic(self):
        audio = make_speech_with_edges()

        first = mfcc.extract(audio, SAMPLE_RATE, {}, make_config())
        second = mfcc.extract(audio, SAMPLE_RATE, {}, make_config())

        assert first == second

    def test_non_centred_frames_are_supported(self):
        config = make_config(center=False)

        values = mfcc.extract(
            make_speech_with_edges(),
            SAMPLE_RATE,
            {},
            config,
        )

        assert list(values) == mfcc.feature_columns(config)

    def test_input_audio_is_not_modified(self):
        audio = make_speech_with_edges()
        original = audio.copy()

        mfcc.extract(audio, SAMPLE_RATE, {}, make_config())

        np.testing.assert_array_equal(audio, original)

    def test_manifest_row_is_not_required(self):
        # The row argument is part of the shared interface but MFCC
        # extraction depends on the audio alone.
        audio = make_speech_with_edges()

        with_row = mfcc.extract(
            audio,
            SAMPLE_RATE,
            {"sample_id": "sample_1"},
            make_config(),
        )
        without_row = mfcc.extract(audio, SAMPLE_RATE, None, make_config())

        assert with_row == without_row


class TestSpeechMasking:

    def test_edge_silence_does_not_change_summary(self):
        speech = make_voiced_tone(2.0)

        short_edges = np.concatenate(
            [make_silence(0.3), speech, make_silence(0.3)]
        )
        long_edges = np.concatenate(
            [make_silence(1.5), speech, make_silence(1.5)]
        )

        short_values = mfcc.extract(short_edges, SAMPLE_RATE, {}, make_config())
        long_values = mfcc.extract(long_edges, SAMPLE_RATE, {}, make_config())

        # Without the mask, three extra seconds of digital silence would
        # dominate the mean of every coefficient.
        for column in short_values:
            assert long_values[column] == pytest.approx(
                short_values[column],
                rel=1e-3,
                abs=1e-3,
            )

    def test_only_frames_inside_speech_intervals_are_summarised(self, monkeypatch):
        audio = make_speech_with_edges()

        monkeypatch.setattr(
            mfcc,
            "speech_intervals",
            lambda audio, sample_rate, config: [(1.0, 1.5)],
        )

        restricted = mfcc.extract(audio, SAMPLE_RATE, {}, make_config())

        monkeypatch.setattr(
            mfcc,
            "speech_intervals",
            lambda audio, sample_rate, config: [(0.0, 3.0)],
        )

        unrestricted = mfcc.extract(audio, SAMPLE_RATE, {}, make_config())

        # Including the silent edges lowers the mean log-energy coefficient.
        assert unrestricted["mfcc_00_mean"] < restricted["mfcc_00_mean"]

    def test_speech_mask_config_is_passed_through(self, monkeypatch):
        received = {}

        def fake_intervals(audio, sample_rate, config):
            received["config"] = config
            received["sample_rate"] = sample_rate
            return [(0.5, 2.5)]

        monkeypatch.setattr(mfcc, "speech_intervals", fake_intervals)

        config = make_config()
        mfcc.extract(make_speech_with_edges(), SAMPLE_RATE, {}, config)

        assert received["config"] == config["speech_mask"]
        assert received["sample_rate"] == SAMPLE_RATE


class TestNoUsableSpeech:

    def test_digital_silence_is_no_usable_speech(self):
        with pytest.raises(
            mfcc.NoUsableSpeech,
            match="digital silence",
        ):
            mfcc.extract(
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

        with pytest.raises(
            mfcc.NoUsableSpeech,
            match="Insufficient speech",
        ):
            mfcc.extract(audio, SAMPLE_RATE, {}, make_config())

    def test_other_speech_mask_errors_are_not_exclusions(self, monkeypatch):
        def broken_intervals(audio, sample_rate, config):
            raise ValueError("hop_length_s must be greater than zero.")

        monkeypatch.setattr(mfcc, "speech_intervals", broken_intervals)

        with pytest.raises(ValueError, match="hop_length_s") as error:
            mfcc.extract(
                make_speech_with_edges(),
                SAMPLE_RATE,
                {},
                make_config(),
            )

        assert not isinstance(error.value, mfcc.NoUsableSpeech)

    def test_missing_speech_mask_key_is_not_an_exclusion(self):
        config = make_config()
        del config["speech_mask"]["threshold_db"]

        with pytest.raises(KeyError, match="threshold_db"):
            mfcc.extract(
                make_speech_with_edges(),
                SAMPLE_RATE,
                {},
                config,
            )

    def test_intervals_outside_frame_grid_are_no_usable_speech(self, monkeypatch):
        monkeypatch.setattr(
            mfcc,
            "speech_intervals",
            lambda audio, sample_rate, config: [(100.0, 101.0)],
        )

        with pytest.raises(
            mfcc.NoUsableSpeech,
            match="No MFCC frames fall inside",
        ):
            mfcc.extract(
                make_speech_with_edges(),
                SAMPLE_RATE,
                {},
                make_config(),
            )


# run using:
# python -m pytest tests/test_mfcc.py -v
