"""Tests for recording-level formant (F1-F3) extraction."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from src.common.audio_io import SAMPLE_RATE
from src.dsp.extraction_pipeline import (
    get_result_columns,
    process_sample,
    validate_feature_module,
)
from src.dsp.features import formants
from src.dsp.features.formants import (
    FEATURE_COLUMNS,
    FormantTrack,
    InsufficientFormantFramesError,
    extract,
    extract_formant_track,
    feature_columns,
    summarise_formants,
)
from src.dsp.features.pitch import PitchTrack


FORMANT_CONFIG_PATH = Path("configs/dsp/formants.json")
PITCH_CONFIG_PATH = Path("configs/dsp/pitch.json")


def make_config() -> dict:
    """Return fixed test settings independent of later parameter tuning."""

    return {
        "feature": {
            "time_step_s": 0.01,
            "max_number_of_formants": 5,
            "formant_ceiling_hz": 5250.0,
            "window_length_s": 0.025,
            "pre_emphasis_from_hz": 50.0,
            "min_usable_formant_s": 1.0,
            "voicing_pitch": {
                "time_step_s": 0.01,
                "pitch_floor_hz": 65.0,
                "pitch_ceiling_hz": 400.0,
                "max_number_of_candidates": 15,
                "very_accurate": False,
                "silence_threshold": 0.03,
                "voicing_threshold": 0.45,
                "octave_cost": 0.01,
                "octave_jump_cost": 0.35,
                "voiced_unvoiced_cost": 0.14,
                "min_voiced_s": 1.0,
            },
        },
        "speech_mask": {
            "frame_length_s": 0.025,
            "hop_length_s": 0.010,
            "reference_percentile": 95,
            "threshold_db": 30,
            "max_gap_s": 0.10,
            "min_speech_run_s": 0.05,
            "min_total_speech_s": 1.0,
        },
    }


def make_vowel(
    formants_hz=(700.0, 1200.0, 2500.0),
    f0_hz: float = 120.0,
    duration_s: float = 2.0,
) -> np.ndarray:
    """Create a periodic voice source filtered by five known resonances.

    A source-filter signal tests Burg's spectral-envelope estimates. A plain
    sine wave would not provide meaningful formant frequencies.
    """

    n_samples = int(SAMPLE_RATE * duration_s)
    pulse_positions = np.arange(0, n_samples, SAMPLE_RATE / f0_hz).astype(int)
    pulses = np.zeros(n_samples)
    pulses[pulse_positions] = 1.0

    # A falling source spectrum better represents voiced speech and matches
    # Praat's use of pre-emphasis during formant estimation.
    signal = lfilter([1.0], [1.0, -0.97], pulses)
    frequencies = (*formants_hz, 3500.0, 4500.0)
    bandwidths = (80.0, 90.0, 120.0, 150.0, 200.0)

    for frequency_hz, bandwidth_hz in zip(frequencies, bandwidths):
        radius = np.exp(-np.pi * bandwidth_hz / SAMPLE_RATE)
        angle = 2 * np.pi * frequency_hz / SAMPLE_RATE
        signal = lfilter([1.0], [1.0, -2 * radius * np.cos(angle), radius**2], signal)

    return (0.2 * signal / np.max(np.abs(signal))).astype(np.float32)


def add_silence(signal: np.ndarray, duration_s: float = 0.5) -> np.ndarray:
    """Place digital silence before and after the synthetic vowel."""

    silence = np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)
    return np.concatenate([silence, signal, silence])


def make_track(formants_hz, in_speech=None, voiced=None, time_step_s=0.01) -> FormantTrack:
    """Build known frame values directly to test selection and summaries."""

    values = np.asarray(formants_hz, dtype=np.float64)
    n_frames = len(values)
    if in_speech is None:
        in_speech = np.ones(n_frames, dtype=bool)
    if voiced is None:
        voiced = np.ones(n_frames, dtype=bool)

    in_speech = np.asarray(in_speech, dtype=bool)
    voiced = np.asarray(voiced, dtype=bool)
    valid = np.isfinite(values).all(axis=1)

    return FormantTrack(
        times_s=np.arange(n_frames) * time_step_s,
        formants_hz=values,
        bandwidths_hz=np.full_like(values, 100.0),
        in_speech=in_speech,
        voiced=voiced,
        valid_formants=valid,
        used=in_speech & voiced & valid,
        time_step_s=time_step_s,
    )


def make_pitch_track(voiced, start_s=0.035) -> PitchTrack:
    """Create known voicing decisions on a 10 ms pitch grid."""

    voiced = np.asarray(voiced, dtype=bool)
    times_s = start_s + np.arange(len(voiced)) * 0.01
    return PitchTrack(
        times_s=times_s,
        f0_hz=np.where(voiced, 150.0, 0.0),
        in_speech=np.ones(len(voiced), dtype=bool),
        voiced=voiced,
        used=voiced.copy(),
        time_step_s=0.01,
    )


class TestModuleInterface:

    def test_feature_interface_and_columns(self):
        validate_feature_module(formants)
        assert formants.FEATURE_NAME == "formants"
        assert isinstance(formants.FEATURE_VERSION, str)

        assert feature_columns(make_config()) == [
            "f1_median_hz", "f1_p10_hz", "f1_p90_hz",
            "f2_median_hz", "f2_p10_hz", "f2_p90_hz",
            "f3_median_hz", "f3_p10_hz", "f3_p90_hz",
            "n_formant_frames", "formant_frame_fraction",
        ]
        assert set(FEATURE_COLUMNS).issubset(get_result_columns(formants, make_config()))


class TestConfigValidation:

    @pytest.mark.parametrize("key", ["formant_ceiling_hz", "voicing_pitch"])
    def test_missing_required_setting(self, key):
        config = make_config()
        del config["feature"][key]
        with pytest.raises(KeyError, match=key):
            feature_columns(config)

    @pytest.mark.parametrize("key, bad_value", [
        ("time_step_s", 0),
        ("formant_ceiling_hz", float("nan")),
        ("pre_emphasis_from_hz", True),
        ("max_number_of_formants", 2.5),
        ("max_number_of_formants", 5.25),
        ("min_usable_formant_s", -1.0),
    ])
    def test_invalid_formant_settings(self, key, bad_value):
        config = make_config()
        config["feature"][key] = bad_value
        with pytest.raises(ValueError, match=key):
            feature_columns(config)

    def test_invalid_voicing_settings_fail_before_extraction(self):
        """The embedded pitch settings must pass pitch.py's value checks."""

        config = make_config()
        config["feature"]["voicing_pitch"]["pitch_ceiling_hz"] = 50.0
        with pytest.raises(ValueError, match="pitch_ceiling_hz"):
            feature_columns(config)

    def test_formant_ceiling_must_be_below_nyquist(self):
        config = make_config()["feature"]
        config["formant_ceiling_hz"] = SAMPLE_RATE / 2
        with pytest.raises(ValueError, match="formant_ceiling_hz"):
            formants._run_praat_formants(make_vowel(), SAMPLE_RATE, config)

    def test_project_configuration_and_frozen_pitch_copy(self):
        """Catch accidental divergence between formants.json and pitch.json."""

        if not FORMANT_CONFIG_PATH.exists() or not PITCH_CONFIG_PATH.exists():
            pytest.skip("Project configuration files are not available here.")

        formant_config = json.loads(FORMANT_CONFIG_PATH.read_text(encoding="utf-8"))
        pitch_config = json.loads(PITCH_CONFIG_PATH.read_text(encoding="utf-8"))

        config = make_config()
        config["feature"] = formant_config
        assert feature_columns(config) == FEATURE_COLUMNS
        assert formant_config["voicing_pitch"] == pitch_config


class TestSummaryStatistics:

    def test_known_percentiles_for_all_three_formants(self):
        steps = np.arange(101)
        values = np.column_stack([300 + steps, 1500 + 2 * steps, 2500 + 3 * steps])
        summary = summarise_formants(make_track(values), make_config()["feature"])

        assert summary["f1_p10_hz"] == pytest.approx(310.0)
        assert summary["f1_median_hz"] == pytest.approx(350.0)
        assert summary["f1_p90_hz"] == pytest.approx(390.0)
        assert summary["f2_p10_hz"] == pytest.approx(1520.0)
        assert summary["f2_median_hz"] == pytest.approx(1600.0)
        assert summary["f2_p90_hz"] == pytest.approx(1680.0)
        assert summary["f3_p10_hz"] == pytest.approx(2530.0)
        assert summary["f3_median_hz"] == pytest.approx(2650.0)
        assert summary["f3_p90_hz"] == pytest.approx(2770.0)
        assert summary["n_formant_frames"] == 101
        assert summary["formant_frame_fraction"] == pytest.approx(1.0)

    def test_only_speech_active_voiced_valid_frames_are_summarised(self):
        values = np.tile([500.0, 1500.0, 2500.0], (300, 1))
        values[150:] = [900.0, 2000.0, 3000.0]
        values[260:, 2] = np.nan

        in_speech = np.ones(300, dtype=bool)
        in_speech[200:260] = False
        voiced = np.ones(300, dtype=bool)
        voiced[150:200] = False

        summary = summarise_formants(
            make_track(values, in_speech, voiced), make_config()["feature"]
        )
        assert summary["n_formant_frames"] == 150
        assert summary["formant_frame_fraction"] == pytest.approx(150 / 240)
        assert summary["f1_p90_hz"] == pytest.approx(500.0)
        assert summary["f2_median_hz"] == pytest.approx(1500.0)
        assert summary["f3_p10_hz"] == pytest.approx(2500.0)

    def test_can_resummarise_the_same_track_with_another_selection(self):
        """Frame-policy comparisons must not require another Burg run."""

        values = np.vstack([
            np.tile([500.0, 1500.0, 2500.0], (150, 1)),
            np.tile([700.0, 1200.0, 2600.0], (150, 1)),
        ])
        track = make_track(values)
        subset = np.zeros(300, dtype=bool)
        subset[150:] = True

        selected = dataclasses.replace(track, used=track.used & subset)
        summary = summarise_formants(selected, make_config()["feature"])
        assert summary["n_formant_frames"] == 150
        assert summary["f1_median_hz"] == pytest.approx(700.0)
        assert summary["f2_median_hz"] == pytest.approx(1200.0)


class TestMinimumUsableFrames:

    @pytest.mark.parametrize("minimum_s, frame_count, passes", [
        (1.0, 100, True),
        (1.0, 99, False),
        (0.07, 7, True),  # Floating-point rounding must not require eight frames.
        (0.105, 10, False),
        (0.105, 11, True),
    ])
    def test_minimum_counts_cumulative_usable_frames(self, minimum_s, frame_count, passes):
        config = make_config()["feature"]
        config["min_usable_formant_s"] = minimum_s
        values = np.tile([500.0, 1500.0, 2500.0], (frame_count, 1))

        if passes:
            assert summarise_formants(make_track(values), config)["n_formant_frames"] == frame_count
        else:
            with pytest.raises(InsufficientFormantFramesError):
                summarise_formants(make_track(values), config)

    def test_nonconsecutive_frames_count_towards_minimum(self):
        voiced = np.zeros(200, dtype=bool)
        voiced[::2] = True
        values = np.tile([500.0, 1500.0, 2500.0], (200, 1))
        summary = summarise_formants(make_track(values, voiced=voiced), make_config()["feature"])
        assert summary["n_formant_frames"] == 100

    def test_no_usable_frames_raises_even_with_zero_minimum(self):
        config = make_config()["feature"]
        config["min_usable_formant_s"] = 0.0
        values = np.tile([500.0, 1500.0, 2500.0], (100, 1))
        with pytest.raises(InsufficientFormantFramesError):
            summarise_formants(make_track(values, voiced=np.zeros(100, dtype=bool)), config)


    def test_minimum_uses_the_track_time_step(self):
        """The minimum is counted with the frame spacing Praat actually
        produced, not the one in the config. At 0.02 s per frame, 1.0 s is
        50 frames, although the config time step is 0.01 s."""

        config = make_config()["feature"]
        values = np.tile([500.0, 1500.0, 2500.0], (50, 1))

        summary = summarise_formants(make_track(values, time_step_s=0.02), config)
        assert summary["n_formant_frames"] == 50

        with pytest.raises(InsufficientFormantFramesError):
            summarise_formants(make_track(values[:49], time_step_s=0.02), config)

    def test_error_message_reports_the_frame_counts(self):
        """The pipeline leaves the feature columns of an error row empty,
        so the message is the only place these counts are recorded."""

        # 200 speech-active frames, 60 of them voiced, and 40 of those
        # with all three formants.
        values = np.tile([500.0, 1500.0, 2500.0], (200, 1))
        values[40:60, 2] = np.nan
        voiced = np.zeros(200, dtype=bool)
        voiced[:60] = True

        with pytest.raises(InsufficientFormantFramesError) as error:
            summarise_formants(make_track(values, voiced=voiced), make_config()["feature"])

        message = str(error.value)
        assert "40 used" in message
        assert "60 speech-active voiced" in message
        assert "200 speech-active" in message
        assert "minimum 100" in message


class TestVoicingAlignment:

    def test_aligned_frames_use_the_matching_pitch_frame(self):
        pitch_track = make_pitch_track([False, True, True, False])
        assert formants._voiced_at_times(pitch_track, pitch_track.times_s).tolist() == [
            False, True, True, False,
        ]

    def test_halfway_frames_require_both_pitch_neighbours(self):
        pitch_track = make_pitch_track([False, True, True, False, True])
        times_s = np.array([0.04, 0.05, 0.06, 0.07])
        assert formants._voiced_at_times(pitch_track, times_s).tolist() == [
            False, True, False, False,
        ]

    def test_midpoint_rounding_does_not_change_selection(self):
        pitch_track = make_pitch_track([True, False])
        times_s = np.array([0.040 - 1e-10, 0.040 + 1e-10])
        assert not formants._voiced_at_times(pitch_track, times_s).any()

    def test_times_outside_pitch_grid_are_unvoiced(self):
        pitch_track = make_pitch_track([True, True, True])
        times_s = np.array([0.0, 0.02, 0.035, 0.055, 0.08])
        assert formants._voiced_at_times(pitch_track, times_s).tolist() == [
            False, False, True, True, False,
        ]


class TestFrameSelection:

    def test_invalid_formants_are_excluded_without_discarding_the_track(self, monkeypatch):
        times_s = 0.035 + 0.01 * np.arange(7)
        values = np.array([
            [500.0, 1500.0, 2500.0],
            [500.0, 1500.0, np.nan],   # F3 missing
            [1500.0, 500.0, 2500.0],   # F1 above F2
            [500.0, 1500.0, 1500.0],   # F2 equal to F3: must be strictly increasing
            [0.0, 1500.0, 2500.0],     # F1 not above 0 Hz
            [500.0, 1500.0, 2500.0],
            [500.0, 1500.0, 2500.0],   # valid, but outside the speech interval
        ])
        widths = np.full_like(values, 100.0)
        widths[-1] = [700.0, 900.0, 1500.0]  # Wide bandwidths are diagnostics, not exclusions.

        monkeypatch.setattr(formants, "speech_intervals", lambda **kwargs: [(0.03, 0.086)])
        monkeypatch.setattr(formants, "extract_pitch_track", lambda **kwargs: make_pitch_track([True] * 7))
        monkeypatch.setattr(formants, "_run_praat_formants", lambda *args, **kwargs: (
            times_s, values, widths, 0.01,
        ))

        config = make_config()
        track = extract_formant_track(
            np.ones(SAMPLE_RATE), SAMPLE_RATE, config["feature"], config["speech_mask"]
        )

        assert track.valid_formants.tolist() == [True, False, False, False, False, True, True]
        assert track.used.tolist() == [True, False, False, False, False, True, False]
        assert np.array_equal(track.formants_hz, values, equal_nan=True)
        assert np.array_equal(track.bandwidths_hz, widths)


class TestSyntheticExtraction:

    @pytest.mark.parametrize("target, f0_hz", [
        ((700.0, 1200.0, 2500.0), 120.0),
        ((300.0, 2300.0, 3000.0), 150.0),
    ])
    def test_burg_recovers_known_vowel_resonances(self, target, f0_hz):
        """Measured formants should be near the known filter resonances.

        Praat estimates a harmonic signal's spectral envelope, so exact
        recovery is not expected. The tolerance permits modest LPC bias,
        and is tighter for F1 because 80 Hz would be a quarter of a
        300 Hz F1.
        """

        config = make_config()
        track = extract_formant_track(
            add_silence(make_vowel(target, f0_hz)),
            SAMPLE_RATE,
            config["feature"],
            config["speech_mask"],
        )
        assert np.count_nonzero(track.used) > 100
        estimated = np.median(track.formants_hz[track.used], axis=0)
        assert estimated[0] == pytest.approx(target[0], abs=25.0)
        assert estimated[1] == pytest.approx(target[1], abs=50.0)
        assert estimated[2] == pytest.approx(target[2], abs=50.0)

    def test_full_waveform_track_and_output_schema(self):
        config = make_config()
        audio = add_silence(make_vowel())
        track = extract_formant_track(
            audio, SAMPLE_RATE, config["feature"], config["speech_mask"]
        )
        n_frames = len(track.times_s)

        assert track.formants_hz.shape == (n_frames, 3)
        assert track.bandwidths_hz.shape == (n_frames, 3)
        assert np.array_equal(track.used, track.in_speech & track.voiced & track.valid_formants)
        assert track.times_s[0] < 0.1
        assert track.times_s[-1] > len(audio) / SAMPLE_RATE - 0.1
        assert (~track.in_speech).any()  # Silent frames retained but excluded from the summary.
        assert np.all(np.diff(track.times_s) > 0)

        summary = summarise_formants(track, config["feature"])
        assert list(summary) == FEATURE_COLUMNS
        assert all(np.isfinite(float(value)) for value in summary.values())

    def test_results_are_deterministic_and_ignore_metadata(self):
        audio = add_silence(make_vowel())
        config = make_config()
        first = extract(audio, SAMPLE_RATE, pd.Series({"sex": "male"}), config)
        second = extract(audio, SAMPLE_RATE, pd.Series({"sex": "female"}), config)
        assert first == second

    def test_unvoiced_noise_has_no_usable_formant_frames(self):
        """Noise can have spectral peaks but should not pass the voicing gate."""

        rng = np.random.default_rng(0)
        noise = (0.05 * rng.standard_normal(2 * SAMPLE_RATE)).astype(np.float32)

        with pytest.raises(InsufficientFormantFramesError):
            extract(add_silence(noise), SAMPLE_RATE, None, make_config())

    def test_insufficient_speech_fails_in_shared_mask(self):
        with pytest.raises(ValueError, match="Insufficient speech"):
            extract(add_silence(make_vowel(duration_s=0.4)), SAMPLE_RATE, None, make_config())


class TestPipelineIntegration:

    @staticmethod
    def run_sample(audio: np.ndarray, config: dict) -> dict:
        row = pd.Series({
            "sample_id": "synthetic_1",
            "dataset": "synthetic",
            "processed_filepath": "unused.wav",
            "sex": "female",
            "accent_group": "synthetic",
        })
        return process_sample(
            row=row,
            manifest_index=0,
            feature_module=formants,
            config=config,
            feature_columns=feature_columns(config),
            processed_root=Path("unused"),
            load_audio=lambda processed_root, processed_filepath: audio,
        )

    def test_valid_recording_produces_success_row(self):
        record = self.run_sample(add_silence(make_vowel()), make_config())
        assert record["status"] == "ok"
        assert record["feature_name"] == "formants"
        assert all(np.isfinite(record[column]) for column in FEATURE_COLUMNS)

    def test_insufficient_formant_material_produces_error_row(self):
        config = make_config()
        config["feature"]["min_usable_formant_s"] = 4.0
        record = self.run_sample(add_silence(make_vowel()), config)
        assert record["status"] == "error"
        assert record["error_type"] == "InsufficientFormantFramesError"
        assert all(column not in record for column in FEATURE_COLUMNS)


# run using:
# python -m pytest tests/test_formants.py -v
