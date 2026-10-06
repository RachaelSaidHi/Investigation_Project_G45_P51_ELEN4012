"""Tests for recording-level pitch (F0) extraction."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.audio_io import SAMPLE_RATE
from src.dsp.extraction_pipeline import (
    get_result_columns,
    process_sample,
    validate_feature_module,
)
from src.dsp.features import pitch
from src.dsp.features.pitch import (
    FEATURE_COLUMNS,
    InsufficientVoicingError,
    PitchTrack,
    extract,
    extract_pitch_track,
    feature_columns,
    summarise_pitch,
)


PITCH_CONFIG_PATH = Path("configs/dsp/pitch.json")

N_HARMONICS = 8


def make_config() -> dict:
    """Return the config structure the pipeline passes to a feature module.

    Defined here rather than read from configs/dsp/ so these tests do not
    change if a setting is tuned during validation.
    """

    return {
        "feature": {
            "time_step_s": 0.01,
            "pitch_floor_hz": 75.0,
            "pitch_ceiling_hz": 600.0,
            "max_number_of_candidates": 15,
            "very_accurate": False,
            "silence_threshold": 0.03,
            "voicing_threshold": 0.45,
            "octave_cost": 0.01,
            "octave_jump_cost": 0.35,
            "voiced_unvoiced_cost": 0.14,
            "min_voiced_s": 1.0,
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


def make_voiced(
    f0_hz,
    duration_s: float = 3.0,
    amplitude: float = 0.2,
) -> np.ndarray:
    """Create a voice-like harmonic signal with a known F0.

    f0_hz is either one value or one value per sample, so the same helper
    makes steady tones and glides.
    """

    n_samples = int(SAMPLE_RATE * duration_s)

    f0_per_sample = np.broadcast_to(
        np.asarray(
            f0_hz,
            dtype=np.float64,
        ),
        (n_samples,),
    )

    phase = (
        2
        * np.pi
        * np.cumsum(f0_per_sample)
        / SAMPLE_RATE
    )

    signal = sum(
        np.sin(harmonic * phase) / harmonic
        for harmonic in range(1, N_HARMONICS + 1)
    )

    return (
        amplitude
        * signal
        / np.max(np.abs(signal))
    ).astype(np.float32)


def add_silence(
    signal: np.ndarray,
    before_s: float = 1.0,
    after_s: float = 1.0,
) -> np.ndarray:
    """Put digital silence before and after a signal."""

    return np.concatenate(
        [
            np.zeros(
                int(SAMPLE_RATE * before_s),
                dtype=np.float32,
            ),
            signal,
            np.zeros(
                int(SAMPLE_RATE * after_s),
                dtype=np.float32,
            ),
        ]
    )


def make_track(
    f0_hz,
    in_speech=None,
    time_step_s: float = 0.01,
) -> PitchTrack:
    """Build a pitch track directly, without running Praat.

    Lets the summary statistics be tested on values that are known exactly.
    """

    f0_hz = np.asarray(
        f0_hz,
        dtype=np.float64,
    )

    if in_speech is None:
        in_speech = np.ones(
            len(f0_hz),
            dtype=bool,
        )

    in_speech = np.asarray(
        in_speech,
        dtype=bool,
    )

    voiced = f0_hz > 0

    return PitchTrack(
        times_s=np.arange(len(f0_hz)) * time_step_s,
        f0_hz=f0_hz,
        in_speech=in_speech,
        voiced=voiced,
        used=in_speech & voiced,
        time_step_s=time_step_s,
    )


class TestModuleInterface:

    def test_module_has_what_the_pipeline_requires(self):
        validate_feature_module(pitch)

        assert pitch.FEATURE_NAME == "pitch"
        assert isinstance(pitch.FEATURE_VERSION, str)

    def test_feature_columns(self):
        assert feature_columns(make_config()) == [
            "f0_median_hz",
            "f0_span_st",
            "f0_p10_hz",
            "f0_p90_hz",
            "n_voiced_frames",
            "voiced_fraction",
        ]

    def test_columns_do_not_clash_with_pipeline_columns(self):
        """get_result_columns raises if a pitch column is duplicated or
        reuses one of the pipeline's own column names."""

        result_columns = get_result_columns(
            pitch,
            make_config(),
        )

        for column in FEATURE_COLUMNS:
            assert column in result_columns


class TestConfigValidation:

    def test_missing_setting_raises_error(self):
        config = make_config()
        del config["feature"]["pitch_floor_hz"]

        with pytest.raises(
            KeyError,
            match="pitch_floor_hz",
        ):
            feature_columns(config)

    @pytest.mark.parametrize(
        "section",
        [
            "feature",
            "speech_mask",
        ],
    )
    def test_missing_section_raises_error(self, section):
        config = make_config()
        del config[section]

        with pytest.raises(
            KeyError,
            match=section,
        ):
            feature_columns(config)

    @pytest.mark.parametrize(
        "key, bad_value",
        [
            ("time_step_s", 0),
            ("pitch_floor_hz", 0),
            # The ceiling must be above the floor, not equal to it.
            ("pitch_ceiling_hz", 75.0),
            # Three separate rules: Praat needs at least 2 candidates, the
            # value must be an integer, and True must not pass as the
            # integer 1.
            ("max_number_of_candidates", 1),
            ("max_number_of_candidates", 15.0),
            ("max_number_of_candidates", True),
            ("very_accurate", "false"),
            ("silence_threshold", 1.1),
            ("voicing_threshold", -0.1),
            ("octave_cost", -0.01),
            ("octave_jump_cost", -0.01),
            ("voiced_unvoiced_cost", -0.01),
            ("min_voiced_s", -0.01),
        ],
    )
    def test_invalid_value_raises_error(
        self,
        key,
        bad_value,
    ):
        config = make_config()
        config["feature"][key] = bad_value

        with pytest.raises(
            ValueError,
            match=key,
        ):
            feature_columns(config)

    def test_two_candidates_is_accepted(self):
        config = make_config()
        config["feature"]["max_number_of_candidates"] = 2

        assert feature_columns(config) == FEATURE_COLUMNS

    def test_invalid_config_is_also_rejected_during_extraction(self):
        """The pipeline validates through feature_columns() before a run,
        but the inspection script calls the extractor directly."""

        config = make_config()
        config["feature"]["pitch_ceiling_hz"] = 50.0

        with pytest.raises(
            ValueError,
            match="pitch_ceiling_hz",
        ):
            extract_pitch_track(
                make_voiced(150.0),
                SAMPLE_RATE,
                config["feature"],
                config["speech_mask"],
            )

    @pytest.mark.skipif(
        not PITCH_CONFIG_PATH.exists(),
        reason="Project pitch config not available locally.",
    )
    def test_project_pitch_config_is_valid(self):
        config = make_config()

        config["feature"] = json.loads(
            PITCH_CONFIG_PATH.read_text(encoding="utf-8")
        )

        assert feature_columns(config) == FEATURE_COLUMNS


class TestSummaryStatistics:

    def test_known_percentiles_and_span(self):
        """With 100, 101, ..., 200 Hz the percentiles fall exactly on
        values."""

        summary = summarise_pitch(
            make_track(np.arange(100, 201)),
            make_config()["feature"],
        )

        assert summary["f0_p10_hz"] == pytest.approx(110.0)
        assert summary["f0_median_hz"] == pytest.approx(150.0)
        assert summary["f0_p90_hz"] == pytest.approx(190.0)
        assert summary["f0_span_st"] == pytest.approx(
            12 * math.log2(190.0 / 110.0)
        )
        assert summary["n_voiced_frames"] == 101
        assert summary["voiced_fraction"] == pytest.approx(1.0)

    def test_one_octave_is_twelve_semitones(self):
        # Half the frames at 100 Hz and half at 200 Hz, so P10 = 100 Hz
        # and P90 = 200 Hz.
        f0_hz = np.concatenate(
            [
                np.full(100, 100.0),
                np.full(100, 200.0),
            ]
        )

        summary = summarise_pitch(
            make_track(f0_hz),
            make_config()["feature"],
        )

        assert summary["f0_span_st"] == pytest.approx(12.0)

    def test_span_is_the_same_for_low_and_high_voices(self):
        """Doubling every F0 value doubles the level but must not change
        the span. This is why the span is measured in semitones."""

        f0_hz = np.linspace(100.0, 160.0, 200)
        pitch_config = make_config()["feature"]

        low = summarise_pitch(
            make_track(f0_hz),
            pitch_config,
        )
        high = summarise_pitch(
            make_track(2 * f0_hz),
            pitch_config,
        )

        assert high["f0_median_hz"] == pytest.approx(
            2 * low["f0_median_hz"]
        )
        assert high["f0_span_st"] == pytest.approx(
            low["f0_span_st"]
        )

    def test_only_speech_active_voiced_frames_are_used(self):
        f0_hz = np.concatenate(
            [
                np.full(150, 200.0),  # voiced, in speech: used
                np.zeros(50),         # unvoiced, in speech: not used
                np.full(100, 500.0),  # voiced, outside speech: not used
            ]
        )

        in_speech = np.concatenate(
            [
                np.ones(200, dtype=bool),
                np.zeros(100, dtype=bool),
            ]
        )

        summary = summarise_pitch(
            make_track(
                f0_hz,
                in_speech=in_speech,
            ),
            make_config()["feature"],
        )

        assert summary["f0_median_hz"] == pytest.approx(200.0)
        assert summary["f0_p90_hz"] == pytest.approx(200.0)
        assert summary["n_voiced_frames"] == 150
        assert summary["voiced_fraction"] == pytest.approx(150 / 200)

    def test_a_few_octave_errors_do_not_move_the_summaries(self):
        """5% of frames are an octave too low. Fewer than 10% are affected,
        so the median, P10, P90 and span must not change. This is the
        reason for using percentiles instead of the mean, minimum or
        maximum."""

        f0_hz = np.full(200, 200.0)
        f0_hz[:10] = 100.0

        summary = summarise_pitch(
            make_track(f0_hz),
            make_config()["feature"],
        )

        assert summary["f0_p10_hz"] == pytest.approx(200.0)
        assert summary["f0_median_hz"] == pytest.approx(200.0)
        assert summary["f0_p90_hz"] == pytest.approx(200.0)
        assert summary["f0_span_st"] == pytest.approx(0.0)


class TestMinimumVoicedSpeech:

    def test_exact_minimum_passes_and_one_frame_less_fails(self):
        # 100 frames at 0.01 s is exactly the 1.0 s minimum.
        pitch_config = make_config()["feature"]

        summary = summarise_pitch(
            make_track(np.full(100, 150.0)),
            pitch_config,
        )

        assert summary["n_voiced_frames"] == 100

        with pytest.raises(
            InsufficientVoicingError,
            match="99 voiced frames",
        ):
            summarise_pitch(
                make_track(np.full(99, 150.0)),
                pitch_config,
            )

    @pytest.mark.parametrize(
        "min_voiced_s, required_frames",
        [
            # 0.07 / 0.01 is 7.000000000000001 in floating point. Without
            # rounding, this would wrongly require 8 frames.
            (0.07, 7),
            # A part frame is rounded up.
            (0.105, 11),
        ],
    )
    def test_minimum_is_converted_to_whole_frames(
        self,
        min_voiced_s,
        required_frames,
    ):
        pitch_config = make_config()["feature"]
        pitch_config["min_voiced_s"] = min_voiced_s

        summarise_pitch(
            make_track(np.full(required_frames, 150.0)),
            pitch_config,
        )

        with pytest.raises(InsufficientVoicingError):
            summarise_pitch(
                make_track(np.full(required_frames - 1, 150.0)),
                pitch_config,
            )

    def test_minimum_uses_the_track_time_step(self):
        """The minimum is counted with the frame spacing Praat actually
        produced, not the one in the config. At 0.02 s per frame, 1.0 s is
        50 frames, although the config time step is 0.01 s."""

        pitch_config = make_config()["feature"]

        summarise_pitch(
            make_track(
                np.full(50, 150.0),
                time_step_s=0.02,
            ),
            pitch_config,
        )

        with pytest.raises(InsufficientVoicingError):
            summarise_pitch(
                make_track(
                    np.full(49, 150.0),
                    time_step_s=0.02,
                ),
                pitch_config,
            )

    def test_no_voiced_frames_raises_error_even_with_zero_minimum(self):
        pitch_config = make_config()["feature"]
        pitch_config["min_voiced_s"] = 0.0

        with pytest.raises(InsufficientVoicingError):
            summarise_pitch(
                make_track(np.zeros(200)),
                pitch_config,
            )

    def test_no_speech_frames_raises_error(self):
        with pytest.raises(InsufficientVoicingError):
            summarise_pitch(
                make_track(
                    np.full(200, 150.0),
                    in_speech=np.zeros(200, dtype=bool),
                ),
                make_config()["feature"],
            )

    def test_error_message_reports_the_frame_counts(self):
        """The pipeline leaves the feature columns of an error row empty,
        so the message is the only place these counts are recorded."""

        f0_hz = np.concatenate(
            [
                np.full(40, 150.0),
                np.zeros(160),
            ]
        )

        with pytest.raises(InsufficientVoicingError) as error:
            summarise_pitch(
                make_track(f0_hz),
                make_config()["feature"],
            )

        message = str(error.value)

        assert "40 voiced frames" in message
        assert "200" in message
        assert "100 frames" in message


class TestPitchTrack:

    def test_track_arrays_are_consistent(self):
        config = make_config()

        track = extract_pitch_track(
            add_silence(make_voiced(150.0)),
            SAMPLE_RATE,
            config["feature"],
            config["speech_mask"],
        )

        assert track.times_s.ndim == 1
        assert track.f0_hz.shape == track.times_s.shape
        assert track.in_speech.shape == track.times_s.shape
        assert track.voiced.shape == track.times_s.shape
        assert track.used.shape == track.times_s.shape

        assert np.array_equal(
            track.voiced,
            track.f0_hz > 0,
        )
        assert np.array_equal(
            track.used,
            track.in_speech & track.voiced,
        )

    def test_track_stores_the_praat_time_step(self):
        config = make_config()

        track = extract_pitch_track(
            add_silence(make_voiced(150.0)),
            SAMPLE_RATE,
            config["feature"],
            config["speech_mask"],
        )

        assert track.time_step_s == pytest.approx(0.01)
        assert np.allclose(
            np.diff(track.times_s),
            track.time_step_s,
        )

    def test_track_covers_the_whole_recording(self):
        """Praat must analyse the full waveform, not only the speech
        intervals, so frames must exist across the silent parts too."""

        config = make_config()
        audio = add_silence(make_voiced(150.0))
        duration_s = len(audio) / SAMPLE_RATE

        track = extract_pitch_track(
            audio,
            SAMPLE_RATE,
            config["feature"],
            config["speech_mask"],
        )

        assert track.times_s[0] < 0.05
        assert track.times_s[-1] > duration_s - 0.05

    def test_speech_mask_removes_voiced_frames_outside_speech(self):
        """A loud 150 Hz section is followed by a 300 Hz section 20 dB
        quieter. Praat tracks pitch in both. With a 10 dB speech threshold
        only the loud section counts as speech, so only it may be
        summarised."""

        audio = np.concatenate(
            [
                make_voiced(
                    150.0,
                    duration_s=2.0,
                    amplitude=0.2,
                ),
                make_voiced(
                    300.0,
                    duration_s=2.0,
                    amplitude=0.02,
                ),
            ]
        )

        config = make_config()
        config["speech_mask"]["threshold_db"] = 10

        track = extract_pitch_track(
            audio,
            SAMPLE_RATE,
            config["feature"],
            config["speech_mask"],
        )

        voiced_outside_speech = track.voiced & ~track.in_speech

        assert voiced_outside_speech.sum() > 100
        assert np.median(
            track.f0_hz[voiced_outside_speech]
        ) == pytest.approx(300.0, abs=2.0)

        summary = summarise_pitch(
            track,
            config["feature"],
        )

        assert summary["f0_median_hz"] == pytest.approx(150.0, abs=1.0)

        # The median alone cannot detect a missing mask: slightly more
        # than half the voiced frames are at 150 Hz, so it stays at 150 Hz
        # either way. P90 moves to 300 Hz if the quiet section is included.
        assert summary["f0_p90_hz"] == pytest.approx(150.0, abs=1.0)


class TestSyntheticExtraction:

    @pytest.mark.parametrize(
        "f0_hz",
        [
            100.0,
            220.0,
            500.0,
        ],
    )
    def test_steady_pitch_is_recovered(self, f0_hz):
        summary = extract(
            add_silence(make_voiced(f0_hz)),
            SAMPLE_RATE,
            None,
            make_config(),
        )

        assert summary["f0_median_hz"] == pytest.approx(f0_hz, abs=0.5)
        assert summary["f0_span_st"] == pytest.approx(0.0, abs=0.05)

        # About 3 s of voicing at a 0.01 s step.
        assert summary["n_voiced_frames"] > 250

    def test_one_octave_glide_is_tracked(self):
        """F0 rises steadily (in semitones) from 120 Hz to 240 Hz. The
        middle 80% of a 12-semitone glide spans 9.6 semitones, and the
        median is the geometric midpoint, 120 * sqrt(2) Hz."""

        n_samples = 3 * SAMPLE_RATE

        f0_hz = 120.0 * 2 ** (
            np.arange(n_samples) / n_samples
        )

        summary = extract(
            add_silence(make_voiced(f0_hz)),
            SAMPLE_RATE,
            None,
            make_config(),
        )

        assert summary["f0_median_hz"] == pytest.approx(
            120.0 * math.sqrt(2),
            abs=1.5,
        )
        assert summary["f0_span_st"] == pytest.approx(9.6, abs=0.2)
        assert (
            summary["f0_p10_hz"]
            < summary["f0_median_hz"]
            < summary["f0_p90_hz"]
        )

    def test_output_has_exactly_the_declared_columns(self):
        summary = extract(
            add_silence(make_voiced(150.0)),
            SAMPLE_RATE,
            None,
            make_config(),
        )

        assert list(summary) == FEATURE_COLUMNS

        for value in summary.values():
            assert math.isfinite(float(value))

    def test_extraction_is_deterministic(self):
        audio = add_silence(make_voiced(150.0))
        config = make_config()

        first = extract(audio, SAMPLE_RATE, None, config)
        second = extract(audio, SAMPLE_RATE, None, config)

        assert first == second

    def test_input_audio_is_not_modified(self):
        audio = add_silence(make_voiced(150.0))
        original = audio.copy()

        extract(
            audio,
            SAMPLE_RATE,
            None,
            make_config(),
        )

        assert np.array_equal(
            audio,
            original,
        )

    def test_result_does_not_depend_on_the_sex_label(self):
        """Pitch extraction must not use the source-provided sex label,
        which is unverified for AfriSpeech."""

        audio = add_silence(make_voiced(150.0))
        config = make_config()

        male = extract(
            audio,
            SAMPLE_RATE,
            pd.Series({"sex": "male"}),
            config,
        )
        female = extract(
            audio,
            SAMPLE_RATE,
            pd.Series({"sex": "female"}),
            config,
        )

        assert male == female


class TestRejectedAudio:

    def test_too_little_voiced_speech_raises_error(self):
        # 1.5 s of voicing passes the speech mask's 1.0 s minimum but not
        # a 2.0 s minimum of voiced speech.
        config = make_config()
        config["feature"]["min_voiced_s"] = 2.0

        with pytest.raises(InsufficientVoicingError):
            extract(
                add_silence(
                    make_voiced(
                        150.0,
                        duration_s=1.5,
                    )
                ),
                SAMPLE_RATE,
                None,
                config,
            )

    def test_too_little_speech_is_not_a_voicing_error(self):
        """The shared speech mask fails first, with the same error every
        other feature gives for this recording."""

        with pytest.raises(
            ValueError,
            match="Insufficient speech",
        ) as error:
            extract(
                add_silence(
                    make_voiced(
                        150.0,
                        duration_s=0.5,
                    )
                ),
                SAMPLE_RATE,
                None,
                make_config(),
            )

        assert not isinstance(
            error.value,
            InsufficientVoicingError,
        )

    def test_invalid_sample_rate_is_reported_before_praat_runs(self):
        """The speech mask runs before Praat, so a bad sample rate gives
        the mask's clear message and not a Parselmouth TypeError."""

        with pytest.raises(
            ValueError,
            match="sample_rate",
        ):
            extract(
                add_silence(make_voiced(150.0)),
                0,
                None,
                make_config(),
            )


class TestPipelineIntegration:

    @staticmethod
    def run_sample(
        audio: np.ndarray,
        config: dict,
    ) -> dict:
        """Pass one synthetic recording through the real pipeline step."""

        row = pd.Series(
            {
                "sample_id": "synthetic_1",
                "dataset": "synthetic",
                "processed_filepath": "unused.wav",
                "sex": "female",
                "accent_group": "synthetic",
            }
        )

        return process_sample(
            row=row,
            manifest_index=0,
            feature_module=pitch,
            config=config,
            feature_columns=feature_columns(config),
            processed_root=Path("unused"),
            load_audio=lambda processed_root, processed_filepath: audio,
        )

    def test_voiced_recording_gives_an_ok_row(self):
        record = self.run_sample(
            add_silence(make_voiced(150.0)),
            make_config(),
        )

        assert record["status"] == "ok"
        assert record["feature_name"] == "pitch"
        assert record["f0_median_hz"] == pytest.approx(150.0, abs=0.5)

        for column in FEATURE_COLUMNS:
            assert isinstance(record[column], float)
            assert math.isfinite(record[column])

    def test_voicing_failure_gives_an_identifiable_error_row(self):
        """The pipeline records the exception class name, so too little
        voiced speech can be told apart from a crash in the results file."""

        config = make_config()
        config["feature"]["min_voiced_s"] = 2.0

        record = self.run_sample(
            add_silence(
                make_voiced(
                    150.0,
                    duration_s=1.5,
                )
            ),
            config,
        )

        assert record["status"] == "error"
        assert record["error_type"] == "InsufficientVoicingError"
        assert "voiced" in record["error_message"]

        for column in FEATURE_COLUMNS:
            assert column not in record


# run using:
# python -m pytest tests/test_pitch.py -v