"""Tests for the shared energy-based DSP speech mask."""

import librosa
import numpy as np
import pytest

from src.common.audio_io import SAMPLE_RATE
from src.dsp.speech_mask import (
    _bridge_short_gaps,
    _clean_frames,
    _remove_short_runs,
    _runs,
    _seconds_to_frames,
    _seconds_to_samples,
    _threshold_frames,
    detect_speech_frames,
    frame_energy_db,
    frames_to_intervals,
    speech_intervals,
    speech_mask_for_times,
)


FRAME_LENGTH = 400
HOP_LENGTH = 160
HOP_S = 0.010


# Test settings only. The real values will be chosen using dev data and
# stored in configs/dsp/speech_mask.json.
TEST_CONFIG = {
    "frame_length_s": 0.025,
    "hop_length_s": 0.010,
    "reference_percentile": 95,
    "threshold_db": 30,
    "max_gap_s": 0.10,
    "min_speech_run_s": 0.05,
    "min_total_speech_s": 1.0,
}


def make_config(**overrides) -> dict:
    """Return a copy of the test config with optional values replaced."""

    config = dict(TEST_CONFIG)
    config.update(overrides)

    return config


def make_tone(
    duration_s: float,
    amplitude: float = 0.1,
    frequency_hz: float = 220,
) -> np.ndarray:
    """Create a synthetic tone standing in for speech."""

    time = (
        np.arange(
            int(SAMPLE_RATE * duration_s)
        )
        / SAMPLE_RATE
    )

    return (
        amplitude
        * np.sin(
            2
            * np.pi
            * frequency_hz
            * time
        )
    ).astype(np.float32)


def make_silence(
    duration_s: float,
) -> np.ndarray:
    """Create exact digital silence."""

    return np.zeros(
        int(
            SAMPLE_RATE
            * duration_s
        ),
        dtype=np.float32,
    )


def make_noise(
    duration_s: float,
    std: float,
) -> np.ndarray:
    """Create reproducible low-level Gaussian noise."""

    rng = np.random.default_rng(0)

    return (
        std
        * rng.standard_normal(
            int(
                SAMPLE_RATE
                * duration_s
            )
        )
    ).astype(np.float32)


def bools(
    *values: int,
) -> np.ndarray:
    """Build a Boolean frame array from 0/1 values."""

    return np.array(
        values,
        dtype=bool,
    )


class TestConfigValidation:

    def test_missing_key_raises_error(self):
        config = make_config()
        del config["threshold_db"]

        with pytest.raises(
            KeyError,
            match="threshold_db",
        ):
            detect_speech_frames(
                make_tone(2.0),
                SAMPLE_RATE,
                config,
            )

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            (
                "frame_length_s",
                0,
                "frame_length_s must be greater",
            ),
            (
                "hop_length_s",
                0,
                "hop_length_s must be greater",
            ),
            (
                "hop_length_s",
                0.05,
                "must not exceed frame_length_s",
            ),
            (
                "reference_percentile",
                0,
                "between 0 and 100",
            ),
            (
                "reference_percentile",
                100,
                "between 0 and 100",
            ),
            (
                "threshold_db",
                -1,
                "threshold_db must be non-negative",
            ),
            (
                "max_gap_s",
                -0.01,
                "max_gap_s must be non-negative",
            ),
            (
                "min_speech_run_s",
                -0.01,
                "min_speech_run_s must be non-negative",
            ),
            (
                "min_total_speech_s",
                -1,
                "min_total_speech_s must be non-negative",
            ),
        ],
    )
    def test_invalid_values_raise_error(
        self,
        key,
        value,
        message,
    ):
        config = make_config(
            **{
                key: value,
            }
        )

        with pytest.raises(
            ValueError,
            match=message,
        ):
            detect_speech_frames(
                make_tone(2.0),
                SAMPLE_RATE,
                config,
            )

    def test_hop_not_whole_samples_raises_error(self):
        config = make_config(
            hop_length_s=0.0101,
        )

        with pytest.raises(
            ValueError,
            match="hop_length_s must correspond to a whole number of samples",
        ):
            detect_speech_frames(
                make_tone(2.0),
                SAMPLE_RATE,
                config,
            )

    def test_gap_not_whole_frames_raises_error(self):
        config = make_config(
            max_gap_s=0.105,
        )

        with pytest.raises(
            ValueError,
            match="max_gap_s must correspond to a whole number of frame hops",
        ):
            detect_speech_frames(
                make_tone(2.0),
                SAMPLE_RATE,
                config,
            )


class TestDurationConversion:

    def test_seconds_to_samples_exact(self):
        assert _seconds_to_samples(
            0.025,
            SAMPLE_RATE,
            "frame",
        ) == 400

        assert _seconds_to_samples(
            0.010,
            SAMPLE_RATE,
            "hop",
        ) == 160

    def test_seconds_to_samples_rejects_partial_sample(self):
        with pytest.raises(
            ValueError,
            match="whole number of samples",
        ):
            _seconds_to_samples(
                0.0101,
                SAMPLE_RATE,
                "hop",
            )

    def test_seconds_to_frames_tolerates_float_division(self):
        # Floating-point arithmetic may represent 0.35 / 0.01 as slightly
        # below or above 35 even though mathematically it is exactly 35.
        assert _seconds_to_frames(
            0.35,
            HOP_S,
            "gap",
        ) == 35

    def test_seconds_to_frames_rejects_partial_frames(self):
        with pytest.raises(
            ValueError,
            match="whole number of frame hops",
        ):
            _seconds_to_frames(
                0.105,
                HOP_S,
                "gap",
            )


class TestAudioValidation:

    @pytest.mark.parametrize(
        ("audio", "message"),
        [
            (
                np.zeros(
                    (2, 1000),
                    dtype=np.float32,
                ),
                "one-dimensional audio",
            ),
            (
                np.array(
                    [],
                    dtype=np.float32,
                ),
                "no samples",
            ),
            (
                np.array(
                    [
                        0.1,
                        np.nan,
                        0.1,
                    ],
                    dtype=np.float32,
                ),
                "NaN or infinite",
            ),
        ],
    )
    def test_invalid_audio_raises_error(
        self,
        audio,
        message,
    ):
        with pytest.raises(
            ValueError,
            match=message,
        ):
            detect_speech_frames(
                audio,
                SAMPLE_RATE,
                TEST_CONFIG,
            )

    def test_invalid_sample_rate_raises_error(self):
        with pytest.raises(
            ValueError,
            match="sample_rate must be greater than zero",
        ):
            detect_speech_frames(
                make_tone(2.0),
                0,
                TEST_CONFIG,
            )


class TestFrameEnergy:

    def test_constant_signal_has_expected_energy(self):
        audio = np.full(
            SAMPLE_RATE,
            0.1,
            dtype=np.float32,
        )

        energy_db, _ = frame_energy_db(
            audio,
            FRAME_LENGTH,
            HOP_LENGTH,
        )

        # Edge frames contain zero padding, so use only fully interior frames.
        interior = energy_db[5:-5]

        # Constant amplitude 0.1 has RMS 0.1:
        # 20 * log10(0.1) = -20 dB.
        assert np.allclose(
            interior,
            -20.0,
            atol=1e-6,
        )

    def test_matches_librosa_centred_framing(self):
        """
        Our manually centred frames should match librosa's centred RMS grid.

        This matters because MFCC extraction also uses librosa's centred
        framing, so equivalent frame settings should line up consistently.
        """

        audio = (
            make_tone(1.03)
            + make_noise(
                1.03,
                std=0.01,
            )
        )

        energy_db, _ = frame_energy_db(
            audio,
            FRAME_LENGTH,
            HOP_LENGTH,
        )

        librosa_rms = librosa.feature.rms(
            y=audio,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            center=True,
            pad_mode="constant",
        )[0]

        assert len(energy_db) == len(librosa_rms)

        assert np.allclose(
            10 ** (energy_db / 20),
            librosa_rms,
            rtol=1e-4,
        )

    def test_all_zero_frames_are_flagged(self):
        audio = np.concatenate(
            [
                make_tone(0.5),
                make_silence(0.5),
            ]
        )

        energy_db, all_zero = frame_energy_db(
            audio,
            FRAME_LENGTH,
            HOP_LENGTH,
        )

        assert np.all(
            all_zero[-10:]
        )

        assert np.all(
            np.isneginf(
                energy_db[-10:]
            )
        )

        assert not np.any(
            all_zero[:40]
        )


class TestThresholdFrames:

    def test_threshold_boundary(self):
        """
        A frame exactly at the threshold is active; a quieter frame is not.
        """

        energy_db = np.array(
            [0.0] * 97
            + [
                -30.0,
                -30.001,
                -60.0,
            ]
        )

        all_zero = np.zeros(
            len(energy_db),
            dtype=bool,
        )

        active = _threshold_frames(
            energy_db,
            all_zero,
            reference_percentile=95,
            threshold_db=30,
        )

        assert active[97]
        assert not active[98]
        assert not active[99]

    def test_all_zero_frames_are_never_active(self):
        energy_db = np.array(
            [
                0.0,
                0.0,
                -np.inf,
                0.0,
            ]
        )

        all_zero = bools(
            0,
            0,
            1,
            0,
        )

        active = _threshold_frames(
            energy_db,
            all_zero,
            reference_percentile=95,
            threshold_db=1000,
        )

        assert not active[2]

    def test_only_digital_silence_raises_error(self):
        energy_db = np.full(
            10,
            -np.inf,
        )

        all_zero = np.ones(
            10,
            dtype=bool,
        )

        with pytest.raises(
            ValueError,
            match="only digital silence",
        ):
            _threshold_frames(
                energy_db,
                all_zero,
                95,
                30,
            )

    def test_isolated_click_does_not_shift_threshold(self):
        """
        A small number of unusually loud frames should not change the
        classification of the ordinary frames when a high percentile is used.
        """

        energy_db = np.array(
            [-20.0] * 90
            + [-45.0] * 10
        )

        all_zero = np.zeros(
            len(energy_db),
            dtype=bool,
        )

        without_click = _threshold_frames(
            energy_db,
            all_zero,
            95,
            30,
        )

        with_click_db = energy_db.copy()
        with_click_db[[10, 11]] = 0.0

        with_click = _threshold_frames(
            with_click_db,
            all_zero,
            95,
            30,
        )

        assert np.array_equal(
            without_click,
            with_click,
        )


class TestRuns:

    def test_runs_use_exclusive_end(self):
        values = bools(
            0,
            1,
            1,
            1,
            0,
        )

        # (1, 4) means frames 1, 2 and 3.
        assert _runs(
            values,
            target=True,
        ) == [
            (1, 4),
        ]

        assert _runs(
            values,
            target=False,
        ) == [
            (0, 1),
            (4, 5),
        ]

    def test_run_reaching_end_closes_at_array_length(self):
        assert _runs(
            bools(
                0,
                1,
                1,
            ),
            target=True,
        ) == [
            (1, 3),
        ]

    def test_no_matching_values_gives_no_runs(self):
        assert _runs(
            bools(
                0,
                0,
                0,
            ),
            target=True,
        ) == []


class TestBridgeShortGaps:

    def test_gap_boundary(self):
        short = bools(
            1,
            1,
            0,
            0,
            0,
            1,
            1,
        )

        long = bools(
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
        )

        bridged_short = _bridge_short_gaps(
            short,
            np.zeros(
                len(short),
                dtype=bool,
            ),
            max_gap_frames=3,
        )

        bridged_long = _bridge_short_gaps(
            long,
            np.zeros(
                len(long),
                dtype=bool,
            ),
            max_gap_frames=3,
        )

        # Exactly the configured maximum is bridged.
        assert np.all(
            bridged_short
        )

        # One frame beyond the maximum is left unchanged.
        assert np.array_equal(
            bridged_long,
            long,
        )

    def test_leading_and_trailing_inactivity_is_not_bridged(self):
        active = bools(
            0,
            1,
            1,
            0,
        )

        bridged = _bridge_short_gaps(
            active,
            np.zeros(
                len(active),
                dtype=bool,
            ),
            max_gap_frames=5,
        )

        assert np.array_equal(
            bridged,
            active,
        )

    def test_gap_containing_digital_silence_is_not_bridged(self):
        active = bools(
            1,
            1,
            0,
            0,
            1,
            1,
        )

        all_zero = bools(
            0,
            0,
            0,
            1,
            0,
            0,
        )

        bridged = _bridge_short_gaps(
            active,
            all_zero,
            max_gap_frames=5,
        )

        assert np.array_equal(
            bridged,
            active,
        )

    def test_zero_max_gap_disables_bridging(self):
        active = bools(
            1,
            0,
            1,
        )

        bridged = _bridge_short_gaps(
            active,
            np.zeros(
                len(active),
                dtype=bool,
            ),
            max_gap_frames=0,
        )

        assert np.array_equal(
            bridged,
            active,
        )


class TestRemoveShortRuns:

    def test_run_boundary(self):
        active = bools(
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
        )

        cleaned = _remove_short_runs(
            active,
            min_speech_run_frames=3,
        )

        # Exactly the minimum length survives.
        assert np.all(
            cleaned[1:4]
        )

        # One frame shorter than the minimum is removed.
        assert not np.any(
            cleaned[6:8]
        )

    def test_zero_minimum_disables_removal(self):
        active = bools(
            0,
            1,
            0,
        )

        cleaned = _remove_short_runs(
            active,
            min_speech_run_frames=0,
        )

        assert np.array_equal(
            cleaned,
            active,
        )


class TestCleanFrames:

    def test_bridging_happens_before_short_run_removal(self):
        active = bools(
            0,
            1,
            1,
            0,
            1,
            1,
            0,
        )

        all_zero = np.zeros(
            len(active),
            dtype=bool,
        )

        cleaned = _clean_frames(
            active,
            all_zero,
            max_gap_frames=1,
            min_speech_run_frames=4,
        )

        # If short-run removal happened first, the two two-frame runs would
        # disappear. Bridging first joins them into one five-frame run.
        assert np.array_equal(
            cleaned,
            bools(
                0,
                1,
                1,
                1,
                1,
                1,
                0,
            ),
        )


class TestFramesToIntervals:

    def test_interior_run_spans_half_hop_either_side(self):
        active = bools(
            0,
            0,
            1,
            1,
            1,
            0,
            0,
        )

        intervals = frames_to_intervals(
            active,
            HOP_S,
            duration_s=1.0,
        )

        # Active frame centres are 0.02, 0.03 and 0.04 s.
        # Their region therefore runs halfway to the neighbouring
        # frame centres: 0.015 s to 0.045 s.
        assert len(intervals) == 1

        assert intervals[0] == pytest.approx(
            (
                0.015,
                0.045,
            )
        )

    def test_intervals_are_clipped_to_recording_boundaries(self):
        active = bools(
            1,
            1,
            0,
            0,
            1,
            1,
        )

        intervals = frames_to_intervals(
            active,
            HOP_S,
            duration_s=0.05,
        )

        assert intervals[0][0] == 0.0
        assert intervals[-1][1] == 0.05

    def test_frame_centres_map_back_to_original_decisions(self):
        """
        Converting frame decisions to intervals and then checking the
        original frame centres should reproduce the same Boolean mask.

        The final frame is deliberately active and centred exactly at the
        recording end, so this also protects the inclusive-end behaviour.
        """

        rng = np.random.default_rng(1)

        active = (
            rng.random(500)
            > 0.5
        )

        active[-1] = True

        duration_s = (
            (len(active) - 1)
            * HOP_S
        )

        intervals = frames_to_intervals(
            active,
            HOP_S,
            duration_s,
        )

        frame_times = librosa.frames_to_time(
            np.arange(
                len(active)
            ),
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )

        recovered = speech_mask_for_times(
            intervals,
            frame_times,
        )

        assert np.array_equal(
            recovered,
            active,
        )


class TestSpeechMaskForTimes:

    def test_times_inside_and_outside_intervals(self):
        intervals = [
            (
                1.0,
                2.0,
            ),
            (
                3.0,
                4.0,
            ),
        ]

        times = np.array(
            [
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.5,
                4.5,
            ]
        )

        mask = speech_mask_for_times(
            intervals,
            times,
        )

        assert np.array_equal(
            mask,
            bools(
                0,
                1,
                1,
                1,
                0,
                1,
                0,
            ),
        )

    def test_no_intervals_gives_no_speech(self):
        mask = speech_mask_for_times(
            [],
            np.array(
                [
                    0.1,
                    0.2,
                ]
            ),
        )

        assert not np.any(
            mask
        )

    def test_two_dimensional_times_raise_error(self):
        with pytest.raises(
            ValueError,
            match="one-dimensional",
        ):
            speech_mask_for_times(
                [
                    (
                        0.0,
                        1.0,
                    ),
                ],
                np.zeros(
                    (2, 2)
                ),
            )

    def test_non_finite_times_raise_error(self):
        with pytest.raises(
            ValueError,
            match="NaN or infinite",
        ):
            speech_mask_for_times(
                [
                    (
                        0.0,
                        1.0,
                    ),
                ],
                np.array(
                    [
                        0.1,
                        np.nan,
                    ]
                ),
            )


class TestSpeechIntervals:

    def test_silence_tone_silence_gives_tone_interval(self):
        audio = np.concatenate(
            [
                make_silence(1.0),
                make_tone(2.0),
                make_silence(1.0),
            ]
        )

        intervals = speech_intervals(
            audio,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        assert len(intervals) == 1

        assert intervals[0][0] == pytest.approx(
            1.0,
            abs=0.02,
        )

        assert intervals[0][1] == pytest.approx(
            3.0,
            abs=0.02,
        )

    def test_quiet_background_noise_is_excluded(self):
        """
        Noise roughly 37 dB below the tone should fall below the 30 dB
        relative threshold, leaving the tone region active.
        """

        audio = np.concatenate(
            [
                make_noise(
                    1.0,
                    std=0.001,
                ),
                (
                    make_tone(2.0)
                    + make_noise(
                        2.0,
                        std=0.001,
                    )
                ),
                make_noise(
                    1.0,
                    std=0.001,
                ),
            ]
        )

        intervals = speech_intervals(
            audio,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        assert len(intervals) == 1

        assert intervals[0][0] == pytest.approx(
            1.0,
            abs=0.02,
        )

        assert intervals[0][1] == pytest.approx(
            3.0,
            abs=0.02,
        )

    def test_result_does_not_depend_on_overall_level(self):
        """
        Scaling the whole recording should leave the mask unchanged because
        the threshold is relative to each recording's own energy reference.
        """

        audio = np.concatenate(
            [
                make_noise(
                    1.0,
                    std=0.001,
                ),
                make_tone(2.0),
                make_noise(
                    1.0,
                    std=0.001,
                ),
            ]
        )

        loud = speech_intervals(
            audio,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        quiet = speech_intervals(
            audio * 0.125,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        assert loud == quiet

    def test_audio_is_not_modified(self):
        audio = np.concatenate(
            [
                make_silence(0.5),
                make_tone(2.0),
            ]
        )

        original = audio.copy()

        speech_intervals(
            audio,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        assert np.array_equal(
            audio,
            original,
        )

    def test_insufficient_speech_raises_error(self):
        audio = np.concatenate(
            [
                make_silence(2.0),
                make_tone(0.5),
            ]
        )

        with pytest.raises(
            ValueError,
            match="Insufficient speech",
        ):
            speech_intervals(
                audio,
                SAMPLE_RATE,
                TEST_CONFIG,
            )

    def test_all_zero_audio_raises_error(self):
        with pytest.raises(
            ValueError,
            match="only digital silence",
        ):
            speech_intervals(
                make_silence(2.0),
                SAMPLE_RATE,
                TEST_CONFIG,
            )

    def test_same_input_gives_same_result(self):
        audio = np.concatenate(
            [
                make_noise(
                    1.0,
                    std=0.001,
                ),
                make_tone(2.0),
                make_noise(
                    1.0,
                    std=0.001,
                ),
            ]
        )

        first = speech_intervals(
            audio,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        second = speech_intervals(
            audio,
            SAMPLE_RATE,
            TEST_CONFIG,
        )

        assert first == second


# run using:
# python -m pytest tests/test_speech_mask.py -v