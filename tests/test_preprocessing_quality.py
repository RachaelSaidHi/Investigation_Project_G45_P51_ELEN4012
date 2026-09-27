"""Tests for the preprocessing quality measurements."""

import numpy as np
import pytest

from src.preprocessing.audio import TARGET_SAMPLE_RATE
from src.preprocessing.quality import (
    WADA_DB_VALUES,
    WADA_G_VALUES,
    estimate_wada_snr,
    measure_audio_quality,
    measure_duration,
    measure_high_frequency_energy,
)


def make_tone(
    frequency_hz,
    duration_s=1.0,
    amplitude=0.5,
):
    """Generate a sine tone at the target sample rate."""

    sample_count = int(
        TARGET_SAMPLE_RATE * duration_s
    )

    time = (
        np.arange(sample_count)
        / TARGET_SAMPLE_RATE
    )

    return (
        amplitude
        * np.sin(
            2 * np.pi * frequency_hz * time
        )
    ).astype(np.float32)


def make_noisy_speech(
    snr_db,
    sample_count=160000,
    seed=0,
):
    """Generate synthetic speech mixed with Gaussian noise at a known SNR.

    The speech amplitudes follow the Gamma(0.4) model used by WADA-SNR,
    with random signs to produce a symmetric waveform.
    """

    rng = np.random.default_rng(
        seed
    )

    speech = (
        rng.gamma(
            shape=0.4,
            size=sample_count,
        )
        * rng.choice(
            [-1.0, 1.0],
            size=sample_count,
        )
    )

    noise = rng.normal(
        size=sample_count
    )

    speech_power = np.mean(
        speech ** 2
    )

    noise_power = np.mean(
        noise ** 2
    )

    noise_scale = np.sqrt(
        speech_power
        / (
            noise_power
            * 10 ** (snr_db / 10)
        )
    )

    return (
        speech
        + noise_scale * noise
    ).astype(np.float32)


def urgent_reference_wada_snr(
    audio,
    min_amplitude=1e-10,
):
    """Reproduce the URGENT WADA-SNR calculation for comparison.

    This keeps URGENT's original low-boundary behaviour so that the tests
    can document the one deliberate correction made in this project's
    implementation.
    """

    audio = (
        audio
        / np.max(np.abs(audio))
    )

    amplitude = np.abs(
        audio
    )

    np.clip(
        amplitude,
        min_amplitude,
        None,
        out=amplitude,
    )

    mean_amplitude = np.mean(
        amplitude
    )

    mean_log_amplitude = np.mean(
        np.log(amplitude)
    )

    statistic = (
        np.log(mean_amplitude)
        - mean_log_amplitude
    )

    lower_indices = np.where(
        WADA_G_VALUES < statistic
    )[0]

    # This deliberately preserves the current URGENT low-boundary
    # behaviour rather than the corrected behaviour in the function 
    # implemented in this project.
    if len(lower_indices) == 0:
        return WADA_G_VALUES[0]

    lower_index = lower_indices.max()

    if lower_index == len(WADA_G_VALUES) - 1:
        return WADA_DB_VALUES[
            lower_index
        ]

    return (
        WADA_DB_VALUES[lower_index]
        + (
            statistic
            - WADA_G_VALUES[lower_index]
        )
        / (
            WADA_G_VALUES[lower_index + 1]
            - WADA_G_VALUES[lower_index]
        )
        * (
            WADA_DB_VALUES[lower_index + 1]
            - WADA_DB_VALUES[lower_index]
        )
    )


class TestMeasureDuration:

    def test_one_second_of_audio(self):
        audio = np.zeros(
            TARGET_SAMPLE_RATE,
            dtype=np.float32,
        )

        assert measure_duration(
            audio
        ) == pytest.approx(
            1.0
        )

    def test_duration_uses_target_sample_rate(self):
        audio = np.zeros(
            8000,
            dtype=np.float32,
        )

        assert measure_duration(
            audio
        ) == pytest.approx(
            0.5
        )


class TestMeasureHighFrequencyEnergy:

    def test_tone_below_cutoff_has_almost_no_high_frequency_energy(self):
        audio = make_tone(
            1000
        )

        hf_energy_pct = measure_high_frequency_energy(
            audio
        )

        assert hf_energy_pct < 0.01

    def test_tone_above_cutoff_has_almost_all_energy_high(self):
        audio = make_tone(
            6000
        )

        hf_energy_pct = measure_high_frequency_energy(
            audio
        )

        assert hf_energy_pct > 99.0

    def test_white_noise_is_split_roughly_evenly(self):
        """White noise should have similar energy below and above 4 kHz."""

        rng = np.random.default_rng(
            0
        )

        noise = rng.normal(
            size=TARGET_SAMPLE_RATE
        ).astype(np.float32)

        hf_energy_pct = measure_high_frequency_energy(
            noise
        )

        # At 16 kHz, the 4-8 kHz region occupies approximately half of
        # the available 0-8 kHz frequency range.
        assert hf_energy_pct == pytest.approx(
            50.0,
            abs=3.0,
        )

    def test_silent_audio_fails(self):
        audio = np.zeros(
            TARGET_SAMPLE_RATE,
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match="silent audio",
        ):
            measure_high_frequency_energy(
                audio
            )


class TestEstimateWadaSnr:

    @pytest.mark.parametrize(
        "true_snr_db",
        [0, 10, 20, 30],
    )
    def test_matches_urgent_reference_within_table_range(
        self,
        true_snr_db,
    ):
        """
        The adapted implementation should match the URGENT reference
        when the lookup-table boundary correction is not involved.
        """

        audio = make_noisy_speech(
            true_snr_db
        )

        estimate, saturated = estimate_wada_snr(
            audio
        )

        reference = urgent_reference_wada_snr(
            audio
        )

        assert not saturated

        assert estimate == pytest.approx(
            reference,
            abs=1e-6,
        )

    def test_estimate_increases_with_true_snr(self):
        """
        Progressively cleaner synthetic recordings should produce
        progressively higher WADA-SNR estimates.
        """

        estimates = [
            estimate_wada_snr(
                make_noisy_speech(snr_db)
            )[0]
            for snr_db in [
                0,
                10,
                20,
                30,
            ]
        ]

        assert estimates == sorted(
            estimates
        )

    def test_estimate_is_unaffected_by_overall_level(self):
        """
        Scaling a recording should not change WADA-SNR because the
        estimator uses the shape of the amplitude distribution.
        """

        audio = make_noisy_speech(
            10
        )

        original_estimate, _ = estimate_wada_snr(
            audio
        )

        quiet_estimate, _ = estimate_wada_snr(
            audio * 0.01
        )

        assert original_estimate == pytest.approx(
            quiet_estimate,
            abs=1e-3,
        )

    def test_lower_boundary_is_corrected_and_flagged(self):
        """
        Gaussian noise falls below the WADA lookup table. URGENT's current
        implementation returns the first G value (~0.41) in this case,
        while the corrected implementation returns the -20 dB boundary.
        """

        rng = np.random.default_rng(
            0
        )

        noise = rng.normal(
            size=160000
        ).astype(np.float32)

        reference = urgent_reference_wada_snr(
            noise
        )

        estimate, saturated = estimate_wada_snr(
            noise
        )

        assert reference == pytest.approx(
            0.41,
            abs=0.01,
        )

        assert estimate == -20.0
        assert saturated

    def test_upper_boundary_is_flagged(self):
        """A very sparse waveform should reach the upper lookup-table limit."""

        audio = np.zeros(
            TARGET_SAMPLE_RATE,
            dtype=np.float32,
        )

        audio[::1000] = 1.0

        estimate, saturated = estimate_wada_snr(
            audio
        )

        assert estimate == 100.0
        assert saturated

    def test_multichannel_audio_fails(self):
        audio = np.ones(
            (2, 100),
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match="mono",
        ):
            estimate_wada_snr(
                audio
            )

    def test_silent_audio_fails(self):
        audio = np.zeros(
            100,
            dtype=np.float32,
        )

        with pytest.raises(
            ValueError,
            match="silent audio",
        ):
            estimate_wada_snr(
                audio
            )


class TestMeasureAudioQuality:

    def test_returns_all_measurements(self):
        audio = make_noisy_speech(
            10
        )

        quality = measure_audio_quality(
            audio
        )

        assert set(quality) == {
            "processed_duration_s",
            "hf_energy_pct",
            "wada_snr_db",
            "wada_snr_saturated",
        }

        assert quality[
            "processed_duration_s"
        ] == pytest.approx(
            10.0
        )

        assert not quality[
            "wada_snr_saturated"
        ]

# run using: 
# python -m pytest tests/test_preprocessing_quality.py -v