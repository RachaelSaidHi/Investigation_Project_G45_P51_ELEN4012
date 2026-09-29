"""Measure quality-control properties of processed recordings.

These functions describe the processed 16 kHz mono audio. They make no
decisions about whether a recording is acceptable; the exclusion rules
are applied separately in exclusion.py.
"""

import librosa
import numpy as np

from src.preprocessing.audio import TARGET_SAMPLE_RATE


# Audio originally limited to 8 kHz sampling would contain no
# meaningful frequency content above 4 kHz.
HF_CUTOFF_HZ = 4000

# Same STFT settings used during the earlier audio inspection:
# 64 ms window and 16 ms hop at 16 kHz.
N_FFT = 1024
HOP_LENGTH = 256


# ---------------------------------------------------------------------
# WADA-SNR
#
# Adapted from the URGENT 2024 analysis implementation:
#   https://github.com/urgent-challenge/urgent2024_analysis
#   (wada_snr/calculate_wada_snr.py)
#
# License:
#
#   MIT License
#
#   Copyright (c) 2025 urgent-challenge
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
#
#
# Method: C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation
# based on waveform amplitude distribution analysis", Proc. Interspeech,
# 2008, pp. 2598-2601.
#
#
#
# Changes from the URGENT implementation:
# - A statistic below the lookup table returns -20 dB, matching the
#   Ellis reference implementation rather than the URGENT low-boundary
#   behaviour.
# - The function accepts mono audio only and also returns whether the
#   estimate reached one of the lookup-table limits.
# - The final energy-ratio calculation is omitted because it reproduces
#   the interpolated SNR value without changing it.
# ---------------------------------------------------------------------


# SNR values represented by the lookup table, from -20 to 100 dB.
WADA_DB_VALUES = np.arange(
    -20.0,
    101.0,
    dtype=np.float64,
)

# Precomputed WADA statistic for each value in WADA_DB_VALUES.
WADA_G_VALUES = np.array([
    0.409747739, 0.409869263, 0.409985656, 0.409690892, 0.409861864,
    0.409990055, 0.410271377, 0.410526266, 0.411010238, 0.411432644,
    0.412317178, 0.413372716, 0.415264259, 0.417819198, 0.420772515,
    0.424527992, 0.429188858, 0.435103734, 0.442341951, 0.451614855,
    0.462211529, 0.474916474, 0.488838093, 0.505092356, 0.523537093,
    0.543720882, 0.565324274, 0.588475317, 0.613462118, 0.639544959,
    0.667508177, 0.695837243, 0.724547622, 0.754147993, 0.783231484,
    0.812409850, 0.842197752, 0.871664058, 0.900305039, 0.928804177,
    0.956554490, 0.983534905, 1.010471548, 1.036209500, 1.061364248,
    1.085793118, 1.109481904, 1.132779949, 1.154728256, 1.176273084,
    1.197035028, 1.216716938, 1.235358982, 1.253643127, 1.271038908,
    1.287180295, 1.303028647, 1.318395272, 1.332948173, 1.347009353,
    1.360572696, 1.373455135, 1.385771224, 1.397335037, 1.408563968,
    1.419596190, 1.429836240, 1.439584667, 1.449021764, 1.458048307,
    1.466695685, 1.474869384, 1.482699650, 1.490343394, 1.497482140,
    1.504351061, 1.510764265, 1.516989146, 1.522909703, 1.528578001,
    1.533898351, 1.539121095, 1.543906502, 1.548585170, 1.553107762,
    1.557443906, 1.561649273, 1.565663481, 1.569386712, 1.573077668,
    1.576547638, 1.579800830, 1.583041292, 1.586024961, 1.588806813,
    1.591624771, 1.594196895, 1.596931549, 1.599446005, 1.601850111,
    1.604086681, 1.606271340, 1.608261987, 1.610045475, 1.611924722,
    1.613696560, 1.615340743, 1.616889049, 1.618389159, 1.619853744,
    1.621358779, 1.622681189, 1.623904229, 1.625131432, 1.626324628,
    1.627402700, 1.628427675, 1.629455321, 1.630330700, 1.631280263,
    1.632041021,
], dtype=np.float64)

# Prevent log(0) when calculating the WADA statistic.
WADA_MIN_AMPLITUDE = 1e-10


def _validate_audio(audio: np.ndarray) -> None:
    """Check that processed audio is a valid mono waveform."""

    if audio.ndim != 1:
        raise ValueError(
            "Quality measurements expect mono audio."
        )

    if audio.size == 0:
        raise ValueError(
            "Quality measurements expect non-empty audio."
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Audio contains non-finite values."
        )


def measure_duration(audio: np.ndarray) -> float:
    """Return the duration of processed audio in seconds."""

    _validate_audio(audio)

    return len(audio) / TARGET_SAMPLE_RATE


def measure_high_frequency_energy(
    audio: np.ndarray,
) -> float:
    """Return the percentage of spectral energy at or above HF_CUTOFF_HZ (4 kHz)."""

    _validate_audio(audio)

    # Use the same STFT-based measure as the earlier inspection so that
    # the final QC value is directly comparable with the threshold that
    # was selected from those results.
    spectrum = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    power = np.abs(spectrum) ** 2
    total_energy = power.sum()

    if total_energy == 0:
        raise ValueError(
            "Cannot measure high-frequency energy of silent audio."
        )

    frequencies = librosa.fft_frequencies(
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
    )

    high_frequency_energy = power[
        frequencies >= HF_CUTOFF_HZ
    ].sum()

    return float(
        100
        * high_frequency_energy
        / total_energy
    )


def estimate_wada_snr(
    audio: np.ndarray,
) -> tuple[float, bool]:
    """Estimate SNR in dB using WADA-SNR.

    Returns the estimated SNR and whether the estimate reached either
    limit of the lookup table (-20 or 100 dB).
    """

    _validate_audio(audio)

    peak = np.max(
        np.abs(audio)
    )

    if peak == 0:
        raise ValueError(
            "Cannot estimate WADA-SNR for silent audio."
        )

    # Peak-normalise only inside the SNR calculation. The processed
    # recording itself is not modified or loudness-normalised.
    
    # WADA depends on the shape of the amplitude distribution rather than
    # absolute recording level, so the waveform can be peak-normalised
    # before calculating the distribution statistic.
    amplitude = np.abs(
        audio / peak
    )

    amplitude = np.maximum(
        amplitude,
        WADA_MIN_AMPLITUDE,
    )

    # This statistic describes the shape of the waveform's amplitude
    # distribution. WADA maps it to an estimated SNR using the
    # precomputed lookup table above.
    statistic = (
        np.log(np.mean(amplitude))
        - np.mean(np.log(amplitude))
    )

    lower_indices = np.where(
        WADA_G_VALUES < statistic
    )[0]

    # The URGENT implementation returns the first G-table value here.
    # Instead, follow the reference lookup-table interpolation. The 
    # low-boundary case deliberately returns -20 dB, matching Ellis's 
    # MATLAB code.
    if len(lower_indices) == 0:
        return float(
            WADA_DB_VALUES[0]
        ), True

    lower_index = lower_indices.max()

    if lower_index == len(WADA_G_VALUES) - 1:
        return float(
            WADA_DB_VALUES[-1]
        ), True

    # Interpolate between the neighbouring lookup-table values instead
    # of restricting the estimate to whole-number SNR values.
    lower_g = WADA_G_VALUES[
        lower_index
    ]
    upper_g = WADA_G_VALUES[
        lower_index + 1
    ]

    lower_db = WADA_DB_VALUES[
        lower_index
    ]
    upper_db = WADA_DB_VALUES[
        lower_index + 1
    ]

    snr_db = (
        lower_db
        + (
            (statistic - lower_g)
            / (upper_g - lower_g)
        )
        * (upper_db - lower_db)
    )

    return float(snr_db), False


def measure_audio_quality(
    audio: np.ndarray,
) -> dict:
    """Return all QC measurements for one processed recording."""

    wada_snr_db, wada_snr_saturated = estimate_wada_snr(audio)

    return {
        "processed_duration_s": measure_duration(audio),
        "hf_energy_pct": measure_high_frequency_energy(audio),
        "wada_snr_db": wada_snr_db,
        "wada_snr_saturated": wada_snr_saturated,
    }