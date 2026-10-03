"""Short-time frame energy shared by audio QC and the DSP speech mask.

Kept in src/common so that preprocessing QC and the DSP speech mask measure
frame energy identically without preprocessing depending on src/dsp.
"""

import librosa
import numpy as np


def frame_energy_db(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-frame RMS energy in dB relative to amplitude 1.0.

    Also returns a Boolean array identifying frames that contain only
    digital zero samples.

    frame_length and hop_length are expressed in samples, not seconds.
    Each output position corresponds to one frame on the same centred frame
    grid: energy_db[k] and all_zero[k] therefore describe the same frame.
    
    RMS provides one overall energy value for each frame. Converting it to
    dB makes relative differences in recording level easier to threshold.
    all_zero is kept separately because exact digital silence receives
    special treatment later in the pipeline.
    """

    if frame_length < 1:
        raise ValueError(
            "frame_length must be at least one sample."
        )

    if hop_length < 1:
        raise ValueError(
            "hop_length must be at least one sample."
        )

    # Zero-padding by half a frame matches librosa's centred framing.
    # Frame k is centred on sample k * hop_length.
    #
    # Padding supplies zeros outside the recording, so the first frame can be
    # centred at time 0 and the final frames can still follow the same regular
    # frame grid near the end of the recording.
    padded = np.pad(
        audio.astype(
            np.float64,
            copy=False,
        ),
        frame_length // 2,
        mode="constant",
    )

    # Each column is one analysis frame; neighbouring columns are
    # hop_length samples apart.
    frames = librosa.util.frame(
        padded,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # This distinguishes exact digital silence from merely low-energy audio.
    # A quiet speech frame can therefore still participate in thresholding,
    # whereas an entirely zero-valued frame is treated explicitly as silence.
    all_zero = np.all(
        frames == 0,
        axis=0,
    )

    # RMS collapses all samples in each frame to one energy measurement.
    rms = np.sqrt(
        np.mean(
            frames ** 2,
            axis=0,
        )
    )

    # All-zero frames produce -inf, which is intentional. They are always
    # non-speech and are excluded from the reference-level calculation.
    with np.errstate(
        divide="ignore",
    ):
        energy_db = (
            20
            * np.log10(rms)
        )

    return energy_db, all_zero
