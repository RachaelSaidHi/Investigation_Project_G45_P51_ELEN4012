"""Energy-based speech mask shared by every DSP feature family.

Marks which parts of a processed recording contain speech activity, so that
feature summaries (MFCC, speech controls, pitch and formants) use relevant
frames only. The audio itself is never trimmed or modified; the mask only
selects frames.

Method: short-time frame energy in dB, a file-relative threshold below a
robust reference level, followed by temporal smoothing. Short inactive gaps
can be bridged and short isolated active runs can be removed.

Frames follow librosa's centred convention (center=True with zero padding),
so frame k is centred at k * hop. Each extractor converts the returned
intervals to its own frame grid using speech_mask_for_times().
"""

import librosa
import numpy as np


# Every parameter comes from configs/dsp/speech_mask.json so that the
# settings used for feature extraction can be recorded with each run.
REQUIRED_CONFIG_KEYS = (
    "frame_length_s",
    "hop_length_s",
    "reference_percentile",
    "threshold_db",
    "max_gap_s",
    "min_speech_run_s",
    "min_total_speech_s",
)


def _validate_config(config: dict) -> None:
    """Validate the required speech-mask configuration."""

    missing = [
        key
        for key in REQUIRED_CONFIG_KEYS
        if key not in config
    ]

    if missing:
        raise KeyError(
            f"Speech-mask config is missing: {', '.join(missing)}"
        )

    if config["frame_length_s"] <= 0:
        raise ValueError(
            "frame_length_s must be greater than zero."
        )

    if config["hop_length_s"] <= 0:
        raise ValueError(
            "hop_length_s must be greater than zero."
        )

    if config["hop_length_s"] > config["frame_length_s"]:
        raise ValueError(
            "hop_length_s must not exceed frame_length_s."
        )

    if not 0 < config["reference_percentile"] < 100:
        raise ValueError(
            "reference_percentile must be between 0 and 100."
        )

    if config["threshold_db"] < 0:
        raise ValueError(
            "threshold_db must be non-negative."
        )

    if config["max_gap_s"] < 0:
        raise ValueError(
            "max_gap_s must be non-negative."
        )

    if config["min_speech_run_s"] < 0:
        raise ValueError(
            "min_speech_run_s must be non-negative."
        )

    if config["min_total_speech_s"] < 0:
        raise ValueError(
            "min_total_speech_s must be non-negative."
        )


def _seconds_to_samples(
    duration_s: float,
    sample_rate: int,
    parameter_name: str,
) -> int:
    """Convert a duration to an exact whole number of samples."""

    samples = duration_s * sample_rate
    rounded_samples = round(samples)

    if not np.isclose(
        samples,
        rounded_samples,
    ):
        raise ValueError(
            f"{parameter_name} must correspond to a whole number of "
            f"samples at {sample_rate} Hz."
        )

    return int(rounded_samples)


def _seconds_to_frames(
    duration_s: float,
    hop_length_s: float,
    parameter_name: str,
) -> int:
    """Convert a duration to an exact whole number of frame hops."""

    frames = duration_s / hop_length_s
    rounded_frames = round(frames)

    if not np.isclose(
        frames,
        rounded_frames,
    ):
        raise ValueError(
            f"{parameter_name} must correspond to a whole number of "
            f"frame hops ({hop_length_s:.6f} s each)."
        )

    return int(rounded_frames)


def frame_energy_db(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return per-frame RMS energy in dB relative to amplitude 1.0.

    Also returns a Boolean array identifying frames that contain only
    digital zero samples.
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
    padded = np.pad(
        audio.astype(
            np.float64,
            copy=False,
        ),
        frame_length // 2,
        mode="constant",
    )

    frames = librosa.util.frame(
        padded,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    all_zero = np.all(
        frames == 0,
        axis=0,
    )

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


def _threshold_frames(
    energy_db: np.ndarray,
    all_zero: np.ndarray,
    reference_percentile: float,
    threshold_db: float,
) -> np.ndarray:
    """
    Apply the recording-relative energy threshold.
    
    Mark frames active if their energy is at least threshold_db below the
    reference level. All-zero frames are never active.
    """

    if np.all(all_zero):
        raise ValueError(
            "Audio contains only digital silence."
        )

    # A high percentile is used instead of the maximum so isolated loud
    # transients have less influence on the recording's reference level.
    reference_db = np.percentile(
        energy_db[~all_zero],
        reference_percentile,
    )

    active = (energy_db >= reference_db - threshold_db)

    # Digital-zero frames are always inactive.
    active[all_zero] = False

    return active


def _runs(
    values: np.ndarray,
    target: bool,
) -> list[tuple[int, int]]:
    """Return contiguous [start, end) runs matching the target value."""

    runs = []
    start = None

    for index, value in enumerate(values):

        if value == target and start is None:
            start = index

        elif value != target and start is not None:
            runs.append(
                (start, index)
            )
            start = None

    if start is not None:
        runs.append(
            (start, len(values))
        )

    return runs


def _bridge_short_gaps(
    active: np.ndarray,
    all_zero: np.ndarray,
    max_gap_frames: int,
) -> np.ndarray:
    """
    Fill sufficiently short inactive gaps between active regions.

    Gaps containing all-zero frames are never bridged.
    """

    bridged = active.copy()

    if max_gap_frames <= 0:
        return bridged

    for start, end in _runs(
        active,
        target=False,
    ):

        # Leading and trailing inactivity is not a gap between speech.
        if start == 0 or end == len(active):
            continue

        # Frames containing only digital zeros remain inactive.
        if np.any(
            all_zero[start:end]
        ):
            continue

        if end - start <= max_gap_frames:
            bridged[start:end] = True

    return bridged


def _remove_short_runs(
    active: np.ndarray,
    min_speech_run_frames: int,
) -> np.ndarray:
    """Mark active runs shorter than min_speech_run_frames as inactive."""

    cleaned = active.copy()

    if min_speech_run_frames <= 0:
        return cleaned

    for start, end in _runs(
        active,
        target=True,
    ):
        if end - start < min_speech_run_frames:
            cleaned[start:end] = False

    return cleaned


def _clean_frames(
    active: np.ndarray,
    all_zero: np.ndarray,
    max_gap_frames: int,
    min_speech_run_frames: int,
) -> np.ndarray:
    """Bridge short gaps, then remove short active speech runs."""

    # Bridging first, so that a short syllable separated from its neighbour
    # by a stop closure is joined up before short-run removal could delete it
    bridged = _bridge_short_gaps(
        active=active,
        all_zero=all_zero,
        max_gap_frames=max_gap_frames,
    )

    return _remove_short_runs(
        active=bridged,
        min_speech_run_frames=min_speech_run_frames,
    )


def frames_to_intervals(
    active: np.ndarray,
    hop_length_s: float,
    duration_s: float,
) -> list[tuple[float, float]]:
    """
    Convert active frame decisions to speech time intervals (inclusive []).

    eg.
        [False, True, True, True, False, ...] 
    becomes
        [
            (0.42, 3.10),
            (3.35, 7.80),
        ]

    Frame k is centred at k * hop and represents half a hop on either
    side, so a run of frames [i, j) spans (i - 0.5) * hop to 
    (j - 0.5) * hop. Intervals are clipped to the recording boundaries.
    """

    intervals = []

    for start, end in _runs(
        active,
        target=True,
    ):
        start_s = max(
            0.0,
            (start - 0.5) * hop_length_s,
        )

        end_s = min(
            duration_s,
            (end - 0.5) * hop_length_s,
        )

        if end_s > start_s:
            intervals.append(
                (
                    float(start_s),
                    float(end_s),
                )
            )

    return intervals


def speech_mask_for_times(
    intervals: list[tuple[float, float]],
    times: np.ndarray,
) -> np.ndarray:
    """
    Return True for each time falling inside a speech interval.

    eg. 
    given 
        intervals =[(0.02, 0.05)]
        times = [0.00, 0.01, 0.02, 0.03, ...]
    
    returns
        [False, False, True, True, True, True, False, ...]
    
    Interval ends are inclusive so a centred frame lying exactly on the
    recording end is still covered. Separate speech intervals are always
    divided by at least one inactive frame, so they cannot overlap.
    """

    times = np.asarray(
        times,
        dtype=float,
    )

    if times.ndim != 1:
        raise ValueError(
            "times must be one-dimensional."
        )

    if not np.isfinite(times).all():
        raise ValueError(
            "times contains NaN or infinite values."
        )

    mask = np.zeros(
        times.shape,
        dtype=bool,
    )

    for start_s, end_s in intervals:
        mask |= (
            (times >= start_s)
            & (times <= end_s)
        )

    return mask


def detect_speech_frames(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> np.ndarray:
    """Return the cleaned per-frame speech decision on the centred frame grid."""

    _validate_config(
        config
    )

    audio = np.asarray(
        audio,
    )

    if audio.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional audio, got shape {audio.shape}."
        )

    if audio.size == 0:
        raise ValueError(
            "Audio contains no samples."
        )

    if not np.isfinite(audio).all():
        raise ValueError(
            "Audio contains NaN or infinite samples."
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be greater than zero."
        )

    frame_length = _seconds_to_samples(
        duration_s=config["frame_length_s"],
        sample_rate=sample_rate,
        parameter_name="frame_length_s",
    )

    hop_length = _seconds_to_samples(
        duration_s=config["hop_length_s"],
        sample_rate=sample_rate,
        parameter_name="hop_length_s",
    )

    actual_hop_length_s = (
        hop_length
        / sample_rate
    )

    max_gap_frames = _seconds_to_frames(
        duration_s=config["max_gap_s"],
        hop_length_s=actual_hop_length_s,
        parameter_name="max_gap_s",
    )

    min_speech_run_frames = _seconds_to_frames(
        duration_s=config["min_speech_run_s"],
        hop_length_s=actual_hop_length_s,
        parameter_name="min_speech_run_s",
    )

    if frame_length < 1 or hop_length < 1:
        raise ValueError(
            "Frame and hop lengths must each be at least one sample."
        )

    energy_db, all_zero = frame_energy_db(
        audio=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    active = _threshold_frames(
        energy_db=energy_db,
        all_zero=all_zero,
        reference_percentile=config["reference_percentile"],
        threshold_db=config["threshold_db"],
    )

    return _clean_frames(
        active=active,
        all_zero=all_zero,
        max_gap_frames=max_gap_frames,
        min_speech_run_frames=min_speech_run_frames,
    )


def speech_intervals(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[tuple[float, float]]:
    """
    Return speech-active intervals in seconds for one recording.

    Raises an error if total detected speech is below the configured
    minimum duration.
    """

    active = detect_speech_frames(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
    )

    hop_length = _seconds_to_samples(
        duration_s=config["hop_length_s"],
        sample_rate=sample_rate,
        parameter_name="hop_length_s",
    )

    actual_hop_length_s = (
        hop_length
        / sample_rate
    )

    intervals = frames_to_intervals(
        active=active,
        hop_length_s=actual_hop_length_s,
        duration_s=len(audio) / sample_rate,
    )

    total_speech_s = sum(
        end_s - start_s
        for start_s, end_s in intervals
    )

    if total_speech_s < config["min_total_speech_s"]:
        raise ValueError(
            f"Insufficient speech: {total_speech_s:.2f} s detected, "
            f"minimum is {config['min_total_speech_s']:.2f} s."
        )

    return intervals