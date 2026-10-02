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
    """
    Validate the required speech-mask configuration.
    
    This checks the configuration before any signal processing takes place,
    so invalid settings fail explicitly rather than producing a mask with
       unintended behaviour.
    """

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
    """
    Convert a duration to an exact whole number of samples.
    
    Audio exists on a discrete sample grid, so a configured duration such
    as the frame or hop length must correspond to a whole number of samples.
    For example, 0.010 s at 16 kHz corresponds exactly to 160 samples.
    
    The function deliberately rejects durations that fall between samples
    rather than silently rounding the configuration to a different value.
    """

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
    """
    Convert a duration to an exact whole number of frame hops.
    
    Once the audio has been divided into frames, temporal smoothing operates
    on discrete frame decisions rather than directly on floating-point
    seconds. For example, with a 0.010 s hop, 0.100 s corresponds to exactly
    10 frame hops.
    
    Requiring an exact frame count avoids ambiguous boundary behaviour caused
    by silently rounding a configured duration to the nearest frame.
    """

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


def _threshold_frames(
    energy_db: np.ndarray,
    all_zero: np.ndarray,
    reference_percentile: float,
    threshold_db: float,
) -> np.ndarray:
    """
    Apply the recording-relative energy threshold.
    
    Mark frames active if their energy is at or above
    (reference_db - threshold_db). All-zero frames are never active.
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

    This smoothing step handles brief low-energy interruptions inside an
    otherwise active region. A short False run between two True regions can
    therefore become True if it is no longer than max_gap_frames. This 
    accommodates the small silences before stop consonants (eg. /p/ or /t/)
    
    Only internal gaps are considered. Silence before the first active
    region or after the last active region is not interpreted as a brief
    interruption within speech.
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
    """
    Mark active runs shorter than min_speech_run_frames as inactive.
    
    This is the opposite type of temporal cleanup to gap bridging. Instead
    of filling a short inactive region, it removes a very short active
    region that is too brief to retain as speech activity. Likely due to 
    noise / microphone clicks etc.
    
    A run exactly equal to min_speech_run_frames is retained; only shorter
    runs are removed.
    """

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
    """
    Bridge short gaps, then remove short active speech runs.
    
    This function coordinates the two temporal smoothing operations. It does
    not calculate energy or thresholds itself; it starts from the initial
    Boolean frame decisions produced by _threshold_frames().
    """

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

    The returned values are real times in seconds. Boundaries are placed
    halfway between neighbouring frame centres because that is the natural
    point at which responsibility changes from one frame decision to the
    next.
    """

    intervals = []

    for start, end in _runs(
        active,
        target=True,
    ):
        # start is the centre index of the first active frame. Subtracting
        # half a hop moves the boundary halfway toward the preceding frame.
        # At the start of the file this is clipped to 0 seconds.
        start_s = max(
            0.0,
            (start - 0.5) * hop_length_s,
        )

        # end is already the first frame index after the run, so
        # (end - 0.5) * hop lies halfway between the final active frame
        # and the following inactive frame. The recording duration clips
        # any boundary that would otherwise extend beyond the audio.
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
    """
    Return the cleaned per-frame speech decision on the centred frame grid.
    
    The main frame-level orchestration function. It:

    1. validates the configuration and audio,
    2. converts configured timing values onto the discrete sample/frame grid,
    3. calculates frame energies,
    4. applies the recording-relative energy threshold, and
    5. cleans the resulting Boolean decisions temporally.

    The returned Boolean array still belongs to this module's energy frame
    grid. speech_intervals() performs the later conversion to real time.
    """

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

    First obtains the final speech/non-speech decision on the energy frame
    grid, then converts consecutive speech frames into time intervals that
    can be used by the other feature extractors.

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