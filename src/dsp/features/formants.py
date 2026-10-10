"""Extract recording-level F1-F3 formant features using Praat's Burg method.

Praat estimates formants across the full recording. The shared speech mask
and the frozen pitch track then identify speech-active, voiced frames.
The resulting summaries describe voiced speech as a whole, not individual
vowel categories.

The full formant track is retained for inspection of rejected frames and 
comparison of frame-selection methods during validation. The formant ceiling,
frame-selection policy and minimum usable duration are still provisional.

This module follows the shared DSP feature interface. The runner supplies
formants.json as config["feature"] and speech_mask.json as
config["speech_mask"]. Formants.json also contains a copy of the frozen
pitch settings used to identify voiced frames, so the runner need not change.
"""

from dataclasses import dataclass

import numpy as np
import parselmouth

from src.dsp.features.pitch import _validate_config as _validate_pitch_config
from src.dsp.features.pitch import PitchTrack, extract_pitch_track
from src.dsp.speech_mask import speech_intervals, speech_mask_for_times


FEATURE_NAME = "formants"
FEATURE_VERSION = "1"

N_FORMANTS = 3
LOWER_PERCENTILE = 10
UPPER_PERCENTILE = 90

# Small tolerance in pitch-frame units to recognise a formant time that lies
# exactly halfway between two pitch frames despite floating-point rounding.
GRID_TOLERANCE = 1e-6

FEATURE_COLUMNS = [
    "f1_median_hz",
    "f1_p10_hz",
    "f1_p90_hz",
    "f2_median_hz",
    "f2_p10_hz",
    "f2_p90_hz",
    "f3_median_hz",
    "f3_p10_hz",
    "f3_p90_hz",
    "n_formant_frames",
    "formant_frame_fraction",
]

REQUIRED_CONFIG_KEYS = (
    "time_step_s",
    "max_number_of_formants",
    "formant_ceiling_hz",
    "window_length_s",
    "pre_emphasis_from_hz",
    "min_usable_formant_s",
    "voicing_pitch",
)


class InsufficientFormantFramesError(ValueError):
    """Raised when too few usable frames remain to summarise a recording."""


@dataclass(frozen=True)
class FormantTrack:
    """Frame-by-frame formant measurements and selection decisions.

    Each row of formants_hz contains [F1, F2, F3] for one Praat frame.
    The matching row of bandwidths_hz contains their bandwidths [B1, B2, B3].
    Both arrays have shape (number of frames, 3). A missing measurement
    stays as NaN so we can recognise it during inspection.

    The Boolean masks have one value per frame:
      in_speech: the frame lies inside a detected speech interval.
      voiced: Praat's pitch tracker considers the frame voiced.
      valid_formants: F1-F3 are present, positive and correctly ordered.
      used: all three conditions are true, so the frame enters the summary.

    All frames, including those not used, remain in this track.
    """

    times_s: np.ndarray
    formants_hz: np.ndarray
    bandwidths_hz: np.ndarray
    in_speech: np.ndarray
    voiced: np.ndarray
    valid_formants: np.ndarray
    used: np.ndarray
    time_step_s: float


def _validate_config(formant_config: dict) -> None:
    """Reject invalid feature settings before running Praat."""

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in formant_config]
    if missing:
        raise KeyError(f"Formant config is missing: {', '.join(missing)}")

    for key in (
        "time_step_s",
        "formant_ceiling_hz",
        "window_length_s",
        "pre_emphasis_from_hz",
    ):
        value = formant_config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be a finite positive number.")

    # Praat permits integer or half-integer formant counts, such as 5 or 5.5.
    # We must model at least three to retrieve F1, F2 and F3.
    count = formant_config["max_number_of_formants"]
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, float))
        or not np.isfinite(count)
        or count < N_FORMANTS
        or not float(2 * count).is_integer()
    ):
        raise ValueError("max_number_of_formants must be a multiple of 0.5 and at least 3.")

    minimum = formant_config["min_usable_formant_s"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not np.isfinite(minimum)
        or minimum < 0
    ):
        raise ValueError("min_usable_formant_s must be finite and non-negative.")

    # Use the pitch module's validation rules for the voicing settings.
    pitch_config = formant_config["voicing_pitch"]
    if not isinstance(pitch_config, dict):
        raise TypeError("voicing_pitch must contain the frozen pitch settings.")

    _validate_pitch_config(pitch_config)


def _run_praat_formants(
    audio: np.ndarray,
    sample_rate: int,
    formant_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run Praat's Burg formant tracker on the complete waveform.
    
    Returns the frame times, F1-F3, their bandwidths and the actual frame
    spacing. No frame selection happens here: every Praat frame is
    returned, with NaN where Praat found no such formant.
    """

    # Praat resamples to twice the ceiling before the LPC analysis. A
    # ceiling above the Nyquist frequency cannot be honoured: Praat would
    # silently analyse up to Nyquist instead.
    ceiling = formant_config["formant_ceiling_hz"]
    if ceiling >= sample_rate / 2:
        raise ValueError(
            "formant_ceiling_hz must be below half the sample rate "
            f"({sample_rate / 2:.0f} Hz)."
        )
    
    sound = parselmouth.Sound(
        np.asarray(audio, dtype=np.float64),
        sampling_frequency=float(sample_rate),
    )

    formant = sound.to_formant_burg(
        time_step=formant_config["time_step_s"],
        max_number_of_formants=formant_config["max_number_of_formants"],
        maximum_formant=ceiling,
        window_length=formant_config["window_length_s"],
        pre_emphasis_from=formant_config["pre_emphasis_from_hz"],
    )

    times_s = np.asarray(formant.xs(), dtype=np.float64)
    time_step_s = float(formant.dx)

    # The speech and pitch masks are mapped using these times, so invalid
    # or unordered frame centres must be rejected.
    if (
        times_s.ndim != 1
        or times_s.size == 0
        or not np.isfinite(times_s).all()
        or np.any(np.diff(times_s) <= 0)
    ):
        raise ValueError("Praat returned invalid formant-frame times.")

    if not np.isfinite(time_step_s) or time_step_s <= 0:
        raise ValueError("Praat returned an invalid formant time step.")

    # Arrays have one row per frame and one column per formant (F1-F3).
    # Missing Praat estimates remain NaN rather than being filled with zero.
    formants_hz = np.full((len(times_s), N_FORMANTS), np.nan)
    bandwidths_hz = np.full((len(times_s), N_FORMANTS), np.nan)

    # Each value is read at the frame's own centre time, so Praat returns
    # that frame's estimate with no interpolation between frames. A formant
    # the frame does not have comes back as NaN.
    for frame_index, time_s in enumerate(times_s):
        for formant_index in range(N_FORMANTS):
            number = formant_index + 1  # Praat numbers formants from 1, not 0.
            formants_hz[frame_index, formant_index] = formant.get_value_at_time(
                number, float(time_s)
            )
            bandwidths_hz[frame_index, formant_index] = formant.get_bandwidth_at_time(
                number, float(time_s)
            )

    return times_s, formants_hz, bandwidths_hz, time_step_s


def _voiced_at_times(
    pitch_track: PitchTrack, 
    times_s: np.ndarray,
) -> np.ndarray:
    """A Boolean array marking which formant frames are considered voiced.

    Returns one Boolean per formant frame. When a formant frame falls exactly 
    between two pitch frames, BOTH must be voiced. An aligned frame needs
    only its matching pitch frame. This avoids nearest-neighbour decisions 
    depending on floating-point rounding. Times with no pitch frame on one
    side count as unvoiced. Actual F0 magnitudes are not used.
    """

    voiced = np.zeros(len(times_s), dtype=bool)
    if len(pitch_track.times_s) == 0 or len(times_s) == 0:
        return voiced

    # Fractional pitch-frame position: 4.0 aligns with frame 4, while 4.5
    # falls halfway between pitch frames 4 and 5.
    position = (times_s - pitch_track.times_s[0]) / pitch_track.time_step_s

    # First and last pitch frames within half a step. These are the same
    # frame unless the time lies midway between two frames.
    first = np.ceil(position - 0.5 - GRID_TOLERANCE).astype(int)
    last = np.floor(position + 0.5 + GRID_TOLERANCE).astype(int)

    covered = (
        (first >= 0)
        & (last < len(pitch_track.times_s))
        & (first <= last)
    )

    # Aligned times use one pitch frame; midpoints require both neighbours
    # to be voiced, excluding uncertain voiced/unvoiced transitions.
    voiced[covered] = (
        pitch_track.voiced[first[covered]] 
        & pitch_track.voiced[last[covered]]
    )

    return voiced


def extract_formant_track(
    audio: np.ndarray,
    sample_rate: int,
    formant_config: dict,
    speech_mask_config: dict,
) -> FormantTrack:
    """Extract frame-level formants and select usable frames.

    Takes processed audio, sample rate and both configurations. Returns a
    FormantTrack with all Burg estimates, speech/voicing masks and a combined
    selection mask. Rejected frames remain available for inspection.
    """

    _validate_config(formant_config)

    # Speech detection and pitch tracking run on the full waveform.
    # Pitch contributes only voiced/unvoiced decisions, not formant values.
    intervals = speech_intervals(
        audio=audio,
        sample_rate=sample_rate,
        config=speech_mask_config,
    )
    
    pitch_track = extract_pitch_track(
        audio=audio,
        sample_rate=sample_rate,
        pitch_config=formant_config["voicing_pitch"],
        speech_mask_config=speech_mask_config,
    )

    times_s, formants_hz, bandwidths_hz, time_step_s = _run_praat_formants(
        audio=audio,
        sample_rate=sample_rate,
        formant_config=formant_config,
    )

    # Speech, pitch and formant frames can have different centre times.
    # Map both masks to the formant grid before combining them.
    in_speech = speech_mask_for_times(intervals=intervals, times=times_s)
    voiced = _voiced_at_times(pitch_track, times_s)

    # Require all three formants to be present, positive and increasing.
    # No additional frequency or bandwidth cutoffs are applied yet.
    valid_formants = (
        np.all(np.isfinite(formants_hz) & (formants_hz > 0), axis=1)
        & np.all(np.diff(formants_hz, axis=1) > 0, axis=1)
    )
    used = in_speech & voiced & valid_formants

    return FormantTrack(
        times_s=times_s,
        formants_hz=formants_hz,
        bandwidths_hz=bandwidths_hz,
        in_speech=in_speech,
        voiced=voiced,
        valid_formants=valid_formants,
        used=used,
        time_step_s=time_step_s,
    )


def summarise_formants(track: FormantTrack, formant_config: dict) -> dict:
    """Calculate recording-level F1-F3 statistics from selected frames.

    Returns median, P10, P90, usable-frame count and usable fraction.
    Raises InsufficientFormantFramesError when the coverage is too low.
    """

    n_speech_frames = int(np.count_nonzero(track.in_speech))
    n_voiced_frames = int(np.count_nonzero(track.in_speech & track.voiced))
    n_used_frames = int(np.count_nonzero(track.used))

    if n_speech_frames == 0:
        raise InsufficientFormantFramesError("No formant frames fall inside speech intervals.")

    # The minimum uses cumulative usable frames, not one continuous segment.
    # For a 10 ms step, 1.0 s of usable material requires 100 frames.
    min_frames = int(np.ceil(round(
        formant_config["min_usable_formant_s"] / track.time_step_s, 9
    )))
    if n_used_frames == 0 or n_used_frames < min_frames:
        raise InsufficientFormantFramesError(
            f"Insufficient usable formant frames: {n_used_frames} used, "
            f"{n_voiced_frames} speech-active voiced, {n_speech_frames} speech-active; "
            f"minimum {min_frames} ({formant_config['min_usable_formant_s']:.2f} s)."
        )

    # Percentiles are calculated per formant (each array column) using
    # the same selected frames for F1, F2 and F3.
    p10_hz, median_hz, p90_hz = np.percentile(
        track.formants_hz[track.used],
        [LOWER_PERCENTILE, 50, UPPER_PERCENTILE],
        axis=0,
    )

    summary = {}
    for index in range(N_FORMANTS):
        label = f"f{index + 1}"
        summary[f"{label}_median_hz"] = float(median_hz[index])
        summary[f"{label}_p10_hz"] = float(p10_hz[index])
        summary[f"{label}_p90_hz"] = float(p90_hz[index])

    summary["n_formant_frames"] = n_used_frames
    summary["formant_frame_fraction"] = float(n_used_frames / n_speech_frames)
    return summary


def feature_columns(config: dict) -> list[str]:
    """Return the formant output columns required by the DSP runner."""

    if "feature" not in config or "speech_mask" not in config:
        raise KeyError("Formants require feature and speech_mask config sections.")

    _validate_config(config["feature"])
    return list(FEATURE_COLUMNS)


def extract(
    audio: np.ndarray, 
    sample_rate: int, 
    row, 
    config: dict
) -> dict:
    """Extract and summarise formants for one recording in the DSP pipeline.

    The row argument is required by the shared interface but is unused.
    Formant extraction does not depend on manifest sex or accent labels.
    """

    track = extract_formant_track(
        audio=audio,
        sample_rate=sample_rate,
        formant_config=config["feature"],
        speech_mask_config=config["speech_mask"],
    )
    return summarise_formants(track, config["feature"])
