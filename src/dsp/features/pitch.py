"""Extract recording-level pitch features (F0) using Praat raw autocorrelation.

Runs Praat's autocorrelation pitch tracker (Boersma, 1993) through
Parselmouth on the full processed waveform. Only frames that are both
inside a shared speech interval and voiced by Praat are summarised.

The same pitch range is used for every recording. The manifest sex label
is deliberately not used, so pitch extraction does not depend on
source-provided sex metadata.

Feature module for src/dsp/extraction_pipeline.py. Settings come from
configs/dsp/pitch.json and configs/dsp/speech_mask.json.
"""

from dataclasses import dataclass

import numpy as np
import parselmouth

from src.dsp.speech_mask import (
    speech_intervals,
    speech_mask_for_times,
)


FEATURE_NAME = "pitch"
FEATURE_VERSION = "1"

# Fixed here rather than in pitch.json because the output column names
# (f0_p10_hz, f0_p90_hz) depend on them.
LOWER_PERCENTILE = 10
UPPER_PERCENTILE = 90

SEMITONES_PER_OCTAVE = 12

FEATURE_COLUMNS = [
    "f0_median_hz",
    "f0_span_st",
    "f0_p10_hz",
    "f0_p90_hz",
    "n_voiced_frames",
    "voiced_fraction",
]

REQUIRED_CONFIG_KEYS = (
    "time_step_s",
    "pitch_floor_hz",
    "pitch_ceiling_hz",
    "max_number_of_candidates",
    "very_accurate",
    "silence_threshold",
    "voicing_threshold",
    "octave_cost",
    "octave_jump_cost",
    "voiced_unvoiced_cost",
    "min_voiced_s",
)


class InsufficientVoicingError(ValueError):
    """Too little voiced speech to summarise pitch for this recording."""


@dataclass(frozen=True)
class PitchTrack:
    """Frame-level pitch information on Praat's own time grid."""

    times_s: np.ndarray     # Praat pitch-frame times
    f0_hz: np.ndarray       # Praat F0 values; unvoiced frames are 0 Hz
    in_speech: np.ndarray   # whether each pitch frame lies inside speech
    voiced: np.ndarray      # whether Praat assigned the frame a positive F0
    used: np.ndarray        # frames that are both speech-active and voiced
    time_step_s: float      # actual spacing between Praat pitch frames


def _validate_config(pitch_config: dict) -> None:
    """Check the pitch configuration before analysis."""

    missing = [
        key
        for key in REQUIRED_CONFIG_KEYS
        if key not in pitch_config
    ]

    if missing:
        raise KeyError(
            f"Pitch config is missing: {', '.join(missing)}"
        )

    if pitch_config["time_step_s"] <= 0:
        raise ValueError(
            "time_step_s must be greater than zero."
        )

    if pitch_config["pitch_floor_hz"] <= 0:
        raise ValueError(
            "pitch_floor_hz must be greater than zero."
        )

    if (
        pitch_config["pitch_ceiling_hz"]
        <= pitch_config["pitch_floor_hz"]
    ):
        raise ValueError(
            "pitch_ceiling_hz must be greater than pitch_floor_hz."
        )

    max_candidates = pitch_config[
        "max_number_of_candidates"
    ]

    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates < 2
    ):
        raise ValueError(
            "max_number_of_candidates must be an integer of at least 2."
        )

    if not isinstance(
        pitch_config["very_accurate"],
        bool,
    ):
        raise ValueError(
            "very_accurate must be a Boolean."
        )

    if not 0 <= pitch_config["silence_threshold"] <= 1:
        raise ValueError(
            "silence_threshold must be between 0 and 1."
        )

    if not 0 <= pitch_config["voicing_threshold"] <= 1:
        raise ValueError(
            "voicing_threshold must be between 0 and 1."
        )

    for key in (
        "octave_cost",
        "octave_jump_cost",
        "voiced_unvoiced_cost",
        "min_voiced_s",
    ):
        if pitch_config[key] < 0:
            raise ValueError(
                f"{key} must be non-negative."
            )


def _minimum_voiced_frames(
    min_voiced_s: float,
    time_step_s: float,
) -> int:
    """Convert the minimum voiced duration to a whole number of frames."""

    # Rounding removes small floating-point errors at exact boundaries.
    # For example, 1.0 s at a 0.01 s step should require exactly 100 frames.
    frames = round(
        min_voiced_s / time_step_s,
        9,
    )

    return int(
        np.ceil(frames)
    )


def extract_pitch_track(
    audio: np.ndarray,
    sample_rate: int,
    pitch_config: dict,
    speech_mask_config: dict,
) -> PitchTrack:
    """
    Return Praat's frame-level pitch track and frame-selection masks.
    
    Praat analyses the complete waveform. The shared speech mask is then
    mapped onto Praat's own frame times so the pitch tracker does not need
    to use the same frame grid as the energy-based speech detector.
    
    Returns PitchTrack object
    """

    _validate_config(
        pitch_config
    )

    # Speech detector runs on the complete waveform.
    # Intervals are used only to decide which Praat frames contribute
    # to the recording-level pitch summaries.
    intervals = speech_intervals(
        audio=audio,
        sample_rate=sample_rate,
        config=speech_mask_config,
    )

    # Praat analyses the complete waveform so its pitch-path search retains
    # the natural temporal context across the recording.
    sound = parselmouth.Sound(
        np.asarray(
            audio,
            dtype=np.float64,
        ),
        sampling_frequency=float(sample_rate),
    )

    pitch = sound.to_pitch_ac(
        time_step=pitch_config["time_step_s"],
        pitch_floor=pitch_config["pitch_floor_hz"],
        max_number_of_candidates=pitch_config[
            "max_number_of_candidates"
        ],
        very_accurate=pitch_config["very_accurate"],
        silence_threshold=pitch_config["silence_threshold"],
        voicing_threshold=pitch_config["voicing_threshold"],
        octave_cost=pitch_config["octave_cost"],
        octave_jump_cost=pitch_config["octave_jump_cost"],
        voiced_unvoiced_cost=pitch_config[
            "voiced_unvoiced_cost"
        ],
        pitch_ceiling=pitch_config["pitch_ceiling_hz"],
    )

    times_s = np.asarray(
        pitch.xs(),
        dtype=np.float64,
    )

    f0_hz = np.asarray(
        pitch.selected_array["frequency"],
        dtype=np.float64,
    )

    if times_s.ndim != 1 or f0_hz.ndim != 1:
        raise ValueError(
            "Praat pitch output must be one-dimensional."
        )

    if times_s.shape != f0_hz.shape:
        raise ValueError(
            "Praat pitch times and F0 values have different lengths."
        )

    if not np.isfinite(times_s).all():
        raise ValueError(
            "Praat pitch times contain NaN or infinite values."
        )

    if not np.isfinite(f0_hz).all():
        raise ValueError(
            "Praat pitch values contain NaN or infinite values."
        )

    if np.any(f0_hz < 0):
        raise ValueError(
            "Praat pitch values contain negative frequencies."
        )

    time_step_s = float(
        pitch.dx
    )

    if (
        not np.isfinite(time_step_s)
        or time_step_s <= 0
    ):
        raise ValueError(
            "Praat returned an invalid pitch time step."
        )

    in_speech = speech_mask_for_times(
        intervals=intervals,
        times=times_s,
    )

    # Praat represents an unvoiced selected candidate as 0 Hz.
    voiced = (
        f0_hz > 0
    )

    used = (
        in_speech
        & voiced
    )

    return PitchTrack(
        times_s=times_s,
        f0_hz=f0_hz,
        in_speech=in_speech,
        voiced=voiced,
        used=used,
        time_step_s=time_step_s,
    )


def summarise_pitch(
    track: PitchTrack,
    pitch_config: dict,
) -> dict:
    """Summarise the usable pitch frames for one recording."""

    n_speech_frames = int(
        np.count_nonzero(
            track.in_speech
        )
    )

    n_voiced_frames = int(
        np.count_nonzero(
            track.used
        )
    )

    if n_speech_frames == 0:
        raise InsufficientVoicingError(
            "No Praat pitch frames fall inside the detected speech intervals."
        )

    min_voiced_frames = _minimum_voiced_frames(
        min_voiced_s=pitch_config["min_voiced_s"],
        time_step_s=track.time_step_s,
    )

    if (
        n_voiced_frames == 0
        or n_voiced_frames < min_voiced_frames
    ):
        raise InsufficientVoicingError(
            "Insufficient voiced speech: "
            f"{n_voiced_frames} voiced frames out of "
            f"{n_speech_frames} speech-active pitch frames; "
            f"minimum is {min_voiced_frames} frames "
            f"({pitch_config['min_voiced_s']:.2f} s)."
        )

    used_f0_hz = track.f0_hz[
        track.used
    ]

    f0_p10_hz, f0_median_hz, f0_p90_hz = np.percentile(
        used_f0_hz,
        [
            LOWER_PERCENTILE,
            50,
            UPPER_PERCENTILE,
        ],
    )

    # P90/P10 is converted to semitones so pitch span represents a
    # proportional frequency range rather than an absolute difference in Hz.
    f0_span_st = (
        SEMITONES_PER_OCTAVE
        * np.log2(
            f0_p90_hz
            / f0_p10_hz
        )
    )

    voiced_fraction = (
        n_voiced_frames
        / n_speech_frames
    )

    return {
        "f0_median_hz": float(f0_median_hz),
        "f0_span_st": float(f0_span_st),
        "f0_p10_hz": float(f0_p10_hz),
        "f0_p90_hz": float(f0_p90_hz),
        "n_voiced_frames": n_voiced_frames,
        "voiced_fraction": float(voiced_fraction),
    }


def feature_columns(config: dict) -> list[str]:
    """Return the pitch columns written to the results file."""

    if "feature" not in config:
        raise KeyError(
            "Pitch config is missing the 'feature' section."
        )

    if "speech_mask" not in config:
        raise KeyError(
            "Pitch config is missing the 'speech_mask' section."
        )

    # feature_columns() is called before the extraction run starts, so
    # validating here makes a bad pitch config fail immediately rather
    # than producing the same error row for every recording.
    _validate_config(
        config["feature"]
    )

    return list(
        FEATURE_COLUMNS
    )


def extract(
    audio: np.ndarray,
    sample_rate: int,
    row,
    config: dict,
) -> dict:
    """Extract recording-level pitch features for one sample."""

    # row is required by the shared feature-module interface but is
    # deliberately unused: pitch extraction must not depend on metadata
    # such as the source-provided sex label.
    track = extract_pitch_track(
        audio=audio,
        sample_rate=sample_rate,
        pitch_config=config["feature"],
        speech_mask_config=config["speech_mask"],
    )

    return summarise_pitch(
        track=track,
        pitch_config=config["feature"],
    )