"""
Transcript-free acoustic speech-rate features.

Syllable nuclei are estimated from intensity peaks and voicedness using an
adaptation of the de Jong & Wempe (2009) / later Praat-script approach.


The project-wide shared speech mask supplies the speech-active intervals 
used for timing:

- first-to-last speech span,
- active-speech duration,
- pause timing,
- restricting candidate syllable nuclei to detected speech.

The de Jong/Wempe-style processing is used only to estimate the number and
locations of syllable nuclei.

Reference:
    N. H. de Jong and T. Wempe, "Praat script to detect syllable nuclei and
    measure speech rate automatically," Behavior Research Methods, vol. 41,
    no. 2, pp. 385-390, 2009.
"""

import numpy as np
import parselmouth

from src.dsp.speech_mask import speech_intervals


FEATURE_NAME = "speech_rate_acoustic"
FEATURE_VERSION = "1"


SPEECH_EXCLUSION_MESSAGES = (
    "Insufficient speech",
    "Audio contains only digital silence",
)


class NoUsableSpeech(ValueError):
    """
    Deterministic exclusion because no usable speech was detected.
    """


class TooFewSyllables(ValueError):
    """
    Deterministic exclusion because too few syllable nuclei were detected.
    """


# Does not duplicate the pause columns from the speech_rate_refrence module
def feature_columns(config: dict) -> list[str]:
    return [
        "rate_ac_n_syllables",
        "rate_ac_speech_span_s",
        "rate_ac_active_speech_s",
        "rate_ac_speaking_syl_per_s",
        "rate_ac_articulation_syl_per_s",
        "rate_ac_active_speech_fraction",
    ]


def _shared_intervals(
    audio: np.ndarray,
    sample_rate: int,
    config: dict,
) -> list[tuple[float, float]]:
    """
    Return shared speech intervals, mapping deterministic exclusions.
    """

    try:
        return speech_intervals(
            audio=audio,
            sample_rate=sample_rate,
            config=config["speech_mask"],
        )
    except ValueError as error:
        if str(error).startswith(SPEECH_EXCLUSION_MESSAGES):
            raise NoUsableSpeech(str(error)) from error
        raise


def _time_in_intervals(
    time_s: float,
    intervals: list[tuple[float, float]],
) -> bool:
    """
    Return whether a peak time occurs during interval that the 
    speech mask considers speech.
    """

    return any(
        start_s <= time_s <= end_s
        for start_s, end_s in intervals
    )


def _candidate_intensity_peaks(
    values: np.ndarray,
    threshold_db: float,
) -> np.ndarray:
    """
    Return local-maximum indices above the intensity threshold.

    Higher than the frame before and at least as high as the
    frame after.
    """

    if values.ndim != 1:
        raise ValueError("Intensity values must be one-dimensional.")

    if len(values) < 3:
        return np.array([], dtype=int)

    is_peak = (
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
    )

    peaks = np.where(is_peak)[0] + 1

    # removes peaks that are below an intensity threshold
    return peaks[values[peaks] > threshold_db]


def syllable_nuclei(
    values: np.ndarray,
    times: np.ndarray,
    threshold_db: float,
    min_dip_db: float,
    intervals: list[tuple[float, float]],
    pitch: parselmouth.Pitch,
) -> np.ndarray:
    """
    Return times of accepted syllable nuclei.

    Candidate nuclei must:
    1. be local intensity maxima above the threshold,
    2. have a sufficient intensity dip relative to the preceding candidate peak,
    3. fall inside a shared speech-active interval, and
    4. be voiced according to Praat pitch estimation.
    """

    # 1. Gets all the local maxima
    peaks = _candidate_intensity_peaks(
        values=values,
        threshold_db=threshold_db,
    )

    if len(peaks) == 0:
        return np.array([], dtype=float)

    nuclei = []

    for k, peak in enumerate(peaks):
        # De Jong & Wempe use the dip between the current peak and the
        # preceding candidate peak to suppress multiple peaks belonging to
        # the same syllable.
        if k == 0:
            # first peak
            dip_start = 0
        else:
            # previous peak
            dip_start = peaks[k - 1]

        # current peak less than previous
        if peak <= dip_start:
            continue


        # 2. The dip before the current peak must be above the min dip db
        # small ripples in a vowel should not be counted separately
        preceding_min = np.min(values[dip_start:peak])
        dip_db = values[peak] - preceding_min

        if dip_db < min_dip_db:
            continue

        # 3. Ensures peak is within a speech interval from the speech 
        # mask
        time_s = float(times[peak])

        if not _time_in_intervals(
            time_s=time_s,
            intervals=intervals,
        ):
            continue


        # 4. Nucleus must be where praat finds a pitch
        # i.e. it is a voiced region
        pitch_hz = pitch.get_value_at_time(time_s)

        if pitch_hz is None or not np.isfinite(pitch_hz):
            continue

        nuclei.append(time_s)

    return np.asarray(nuclei, dtype=float)


def extract(
    audio,
    sample_rate,
    row,
    config,
) -> dict[str, float]:
    settings = config["feature"]

    intervals = _shared_intervals(
        audio=audio,
        sample_rate=sample_rate,
        config=config,
    )

    speech_span_s = intervals[-1][1] - intervals[0][0]
    active_speech_s = sum(
        end_s - start_s
        for start_s, end_s in intervals
    )

    if speech_span_s <= 0:
        raise NoUsableSpeech("Detected speech span is zero.")

    if active_speech_s <= 0:
        raise NoUsableSpeech("Detected active-speech duration is zero.")

    # creates a Praat compatible object so Parselmouth can access it
    sound = parselmouth.Sound(
        np.asarray(audio, dtype=np.float64),
        sampling_frequency=sample_rate,
    )

    # The original/v2 Praat approach uses a 50 Hz minimum pitch for the
    # intensity contour. time_step = 0 lets Praat choose its standard step.
    intensity = sound.to_intensity(
        minimum_pitch=settings["intensity_min_pitch_hz"], 
        time_step=settings["intensity_time_step_s"],
    )

    values = np.asarray(intensity.values[0], dtype=float)   # intensity in dB
    times = np.asarray(intensity.xs(), dtype=float)         # corresponding times in seconds

    finite = np.isfinite(values)

    if not np.any(finite):
        raise ValueError("Intensity contour contains no finite values.")


    # Sets the intensity threshold (from 2010 v2 script)
    # 99th percentile intensity (loudest part of recording)
    # threshold is that reference - 25 dB
    reference_db = float(
        np.quantile(
            values[finite],
            settings["intensity_reference_quantile"],
        )
    )

    threshold_db = (
        reference_db
        + settings["silence_threshold_db"]
    )

    # A later Praat-script safeguard design is to not place the peak 
    # threshold below the minimum finite intensity of the recording.
    threshold_db = max(
        threshold_db,
        float(np.min(values[finite])),
    )


    # Obtaining Praat's pitch to determine if a peak is likely
    # voiced.
    pitch = sound.to_pitch_ac(
        time_step=settings["pitch_time_step_s"],
        pitch_floor=settings["pitch_floor_hz"],
        pitch_ceiling=settings["pitch_ceiling_hz"],
    )

    # Determining the estimated syllable nuclei
    nuclei = syllable_nuclei(
        values=values,
        times=times,
        threshold_db=threshold_db,
        min_dip_db=settings["min_dip_db"],
        intervals=intervals,
        pitch=pitch,
    )

    n_syllables = len(nuclei)

    if n_syllables < settings["min_syllables"]:
        raise TooFewSyllables(
            f"Only {n_syllables} syllable nuclei found "
            f"(minimum {settings['min_syllables']})."
        )

    speaking_rate = n_syllables / speech_span_s
    articulation_rate = n_syllables / active_speech_s
    active_fraction = active_speech_s / speech_span_s

    if articulation_rate + 1e-12 < speaking_rate:
        raise ValueError(
            "Articulation rate is lower than speaking rate; "
            "speech timing is inconsistent."
        )

    if not 0.0 <= active_fraction <= 1.0 + 1e-9:
        raise ValueError(
            f"Active-speech fraction is outside [0, 1]: {active_fraction}."
        )

    return {
        "rate_ac_n_syllables": float(n_syllables),
        "rate_ac_speech_span_s": float(speech_span_s),
        "rate_ac_active_speech_s": float(active_speech_s),
        "rate_ac_speaking_syl_per_s": float(speaking_rate),
        "rate_ac_articulation_syl_per_s": float(articulation_rate),
        "rate_ac_active_speech_fraction": float(active_fraction),
    }
