import librosa
import numpy as np

from src.dsp.speech_mask import (
    speech_intervals,
    speech_mask_for_times,
)

FEATURE_NAME = "mfcc"
FEATURE_VERSION = "3"

STATS = ("mean", "std")

# Errors from speech_intervals() that mean "no usable speech" (a deterministic
# exclusion), as opposed to a genuine failure that should be retried.
SPEECH_EXCLUSION_MESSAGES = (
    "Insufficient speech",
    "Audio contains only digital silence",
)

class NoUsableSpeech(ValueError):
    """
    Deterministic exclusion because the recording contains
    insufficient usable speech.
    """

def feature_columns(config):
    n = config["feature"]["n_mfcc"]
    return [
        f"mfcc_{i:02d}_{stat}"
        for i in range(n)
        for stat in STATS
    ]

def extract(
    audio,
    sample_rate,
    row,
    config,
):
    
    settings = config["feature"]
    
    if not isinstance(settings["center"], bool):
        raise TypeError(
            f"'center' must be true/false, got {settings['center']!r}."
        )
    if settings["power"] != 2.0:
        raise ValueError(
            "power must be 2.0 for the fixed MFCC extraction configuration."
        )

    # Detect shared speech-active intervals
    # Done early so clips with no usable audio fail fast          
    try:
        intervals = speech_intervals(
            audio=audio, 
            sample_rate=sample_rate, 
            config=config["speech_mask"]
        )
    except ValueError as error:
        if str(error).startswith(SPEECH_EXCLUSION_MESSAGES):
            raise NoUsableSpeech(str(error)) from error
        raise


    # Extract MFCCs over the full waveform
    # shape: (n_mfcc, n_frames)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,

        # Framing / FFT
        n_fft=settings["n_fft"],
        win_length=settings["win_length"],
        hop_length=settings["hop_length"],
        window=settings["window"],
        center=settings["center"],

        # Mel filterbank
        n_mels=settings["n_mels"],
        fmin=settings["fmin"],
        fmax=settings["fmax"],
        power=settings["power"],

        # MFCC
        n_mfcc=settings["n_mfcc"],
        dct_type=settings["dct_type"],
        norm=settings["norm"],
        lifter=settings["lifter"],
    )                 


    # Determine the centre time of every MFCC frame, to match 
    # against the speech intervals.
    frame_numbers = np.arange(mfcc.shape[1])       

    if settings["center"]:
        feature_times = librosa.frames_to_time(
            frame_numbers,
            sr=sample_rate,
            hop_length=settings["hop_length"]
        )
    else:
        feature_times = librosa.frames_to_time(
            frame_numbers,
            sr=sample_rate,
            hop_length=settings["hop_length"],
            n_fft=settings["n_fft"]
        )    


    # Map the shared speech intervals onto the MFCC frame grid
    speech = speech_mask_for_times(
        intervals=intervals, 
        times=feature_times
    )

    frames = mfcc[:, speech]

    if frames.shape[1] == 0:
        raise NoUsableSpeech(
            "No MFCC frames fall inside the detected speech intervals."
        )

    # Summarise speech-active MFCC frames
    values = {}

    for i in range(frames.shape[0]):
        values[f"mfcc_{i:02d}_mean"] = float(
            frames[i].mean()
        )

        values[f"mfcc_{i:02d}_std"] = float(
            frames[i].std()
        )

    return values