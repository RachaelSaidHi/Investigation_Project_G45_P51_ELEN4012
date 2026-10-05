import librosa
import numpy as np

FEATURE_NAME = "mfcc"
FEATURE_VERSION = "1"

def feature_columns(config):
    n = config["feature"]["n_mfcc"]
    return [
        f"mfcc_{i:02d}_{stat}"
        for i in range(n)
        for stat in ("mean", "var")
    ]

def extract(audio, sample_rate, row, config):
    settings = config["feature"]

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

    # TODO: replace with the shared speech mask once speech_mask.py is merged
    speech = np.ones(mfcc.shape[1], dtype=bool)

    frames = mfcc[:, speech]
    
    if frames.shape[1] < settings["min_speech_frames"]:
        raise ValueError(f"Only {frames.shape[1]} speech frames (minimum {settings['min_speech_frames']}).")

    values = {}
    for i in range(frames.shape[0]):
        values[f"mfcc_{i:02d}_mean"] = float(frames[i].mean())
        values[f"mfcc_{i:02d}_var"] = float(frames[i].var())
    return values