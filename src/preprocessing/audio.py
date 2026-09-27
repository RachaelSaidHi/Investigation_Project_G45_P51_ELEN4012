"""Convert raw recordings to the standard audio representation for shared analysis pipeline.

Every study recording is converted to mono, 16 kHz, float32 audio so that
the DSP and ASR branches operate on an identical signal.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# Both the DSP and ASR pipelines will use the same 16 kHz mono audio.
# Whisper requires 16 kHz
TARGET_SAMPLE_RATE = 16000


def load_and_prepare_audio(audio_path: Path) -> np.ndarray:
    """Load one audio file and convert it to 16 kHz mono."""

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    # Keep the original sample rate and channel structure when decoding
    # so that the preprocessing steps are applied explicitly below.
    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=False,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            f"Audio file contains no samples: {audio_path}"
        )

    # Librosa returns mono audio as (samples,) and multichannel audio as
    # (channels, samples). Downmix multichannel recordings to one channel.
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    # resample mono to TARGET_SAMPLE_RATE
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="soxr_hq",
        )

    return np.asarray(
        audio,
        dtype=np.float32,
    )


def save_processed_audio(audio: np.ndarray, output_path: Path) -> None:
    """Save prepared mono audio as a 32-bit floating-point WAV file."""

    if audio.ndim != 1:
        raise ValueError(
            "Processed audio must be mono before it is saved."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Float WAV avoids introducing an additional integer quantisation step
    # after decoding and resampling. No amplitude normalisation is applied.
    sf.write(
        output_path,
        audio,
        TARGET_SAMPLE_RATE,
        format="WAV",
        subtype="FLOAT",
    )


def preprocess_audio_file(audio_path: Path, output_path: Path) -> np.ndarray:
    """Prepare one raw audio file and save the processed WAV."""

    audio = load_and_prepare_audio(
        audio_path
    )

    save_processed_audio(
        audio,
        output_path,
    )

    return audio