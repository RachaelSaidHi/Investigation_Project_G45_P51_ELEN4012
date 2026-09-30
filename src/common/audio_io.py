"""Load processed analysis audio for the downstream pipelines.

Reads one preprocessed WAV (resolved from a manifest's processed_filepath,
relative to data/processed/) and checks that it has the format written by
src/preprocessing/audio.py: 16 kHz, mono, float WAV, finite and non-empty.

Returns audio waveform as one-dimensional NumPy float32 array
"""

from pathlib import Path

import numpy as np
import soundfile as sf


# Must equal TARGET_SAMPLE_RATE in src/preprocessing/audio.py. Defined here
# rather than imported so src/common does not depend on preprocessing modules.
SAMPLE_RATE = 16000
EXPECTED_FORMAT = "WAV"
EXPECTED_SUBTYPE = "FLOAT"


def load_processed_audio(
    processed_root: Path,
    processed_filepath: str,
) -> np.ndarray:
    """Load one processed recording as a 1-D float32 array."""

    # A blank processed_filepath is read by pandas as NaN. This probably 
    # means the full or study manifest was used instead of the analysis
    # or dev manifest, which contain processed audio paths.
    if (
        not isinstance(processed_filepath, str)
        or not processed_filepath.strip()
    ):
        raise ValueError(
            "processed_filepath is blank. Use the analysis or dev "
            "manifest, which point to preprocessed audio."
        )

    relative_path = Path(processed_filepath)

    if relative_path.is_absolute():
        raise ValueError(
            f"processed_filepath must be relative to data/processed/: "
            f"{processed_filepath}"
        )

    audio_path = Path(processed_root) / relative_path

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Processed audio file not found: {audio_path}"
        )

    # Inspect the stored file itself before reading it. Asking SoundFile to
    # return float32 would also convert integer PCM audio to float32, which
    # could otherwise hide an incorrect preprocessing output format.
    with sf.SoundFile(audio_path) as audio_file:
        if audio_file.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz audio, "
                f"found {audio_file.samplerate} Hz: {audio_path}"
            )

        if audio_file.channels != 1:
            raise ValueError(
                f"Expected mono audio, "
                f"found {audio_file.channels} channels: {audio_path}"
            )

        if audio_file.format != EXPECTED_FORMAT:
            raise ValueError(
                f"Expected {EXPECTED_FORMAT} audio, "
                f"found {audio_file.format}: {audio_path}"
            )

        if audio_file.subtype != EXPECTED_SUBTYPE:
            raise ValueError(
                f"Expected {EXPECTED_SUBTYPE} audio subtype, "
                f"found {audio_file.subtype}: {audio_path}"
            )

        audio = audio_file.read(
            dtype="float32",
            always_2d=False,
        )

    if audio.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional mono audio, "
            f"found shape {audio.shape}: {audio_path}"
        )

    if audio.size == 0:
        raise ValueError(
            f"Audio file contains no samples: {audio_path}"
        )

    # NaN or infinite samples would propagate silently into downstream
    # feature statistics such as means, variances and formant summaries.
    if not np.isfinite(audio).all():
        raise ValueError(
            f"Audio contains NaN or infinite samples: {audio_path}"
        )

    return audio