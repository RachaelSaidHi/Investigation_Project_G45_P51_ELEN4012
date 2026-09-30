"""Tests for loading processed analysis audio."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.common.audio_io import (
    EXPECTED_FORMAT,
    EXPECTED_SUBTYPE,
    SAMPLE_RATE,
    load_processed_audio,
)
from src.preprocessing.audio import (
    TARGET_SAMPLE_RATE,
    save_processed_audio,
)


DEV_MANIFEST_PATH = Path("data/manifests/dev_manifest.csv")
PROCESSED_ROOT = Path("data/processed")


def write_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    subtype: str = EXPECTED_SUBTYPE,
) -> None:
    """Write a synthetic WAV file for testing."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        path,
        audio,
        sample_rate,
        subtype=subtype,
    )


def make_tone(
    duration_s: float = 0.25,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.2,
) -> np.ndarray:
    """Create a short synthetic 440 Hz tone."""

    time = (
        np.arange(int(sample_rate * duration_s))
        / sample_rate
    )

    return (
        amplitude
        * np.sin(2 * np.pi * 440 * time)
    ).astype(np.float32)


class TestSampleRateConsistency:

    def test_sample_rate_matches_preprocessing(self):
        assert SAMPLE_RATE == TARGET_SAMPLE_RATE


class TestPathHandling:

    @pytest.mark.parametrize(
        "blank_path",
        [
            float("nan"),
            None,
            "",
            "   ",
        ],
    )
    def test_blank_processed_filepath_raises_error(
        self,
        tmp_path,
        blank_path,
    ):
        with pytest.raises(
            ValueError,
            match="processed_filepath is blank",
        ):
            load_processed_audio(
                tmp_path,
                blank_path,
            )

    def test_missing_file_raises_error(self, tmp_path):
        with pytest.raises(
            FileNotFoundError,
            match="Processed audio file not found",
        ):
            load_processed_audio(
                tmp_path,
                "audio/missing.wav",
            )

    def test_absolute_processed_filepath_raises_error(self, tmp_path):
        audio_path = tmp_path / "example.wav"

        write_wav(
            audio_path,
            make_tone(),
        )

        with pytest.raises(
            ValueError,
            match="must be relative",
        ):
            load_processed_audio(
                tmp_path,
                str(audio_path.resolve()),
            )

    def test_path_is_resolved_relative_to_processed_root(self, tmp_path):
        tone = make_tone()

        write_wav(
            tmp_path / "audio" / "example.wav",
            tone,
        )

        audio = load_processed_audio(
            tmp_path,
            "audio/example.wav",
        )

        assert len(audio) == len(tone)


class TestValidAudio:

    def test_preprocessed_audio_is_returned_unchanged(self, tmp_path):
        tone = make_tone(
            amplitude=0.2,
        )
        audio_path = tmp_path / "example.wav"

        save_processed_audio(
            tone,
            audio_path,
        )

        audio = load_processed_audio(
            tmp_path,
            "example.wav",
        )

        assert audio.ndim == 1
        assert audio.dtype == np.float32
        assert np.array_equal(
            audio,
            tone,
        )

    def test_silence_is_not_trimmed(self, tmp_path):
        silence = np.zeros(
            1000,
            dtype=np.float32,
        )
        tone = make_tone()

        signal = np.concatenate(
            [
                silence,
                tone,
                silence,
            ]
        )

        write_wav(
            tmp_path / "audio" / "example.wav",
            signal,
        )

        audio = load_processed_audio(
            tmp_path,
            "audio/example.wav",
        )

        assert np.array_equal(
            audio,
            signal,
        )


class TestRejectedAudio:

    @pytest.mark.parametrize(
        "sample_rate",
        [
            8000,
            22050,
            44100,
        ],
    )
    def test_wrong_sample_rate_raises_error(
        self,
        tmp_path,
        sample_rate,
    ):
        write_wav(
            tmp_path / "audio" / "example.wav",
            make_tone(
                sample_rate=sample_rate,
            ),
            sample_rate=sample_rate,
        )

        with pytest.raises(
            ValueError,
            match="Expected 16000 Hz audio",
        ):
            load_processed_audio(
                tmp_path,
                "audio/example.wav",
            )

    def test_stereo_audio_raises_error(self, tmp_path):
        tone = make_tone()

        stereo = np.column_stack(
            [
                tone,
                tone,
            ]
        )

        write_wav(
            tmp_path / "audio" / "example.wav",
            stereo,
        )

        with pytest.raises(
            ValueError,
            match="Expected mono audio, found 2 channels",
        ):
            load_processed_audio(
                tmp_path,
                "audio/example.wav",
            )

    def test_non_float_wav_raises_error(self, tmp_path):
        write_wav(
            tmp_path / "audio" / "example.wav",
            make_tone(),
            subtype="PCM_16",
        )

        with pytest.raises(
            ValueError,
            match=f"Expected {EXPECTED_SUBTYPE} audio subtype",
        ):
            load_processed_audio(
                tmp_path,
                "audio/example.wav",
            )

    def test_non_wav_audio_raises_error(self, tmp_path):
        audio_path = tmp_path / "audio" / "example.flac"

        audio_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            audio_path,
            make_tone(),
            SAMPLE_RATE,
        )

        with pytest.raises(
            ValueError,
            match=f"Expected {EXPECTED_FORMAT} audio",
        ):
            load_processed_audio(
                tmp_path,
                "audio/example.flac",
            )

    def test_empty_audio_raises_error(self, tmp_path):
        write_wav(
            tmp_path / "audio" / "example.wav",
            np.array(
                [],
                dtype=np.float32,
            ),
        )

        with pytest.raises(
            ValueError,
            match="contains no samples",
        ):
            load_processed_audio(
                tmp_path,
                "audio/example.wav",
            )

    @pytest.mark.parametrize(
        "bad_value",
        [
            np.nan,
            np.inf,
            -np.inf,
        ],
    )
    def test_non_finite_samples_raise_error(
        self,
        tmp_path,
        bad_value,
    ):
        tone = make_tone()
        tone[100] = bad_value

        write_wav(
            tmp_path / "audio" / "example.wav",
            tone,
        )

        with pytest.raises(
            ValueError,
            match="NaN or infinite",
        ):
            load_processed_audio(
                tmp_path,
                "audio/example.wav",
            )


class TestRealDevAudio:

    @pytest.mark.skipif(
        not DEV_MANIFEST_PATH.exists(),
        reason="Dev manifest not available locally.",
    )
    def test_all_dev_recordings_load(self):
        manifest = pd.read_csv(
            DEV_MANIFEST_PATH,
        )

        for processed_filepath in manifest["processed_filepath"]:
            audio = load_processed_audio(
                PROCESSED_ROOT,
                processed_filepath,
            )

            assert audio.ndim == 1
            assert audio.dtype == np.float32
            assert audio.size > 0
            assert np.isfinite(audio).all()


# run using:
# python -m pytest tests/test_audio_io.py -v