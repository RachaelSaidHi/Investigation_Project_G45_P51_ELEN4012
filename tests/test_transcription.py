"""
Tests for the transcription functionality
"""

from unittest.mock import Mock

import numpy as np
import pytest

import src.asr.transcribe as transcribe_module
from src.asr.transcribe import (
    SAMPLE_RATE,
    load_audio
)

# TODO: if this gets moved to a preprocessing file then move these tests as well

class TestLoadAudio:

    def test_none_filepath_fails(self):
        with pytest.raises(ValueError):
            load_audio(None)

    def test_empty_filepath_fails(self):
        with pytest.raises(ValueError):
            load_audio("")

    def test_missing_file_fails(self, tmp_path):
        missing = tmp_path / "missing.wav"

        with pytest.raises(FileNotFoundError):
            load_audio(str(missing))

    def test_directory_fails(self, tmp_path):
        with pytest.raises(ValueError):
            load_audio(str(tmp_path))

    def test_empty_audio_fails(
        self,
        tmp_path,
        monkeypatch,
    ):
        audio_file = tmp_path / "empty.wav"
        audio_file.touch()

        monkeypatch.setattr(
            transcribe_module.librosa,
            "load",
            Mock(return_value = (np.array([], dtype=np.float32), SAMPLE_RATE)),
        )

        with pytest.raises(
            ValueError,
            match="Audio file is empty",
        ):
            load_audio(str(audio_file))

    def test_duration_calculated_from_resampled_audio(
        self,
        tmp_path,
        monkeypatch,
    ):
        audio_file = tmp_path / "sample.wav"
        audio_file.touch()

        # Exactly two seconds at 16 kHz.
        audio = np.zeros(
            2 * SAMPLE_RATE,
            dtype = np.float32,
        )

        mock_load = Mock(return_value = (audio, SAMPLE_RATE))

        monkeypatch.setattr(
            transcribe_module.librosa,
            "load",
            mock_load,
        )

        loaded_audio, duration_s = load_audio(str(audio_file))

        assert len(loaded_audio) == 2 * SAMPLE_RATE
        assert duration_s == pytest.approx(2.0)

        mock_load.assert_called_once_with(
            str(audio_file),
            sr=SAMPLE_RATE,
            mono=True,
        )

    def test_silent_audio_is_allowed(
        self,
        tmp_path,
        monkeypatch,
    ):
        audio_file = tmp_path / "silent.wav"
        audio_file.touch()

        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        monkeypatch.setattr(
            transcribe_module.librosa,
            "load",
            Mock(
                return_value=(audio, SAMPLE_RATE)
            ),
        )

        loaded_audio, duration_s = load_audio(str(audio_file))

        assert len(loaded_audio) == SAMPLE_RATE
        assert duration_s == pytest.approx(1.0)

    
    def test_non_finite_audio_fails(
        self,
        tmp_path,
        monkeypatch,
    ):
        audio_file = tmp_path / "invalid.wav"
        audio_file.touch()

        audio = np.array([0.0, np.nan, 0.2], dtype=np.float32,)

        monkeypatch.setattr(
            transcribe_module.librosa,
            "load",
            Mock(return_value=(audio, SAMPLE_RATE)),
        )

        with pytest.raises(
            ValueError,
            match="NaN or infinite",
        ):
            load_audio(str(audio_file))


# run using python -m pytest tests/test_transcription.py -v