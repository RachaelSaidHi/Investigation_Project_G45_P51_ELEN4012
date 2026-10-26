"""Tests for shared audio preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing.audio import (
    TARGET_SAMPLE_RATE,
    load_and_prepare_audio,
    preprocess_audio_file,
)


def write_test_wav(
    path: Path,
    sample_rate: int = 16000,
    duration_s: float = 0.25,
    stereo: bool = False,
    amplitude: float = 0.2,
) -> np.ndarray:
    """Create a small synthetic sine-wave WAV file for preprocessing tests."""

    sample_count = int(sample_rate * duration_s)
    time = np.arange(sample_count) / sample_rate

    audio = (
        amplitude
        * np.sin(2 * np.pi * 440 * time)
    ).astype(np.float32)

    # Use identical channels when creating stereo audio. This matches the
    # duplicated-channel structure observed in the AfriSpeech recordings.
    if stereo:
        audio_to_write = np.column_stack([audio, audio])
    else:
        audio_to_write = audio

    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        path,
        audio_to_write,
        sample_rate,
        subtype="FLOAT",
    )

    return audio


class TestLoadAndPrepareAudio:

    def test_missing_audio_raises_error(self, tmp_path):
        """A missing source recording should fail clearly."""

        missing_path = tmp_path / "missing.wav"

        with pytest.raises(
            FileNotFoundError,
            match="Audio file not found",
        ):
            load_and_prepare_audio(missing_path)

    def test_stereo_audio_is_converted_to_mono(self, tmp_path):
        """Identical stereo channels should become one unchanged mono signal."""

        audio_path = tmp_path / "stereo.wav"

        original = write_test_wav(
            audio_path,
            sample_rate=16000,
            stereo=True,
        )

        processed = load_and_prepare_audio(audio_path)

        assert processed.ndim == 1
        assert len(processed) == len(original)
        assert np.allclose(
            processed,
            original,
            atol=1e-6,
        )

    def test_audio_is_upsampled_to_16_khz(self, tmp_path):
        """Audio below the target sample rate should be resampled to 16 kHz."""

        audio_path = tmp_path / "audio_8khz.wav"
        duration_s = 0.25

        write_test_wav(
            audio_path,
            sample_rate=8000,
            duration_s=duration_s,
        )

        processed = load_and_prepare_audio(audio_path)

        expected_samples = int(
            duration_s * TARGET_SAMPLE_RATE
        )

        assert len(processed) == expected_samples
        assert processed.dtype == np.float32

    def test_audio_is_downsampled_to_16_khz(self, tmp_path):
        """Audio at the main source rate should be downsampled to 16 kHz."""

        audio_path = tmp_path / "audio_44khz.wav"
        duration_s = 0.25

        write_test_wav(
            audio_path,
            sample_rate=44100,
            duration_s=duration_s,
        )

        processed = load_and_prepare_audio(audio_path)

        expected_samples = int(
            duration_s * TARGET_SAMPLE_RATE
        )

        assert len(processed) == expected_samples

    def test_content_above_8_khz_is_strongly_attenuated(self, tmp_path):
        """
        Frequencies above the new 8 kHz Nyquist limit should be filtered
        rather than aliased into the retained frequency range.
        """

        audio_path = tmp_path / "tone_10khz.wav"
        sample_rate = 44100
        duration_s = 1.0

        time = (
            np.arange(int(sample_rate * duration_s))
            / sample_rate
        )

        tone = (
            0.5
            * np.sin(2 * np.pi * 10000 * time)
        ).astype(np.float32)

        sf.write(
            audio_path,
            tone,
            sample_rate,
            subtype="FLOAT",
        )

        processed = load_and_prepare_audio(audio_path)

        original_rms = np.sqrt(
            np.mean(tone ** 2)
        )

        # Ignore a short region at each edge so that filter start/end
        # transients do not dominate the anti-aliasing check.
        margin = int(0.05 * TARGET_SAMPLE_RATE)
        processed_middle = processed[margin:-margin]

        processed_rms = np.sqrt(
            np.mean(processed_middle ** 2)
        )

        assert processed_rms < 0.05 * original_rms

    def test_audio_is_not_loudness_normalised(self, tmp_path):
        """
        Preprocessing should preserve signal level rather than forcing
        recordings to a common peak amplitude.
        """

        audio_path = tmp_path / "quiet.wav"

        write_test_wav(
            audio_path,
            amplitude=0.2,
        )

        processed = load_and_prepare_audio(audio_path)

        assert np.max(
            np.abs(processed)
        ) == pytest.approx(
            0.2,
            abs=0.001,
        )

    def test_audio_is_not_trimmed(self, tmp_path):
        """
        Leading and trailing silence should remain because silence trimming
        is deliberately not part of the shared preprocessing stage.
        """

        audio_path = tmp_path / "with_silence.wav"
        sample_rate = 16000

        silence = np.zeros(
            1000,
            dtype=np.float32,
        )

        speech = np.full(
            2000,
            0.2,
            dtype=np.float32,
        )

        audio = np.concatenate(
            [
                silence,
                speech,
                silence,
            ]
        )

        sf.write(
            audio_path,
            audio,
            sample_rate,
            subtype="FLOAT",
        )

        processed = load_and_prepare_audio(audio_path)

        assert len(processed) == len(audio)
        assert np.all(processed[:1000] == 0)
        assert np.all(processed[-1000:] == 0)


class TestPreprocessAudioFile:

    def test_processed_file_is_saved_correctly(self, tmp_path):
        """
        The complete preprocessing function should write a 16 kHz mono
        floating-point WAV and return the same processed waveform.
        """

        input_path = tmp_path / "raw" / "example.wav"
        output_path = tmp_path / "processed" / "audio" / "example.wav"

        write_test_wav(
            input_path,
            sample_rate=44100,
            stereo=True,
        )

        processed = preprocess_audio_file(
            input_path,
            output_path,
        )

        assert output_path.exists()

        info = sf.info(output_path)

        assert info.samplerate == TARGET_SAMPLE_RATE
        assert info.channels == 1
        assert info.subtype == "FLOAT"
        assert info.frames == len(processed)

# run using 
# python -m pytest tests/test_preprocessing_audio.py -v