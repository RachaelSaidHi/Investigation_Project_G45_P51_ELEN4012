"""
Tests for the transcription functionality
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import src.asr.transcribe as transcribe_module
from src.asr.transcribe import (
    MAX_SINGLE_WINDOW_S,
    SAMPLE_RATE,
    TranscriptionResult,
    WhisperTranscriber,
    load_audio,
)

# A dummy whisper object is used so that the code's behaviour around Whisper can be tested 
# Removes the need to load a real Whisper model which would be slow, require internet and potentially a GPU

class DummyProcessor:
    """
    Small stand-in for WhisperProcessor.

    It records how it was called so that tests can verify the
    preprocessing configuration without loading a real Whisper model.
    """

    def __init__(self, decoded_text = "decoded transcript"):
        self.decoded_text = decoded_text
        self.processor_calls = []
        self.decode_calls = []

    def __call__(self, audio, **kwargs):
        self.processor_calls.append(kwargs)

        return SimpleNamespace(
            input_features=torch.ones(
                (1, 80, 10),
                dtype=torch.float32,
            ),
            attention_mask=torch.ones(
                (1, 10),
                dtype=torch.long,
            ),
        )

    def batch_decode(self, predicted_ids, **kwargs):
        self.decode_calls.append(kwargs)

        return [self.decoded_text]


class DummyModel:
    """
    Small stand-in for WhisperForConditionalGeneration.
    """

    def __init__(self):
        self.generate_calls = []
        self.eval_called = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_features, **kwargs):
        self.generate_calls.append(kwargs)

        return torch.tensor([[1, 2, 3]])



@pytest.fixture
def whisper_mocks(monkeypatch):
    """
    Replace Hugging Face model loading with lightweight dummy objects.
    """

    processor = DummyProcessor()
    model = DummyModel()

    processor_loader = Mock(
        return_value=processor
    )

    model_loader = Mock(
        return_value=model
    )

    monkeypatch.setattr(
        transcribe_module.WhisperProcessor,
        "from_pretrained",
        processor_loader,
    )

    monkeypatch.setattr(
        transcribe_module.WhisperForConditionalGeneration,
        "from_pretrained",
        model_loader,
    )

    # Keep unit tests CPU-only.
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    return {
        "processor": processor,
        "model": model,
        "processor_loader": processor_loader,
        "model_loader": model_loader,
    }


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

    def test_empty_audio_fails(self, tmp_path, monkeypatch):
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

    def test_duration_calculated_from_resampled_audio(self, tmp_path, monkeypatch):
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

        mock_load.assert_called_once_with(str(audio_file), sr=SAMPLE_RATE, mono=True,)

    def test_silent_audio_is_allowed(self, tmp_path, monkeypatch):
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

    
    def test_non_finite_audio_fails(self, tmp_path, monkeypatch):
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


class TestWhisperTranscriberInit:

    def test_invalid_decode_mode_fails(self):
        with pytest.raises(
            ValueError,
            match = "decode_mode",
        ):
            WhisperTranscriber(
                model_name = "openai/whisper-small",
                decode_mode = "something_else",
            )

    def test_cuda_requested_when_unavailable_fails(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            torch.cuda,
            "is_available",
            lambda: False,
        )

        with pytest.raises(
            RuntimeError,
            match="CUDA was requested",
        ):
            WhisperTranscriber(
                model_name = "openai/whisper-small",
                device = "cuda",
            )

    def test_cpu_uses_fp32(self, whisper_mocks):
        transcriber = WhisperTranscriber(
            model_name = "openai/whisper-small",
            device = "cpu",
            use_fp16 = True
        )

        assert transcriber.device.type == "cpu"
        assert transcriber.dtype == torch.float32

        transcriber2 = WhisperTranscriber(
            model_name = "openai/whisper-small",
            device = "cpu",
            use_fp16 = True
        )

        assert transcriber2.device.type == "cpu"
        assert transcriber2.dtype == torch.float32
        

    def test_model_and_processor_loaded_once(self, whisper_mocks):
        WhisperTranscriber(
            model_name = "openai/whisper-small",
            device = "cpu",
        )

        whisper_mocks["processor_loader"].assert_called_once_with("openai/whisper-small")
        whisper_mocks["model_loader"].assert_called_once()

        assert whisper_mocks["model"].eval_called is True

    def test_describe_contains_configuration(self, whisper_mocks):
        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="sequential_long_form",
        )

        description = transcriber.describe()

        assert description["model_name"] == ("openai/whisper-small")
        assert description["device"] == "cpu"
        assert description["hardware"] == "CPU"
        assert description["precision"] == ("torch.float32")
        assert description["decode_mode"] == ("sequential_long_form")


class TestShortForm:

    def test_short_audio_is_accepted(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                20.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name = "openai/whisper-small",
            device = "cpu",
            decode_mode = "short_form",
        )

        result = transcriber.transcribe("dummy.wav")

        assert isinstance(result, TranscriptionResult,)
        assert result.text == "decoded transcript"
        assert result.duration_s == 20.0
        assert result.decode_mode == "short_form"

    def test_exactly_30_seconds_is_accepted(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                MAX_SINGLE_WINDOW_S,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="short_form",
        )

        result = transcriber.transcribe("dummy.wav")

        assert result.duration_s == (MAX_SINGLE_WINDOW_S)

    def test_over_30_seconds_is_rejected(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                30.1,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="short_form",
        )

        with pytest.raises(
            ValueError,
            match="short-form Whisper",
        ):
            transcriber.transcribe("dummy.wav")

        # inference should never have started for an invalid input.
        assert (len(whisper_mocks["model"].generate_calls)== 0)

    def test_short_form_generation_arguments(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                10.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="short_form",
        )

        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["model"].generate_calls[0]

        assert kwargs["num_beams"] == 1
        assert kwargs["do_sample"] is False
        assert (kwargs["return_timestamps"] is False)
        assert "attention_mask" not in kwargs
        assert "condition_on_prev_tokens" not in kwargs

    def test_short_form_does_not_use_long_form_processor_options(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                10.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="short_form",
        )

        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["processor"].processor_calls[0]

        assert "truncation" not in kwargs
        assert "return_attention_mask" not in kwargs


class TestSequentialLongForm:

    def test_long_audio_is_accepted(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                70.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="sequential_long_form",
        )

        result = transcriber.transcribe("dummy.wav")

        assert result.duration_s == 70.0
        assert result.decode_mode == ("sequential_long_form")

    def test_short_audio_is_also_accepted(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                20.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="sequential_long_form",
        )

        result = transcriber.transcribe("dummy.wav")

        assert result.duration_s == 20.0

    def test_long_form_processor_arguments(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                70.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="sequential_long_form",
        )

        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["processor"].processor_calls[0]

        assert kwargs["truncation"] is False
        assert (kwargs["return_attention_mask"] is True)

    def test_long_form_generation_arguments(
        self,
        whisper_mocks,
        monkeypatch,
    ):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (
                np.zeros(10, dtype=np.float32),
                70.0,
            ),
        )

        transcriber = WhisperTranscriber(
            model_name="openai/whisper-small",
            device="cpu",
            decode_mode="sequential_long_form",
        )

        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["model"].generate_calls[0]

        assert kwargs["num_beams"] == 1
        assert kwargs["do_sample"] is False
        assert (kwargs["return_timestamps"] is True)
        assert "attention_mask" in kwargs

        assert (kwargs["condition_on_prev_tokens"] is False)
        assert kwargs["temperature"] == 0.0



# run using python -m pytest tests/test_transcription.py -v