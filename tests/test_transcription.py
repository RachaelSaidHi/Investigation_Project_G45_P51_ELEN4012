"""
Tests for the transcription functionality
"""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock
import os

import numpy as np
import pytest
import torch
import soundfile as sf

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
    It records how it was called so that tests can verify the preprocessing configuration without loading a real Whisper model.
    """

    def __init__(self, decoded_text = "decoded transcript"):
        self.decoded_text = decoded_text
        self.processor_calls = []
        self.decode_calls = []

    def __call__(self, audio, **kwargs):
        self.processor_calls.append(kwargs)

        return SimpleNamespace(
            input_features=torch.ones((1, 80, 10), dtype=torch.float32),
            attention_mask=torch.ones((1, 10), dtype=torch.long),
        )

    def batch_decode(self, predicted_ids, **kwargs):
        self.decode_calls.append(kwargs)
        return [self.decoded_text]


class DummyModel:
    """
    Small stand-in for WhisperForConditionalGeneration.
    Records calls to generate() so that the Whisper generation configuration can be tested directly.
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
    Replace Hugging Face model loading with lightweight dummy objects to avoid downloading models, real inference and GPU requirement
    """

    processor = DummyProcessor()
    model = DummyModel()

    processor_loader = Mock(return_value=processor)

    model_loader = Mock(return_value=model)

    monkeypatch.setattr(transcribe_module.WhisperProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(transcribe_module.WhisperForConditionalGeneration, "from_pretrained", model_loader)

    # Keep unit tests CPU-only.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    return {
        "processor": processor,
        "model": model,
        "processor_loader": processor_loader,
        "model_loader": model_loader,
    }


@pytest.fixture
def make_transcriber(whisper_mocks, monkeypatch):
    """Create a mocked WhisperTranscriber with a chosen duration and decoding mode."""

    def _make(duration_s=10.0, decode_mode="short_form", model_name="openai/whisper-small"):
        monkeypatch.setattr(
            transcribe_module,
            "load_audio",
            lambda _: (np.zeros(10, dtype=np.float32), duration_s),
        )

        return WhisperTranscriber(
            model_name=model_name,
            device="cpu",
            decode_mode=decode_mode,
        )

    return _make


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

        with pytest.raises(ValueError, match="Audio file is empty"):
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

    def test_load_audio_resamples_and_downmixes(self, tmp_path):
            """
            A stereo 44.1 kHz audio must become mono 16 kHz audio with the same duration.
            """
            original_sr = 44100
            duration_s = 2.0
            path = tmp_path / "stereo.wav"

            # Fake test audio is 44.1 kHz for 2s on 2 channels
            stereo = np.zeros((int(original_sr * duration_s), 2), dtype=np.float32)

            # Makes the left and right channel different (non-identical)
            stereo[:, 0] = 0.25
            stereo[:, 1] = -0.10

            # Writes it to a WAV file
            sf.write(path, stereo, original_sr)

            # Should transform 44.1 kHz to 16 kHz and stereo to mono
            audio, loaded_duration_s = load_audio(str(path))
    
            assert audio.ndim == 1
            assert len(audio) == int(duration_s * SAMPLE_RATE) 
            assert loaded_duration_s == pytest.approx(duration_s)

    def test_silent_audio_is_allowed(self, tmp_path, monkeypatch):
        path = tmp_path / "silent.wav"
        sf.write(path, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

        audio, duration_s = load_audio(str(path))
        
        assert len(audio) == SAMPLE_RATE
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

        with pytest.raises(ValueError, match="NaN or infinite"):
            load_audio(str(audio_file))


class TestWhisperTranscriberInit:

    def test_invalid_decode_mode_fails(self):
        with pytest.raises(ValueError, match="decode_mode"):
            WhisperTranscriber(
                model_name="openai/whisper-small",
                decode_mode="invalid_mode",
            )

    def test_cuda_requested_when_unavailable_fails(self, whisper_mocks, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            WhisperTranscriber(model_name="openai/whisper-small", device="cuda")

    def test_cpu_uses_fp32_always(self, whisper_mocks):
        transcriber = WhisperTranscriber(
            model_name = "openai/whisper-small",
            device = "cpu",
            use_fp16 = False
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
        transcriber = WhisperTranscriber(
            model_name = "openai/whisper-small",
            device = "cpu",
        )

        whisper_mocks["processor_loader"].assert_called_once_with("openai/whisper-small")
        whisper_mocks["model_loader"].assert_called_once_with("openai/whisper-small", torch_dtype=torch.float32)

        assert transcriber.processor is whisper_mocks["processor"]
        assert transcriber.model is whisper_mocks["model"]
        assert whisper_mocks["model"].device == torch.device("cpu")
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

    @pytest.mark.parametrize("duration_s", [1.0, 20.0, MAX_SINGLE_WINDOW_S])
    def test_short_form_accepts_up_to_30s(self, make_transcriber, duration_s):
        """Short-form must accept audio up to and including 30 seconds."""
        transcriber = make_transcriber(duration_s=duration_s, decode_mode="short_form")
        result = transcriber.transcribe("dummy.wav")

        assert isinstance(result, TranscriptionResult,)
        assert result.text == "decoded transcript"
        assert result.duration_s == duration_s
        assert result.decode_mode == "short_form"

    def test_exactly_30_seconds_is_accepted(self, make_transcriber, monkeypatch):
        transcriber = make_transcriber(duration_s=30, decode_mode="short_form")
        result = transcriber.transcribe("dummy.wav")

        assert result.duration_s == (MAX_SINGLE_WINDOW_S)

    def test_over_30_seconds_is_rejected(self, whisper_mocks, make_transcriber):
        transcriber = make_transcriber(duration_s=30.1, decode_mode="short_form")

        with pytest.raises(ValueError, match="short-form Whisper"):
            transcriber.transcribe("dummy.wav")

        # inference should never have started for an invalid input.
        assert (len(whisper_mocks["model"].generate_calls)== 0)
        assert whisper_mocks["model"].generate_calls == []

    def test_short_form_generation_arguments(self, whisper_mocks, make_transcriber):
        transcriber = make_transcriber(decode_mode="short_form")
        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["model"].generate_calls[0]

        assert kwargs["num_beams"] == 1
        assert kwargs["do_sample"] is False
        assert kwargs["return_timestamps"] is False
        assert "attention_mask" not in kwargs
        assert "condition_on_prev_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_short_form_does_not_use_long_form_processor_options(self, whisper_mocks, make_transcriber):
        transcriber = make_transcriber(decode_mode="short_form")
        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["processor"].processor_calls[0]

        assert "truncation" not in kwargs
        assert "return_attention_mask" not in kwargs


class TestSequentialLongForm:

    @pytest.mark.parametrize("duration_s", [20.0, 70.0])
    def test_long_form_accepts_short_and_long_audio(self, make_transcriber, duration_s):
        """
        Sequential long-form must work for both short and long recordings.
        """
        transcriber = make_transcriber(duration_s=duration_s, decode_mode="sequential_long_form")
        result = transcriber.transcribe("dummy.wav")

        assert result.duration_s == duration_s
        assert result.decode_mode == "sequential_long_form"

    def test_long_form_processor_arguments(self, make_transcriber, whisper_mocks):
        transcriber = make_transcriber(duration_s=70.0, decode_mode="sequential_long_form")
        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["processor"].processor_calls[0]

        assert kwargs["truncation"] is False
        assert (kwargs["return_attention_mask"] is True)

    def test_long_form_generation_arguments(self, make_transcriber, whisper_mocks):
        transcriber = make_transcriber(duration_s=70.0, decode_mode="sequential_long_form")
        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["model"].generate_calls[0]

        assert kwargs["num_beams"] == 1
        assert kwargs["do_sample"] is False
        assert (kwargs["return_timestamps"] is True)
        assert "attention_mask" in kwargs

        assert (kwargs["condition_on_prev_tokens"] is False)
        assert kwargs["temperature"] == 0.0


def test_common_processor_arguments(make_transcriber, whisper_mocks):
        transcriber = make_transcriber()
        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["processor"].processor_calls[0]

        assert kwargs["sampling_rate"] == SAMPLE_RATE
        assert kwargs["return_tensors"] == "pt"


class TestLanguageConfiguration:

    @pytest.mark.parametrize(
        "model_name, is_english_only",
        [
            ("openai/whisper-small", False),
            ("openai/whisper-small.en", True),
        ],
    )
    def test_language_prompt_configuration(
        self,
        make_transcriber,
        whisper_mocks,
        model_name,
        is_english_only,
    ):
        """
        Multilingual models must be forced to English; .en models do not need those prompts.
        """
        transcriber = make_transcriber(model_name=model_name, decode_mode="short_form")
        transcriber.transcribe("dummy.wav")

        kwargs = whisper_mocks["model"].generate_calls[0]

        if is_english_only:
            assert "language" not in kwargs
            assert "task" not in kwargs
        else:
            assert kwargs["language"] == "english"
            assert kwargs["task"] == "transcribe"


class TestTranscriptionResult:

    def test_timing_relationships(self, make_transcriber, monkeypatch):
        transcriber = make_transcriber(duration_s=20.0)
        result = transcriber.transcribe("dummy.wav")

        assert result.preprocess_s >= 0
        assert result.inference_s >= 0
        assert result.decode_s >= 0
        assert result.elapsed_s >= 0

        assert result.elapsed_s == pytest.approx(result.preprocess_s + result.inference_s + result.decode_s)
        assert (result.real_time_factor == pytest.approx(result.elapsed_s/20))

    def test_timing_breakdown(self, make_transcriber, monkeypatch):
        """
        The timer boundaries must produce the correct preprocessing, inference and decode times.
        """
        ticks = iter([0.0, 1.0, 3.0, 3.5])
        monkeypatch.setattr(transcribe_module, "perf_counter", lambda: next(ticks))

        transcriber = make_transcriber(duration_s=7.0)
        result = transcriber.transcribe("dummy.wav")

        assert result.preprocess_s == pytest.approx(1.0)
        assert result.inference_s == pytest.approx(2.0)
        assert result.decode_s == pytest.approx(0.5)
        assert result.elapsed_s == pytest.approx(3.5)
        assert result.real_time_factor == pytest.approx(0.5)

    def test_result_is_frozen(self, make_transcriber):
        transcriber = make_transcriber()
        result = transcriber.transcribe("dummy.wav")

        assert isinstance(result, TranscriptionResult)
        with pytest.raises(FrozenInstanceError):
            result.text = "changed"

    def test_empty_hypothesis_is_allowed(self, make_transcriber, whisper_mocks):
        whisper_mocks["processor"].decoded_text = "   "

        transcriber = make_transcriber()
        result = transcriber.transcribe("dummy.wav")

        assert result.text == ""

    def test_timestamp_tokens_are_not_decoded(self, make_transcriber, whisper_mocks):
        transcriber = make_transcriber(duration_s=70.0, decode_mode="sequential_long_form")
        transcriber.transcribe("dummy.wav")

        decode_kwargs = whisper_mocks["processor"].decode_calls[0]

        assert (decode_kwargs["decode_with_timestamps"] is False)
        assert (decode_kwargs["skip_special_tokens"] is True)
        assert (decode_kwargs["clean_up_tokenization_spaces"] is False)


# ============================================================
# Real Whisper integration tests
# ============================================================

RUN_INTEGRATION_TESTS = os.getenv("RUN_WHISPER_INTEGRATION") == "1"

@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_INTEGRATION_TESTS,
    reason="Set RUN_WHISPER_INTEGRATION=1 to run real Whisper tests.",
)
@pytest.mark.parametrize(
    "decode_mode,duration_s",
    [
        ("short_form", 5.0),
        ("sequential_long_form", 31.0),
    ],
)
def test_real_whisper_smoke(tmp_path, decode_mode, duration_s):
    """
    Verify that the real installed Hugging Face Whisper implementation
    accepts the preprocessing and generation arguments used by the wrapper.
    """

    path = tmp_path / "silence.wav"
    audio = np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)
    sf.write(path, audio, SAMPLE_RATE)

    transcriber = WhisperTranscriber(
        model_name="openai/whisper-tiny",
        device="cpu",
        decode_mode=decode_mode,
    )

    result = transcriber.transcribe(str(path))

    assert isinstance(result, TranscriptionResult)
    assert isinstance(result.text, str)
    assert result.duration_s == pytest.approx(duration_s, abs=1e-3)
    assert result.elapsed_s >= 0
    assert result.real_time_factor >= 0


# run using python -m pytest tests/test_transcription.py -v

# with integration tests:
# export RUN_WHISPER_INTEGRATION=1
# pytest tests/test_transcription.py -m integration
# unset RUN_WHISPER_INTEGRATION