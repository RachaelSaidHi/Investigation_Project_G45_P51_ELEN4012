from dataclasses import dataclass
from time import perf_counter
import torch
import librosa
from transformers import (WhisperProcessor, WhisperForConditionalGeneration)
from typing import Literal
from pathlib import Path
import numpy as np

# TODO: Make a whisper.yaml that can be used to adjust experimental configs

SAMPLE_RATE = 16000

# There are two transcription options considered, short-form and sequential long-form
DecodeMode = Literal["short_form", "sequential_long_form"]

# With short-form decoding, the audio needs to be 30s or less
MAX_SINGLE_WINDOW_S = 30

@dataclass(frozen=True)
class TranscriptionResult:
    """
    Stores the raw result of transcribing one audio sample before any normalization.
    """

    text: str                   # decoded Whisper transcript before project normalization
    duration_s: float           # length of the audio
    preprocess_s: float         # mel-spectrogram extraction (CPU) and transfer to model device
    inference_s: float          # model.generate, the part that scales with model size
    decode_s: float             # token ids to text (CPU)
    elapsed_s: float            # sum of the three (how long the full transcription took)
    real_time_factor: float     # elapsed_s / duration_s (below 1.0 is faster than real time)
    decode_mode: DecodeMode     # short-form or sequential long-form transcription


# TODO: resampling belongs in preprocessing, so this will eventually take already-prepared audio.
def load_audio(filepath: str):
    """
    Load an audio file as mono 16 kHz audio and return it with its duration
    """

    if not filepath:
        raise ValueError("Audio filepath cannot be None or empty.")

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")

    audio, _ = librosa.load(
        filepath,
        sr = SAMPLE_RATE,
        mono = True,
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite samples.")


    duration_s = len(audio) / SAMPLE_RATE

    return audio, duration_s


class WhisperTranscriber:
    """
    Loads one Whisper model and reuses it for multiple recordings.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        use_fp16: bool = True,
        decode_mode: DecodeMode = "sequential_long_form"
    ):
        """
        Args:
            model_name: Hugging Face checkpoint, e.g. "openai/whisper-small".
            device:     "cuda" or "cpu". Detected automatically if not given.
            use_fp16:   half precision on GPU. Can affect the transcript as well as speed, so it must match across the model sweep.
        """

        if decode_mode not in {"short_form", "sequential_long_form"}:
            raise ValueError("decode_mode must be 'short_form' or 'sequential_long_form'.")

        self.decode_mode = decode_mode
        
        if device is None:
            device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(device)

        if (self.device.type == "cuda"and not torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested but is not available.")

        # uses fp16 with GPU and fp32 with CPU (half precision is only worth using on GPU; on CPU it is slower)
        if use_fp16 and self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.model_name = model_name

        # The .en checkpoints are English-only and take no language or task tokens, so the decoder prompt has to be built differently for them.
        self.is_english_only = model_name.endswith(".en")

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype = self.dtype).to(self.device)
        self.model.eval()


    def describe(self) -> dict:
        """
        Returns the inference configuration for reproducibility.
        """

        if self.device.type == "cuda":
            hardware = torch.cuda.get_device_name(self.device)
        else:
            hardware = "CPU"

        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "hardware": hardware,
            "precision": str(self.dtype),
            "decode_mode": self.decode_mode
        }


    def _synchronise(self):
        """
        Wait for outstanding/queued CUDA work to finish before timing.
        """

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


    def transcribe(self, filepath: str) -> TranscriptionResult:
        """
        Transcribe one audio file.

        Audio longer than 30 seconds when short-form is chosen is rejected with an error. 
        The class object has a decode_path set on initialisation.
        This ensures that every sample in an analysis goes through the same decoding path.
        """

        audio, duration_s = load_audio(filepath)

        if self.decode_mode == "short_form":
            if duration_s > MAX_SINGLE_WINDOW_S:
                raise ValueError(
                f"Audio is {duration_s:.1f} seconds long, but short-form Whisper supports at most {MAX_SINGLE_WINDOW_S:.1f} s.\n"
                f"Use decode_mode = 'sequential_long_form'."
                )
            processor_kwargs = {}

            return_timestamps = False

        elif self.decode_mode == "sequential_long_form":
            processor_kwargs = {
                "truncation": False,
                "return_attention_mask": True
            }

            return_timestamps = True


        # Time the complete transcription computation, but not file loading or model loading.
        # RTF = (feature extraction + inference + decoding time)/audio duration not just generation time
        self._synchronise()
        t0 = perf_counter()

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            **processor_kwargs
        )

        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.dtype,
        )

        # Greedy decoding with no stochastic sampling. The same decoding config is used for every model in the sweep. so the output is deterministic for a given model, precision and device.
        generate_kwargs = {
            "num_beams": 1,
            "do_sample": False,
            "return_timestamps": return_timestamps,
        }

        attention_mask = None
        
        if self.decode_mode == "sequential_long_form":
            attention_mask = inputs.attention_mask.to(self.device)
            generate_kwargs["attention_mask"] = attention_mask

            # TODO: Might need to adjust these
            generate_kwargs["temperature"] = 0.0
            generate_kwargs["condition_on_prev_tokens"] = False


        # Multilingual Whisper models are explicitly constrained to English speech recognition.
        if not self.is_english_only:
            generate_kwargs["language"] = "english"
            generate_kwargs["task"] = "transcribe"


        self._synchronise()
        t1 = perf_counter()

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs,
            )

        self._synchronise()
        t2 = perf_counter()

        # clean_up_tokenization_spaces is off so that spacing is not altered before normalisation sees the text.
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            decode_with_timestamps=False,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        self._synchronise()
        t3 = perf_counter()

        preprocess_s = t1 - t0
        inference_s = t2 - t1
        decode_s = t3 - t2
        elapsed_s = t3 - t0

        return TranscriptionResult(
            text = text,
            duration_s = float(duration_s),
            preprocess_s = float(preprocess_s),
            inference_s = float(inference_s),
            decode_s = float(decode_s),
            elapsed_s = float(elapsed_s),
            real_time_factor = float(elapsed_s / duration_s),
            decode_mode = self.decode_mode
        )


