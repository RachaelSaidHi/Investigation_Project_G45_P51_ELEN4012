from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
import librosa
from transformers import (WhisperProcessor, WhisperForConditionalGeneration)

SAMPLE_RATE = 16000

# TODO: Decide whether to use only 30s audio clips or less or along for longer audios to be analysed (then need to do chunking)
MAX_SINGLE_WINDOW_S = 30

@dataclass(frozen=True)
class TranscriptionResult:
    """
    Stores the raw result of transcribing one audio sample before any normalization.
    """

    text: str                   # the raw transcript, exactly as Whisper generated it
    duration_s: float           # length of the audio
    feature_s: float            # mel-spectrogram extraction (CPU, roughly constant per model)
    inference_s: float          # model.generate, the part that scales with model size
    decode_s: float             # token ids to text (CPU, negligible)
    elapsed_s: float            # sum of the three (how long the full transcription took)
    real_time_factor: float     # elapsed_s / duration_s (below 1.0 is faster than real time)


# TODO: check this with Rachael's manifest because she has a duration_s row 
# Also the resampling to 16 kHz should actually be part of preprocessing so will eventually be removed from here
def load_audio(filepath: str):
    """
    Load an audio file as mono 16 kHz audio
    """

    audio, _ = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True,
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

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
    ):
        
        if device is None:
            device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(device)

        # uses fp16 with GPU and fp32 with CPU
        if use_fp16 and self.device.type == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.model_name = model_name
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
        }


    def _synchronise(self):
        """
        Wait for outstanding CUDA work before timing.
        """

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # TODO: decide if only ever using 30s or less audio else have some sort of chunking strategy 
    def transcribe(self, filepath: str) -> TranscriptionResult:
        """
        Transcribe one mono 16 kHz waveform.

        Audio longer than 30 seconds is rejected so that all samples use the same short-form Whisper decoding procedure.
        """

        if filepath is None or filepath == "":
            raise ValueError("Audio cannot be None or empty.")

        audio, duration_s = load_audio(filepath)

        if duration_s > MAX_SINGLE_WINDOW_S:
            raise ValueError(
                f"Audio is {duration_s:.1f} seconds long. "
                f"Maximum supported duration is {MAX_SINGLE_WINDOW_S:.1f} seconds."
            )

        generate_kwargs = {
            "num_beams": 1,
            "do_sample": False,
            "return_timestamps": False,
            "task": "transcribe"
        }

        # Multilingual Whisper models are explicitly constrained to English speech recognition.
        if not self.is_english_only:
            generate_kwargs["language"] = "english"


        # Time the complete transcription computation, but not file loading or model loading.
        # RTF = (feature extraction + inference + decoding time)/audio duration not just generation time
        self._synchronise()
        t0 = perf_counter()

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.dtype,
        )

        self._synchronise()
        t1 = perf_counter()

        with torch.inference_mode():
            predicted_ids = self.model.generate(
                input_features,
                **generate_kwargs,
            )

        self._synchronise()
        t2 = perf_counter()

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        self._synchronise()
        t3 = perf_counter()

        feature_s = t1 - t0
        inference_s = t2 - t1
        decode_s = t3 - t2
        elapsed_s = t3 - t0

        return TranscriptionResult(
            text=text,
            duration_s=float(duration_s),
            feature_s=float(feature_s),
            inference_s=float(inference_s),
            decode_s=float(decode_s),
            elapsed_s=float(elapsed_s),
            real_time_factor=float(elapsed_s / duration_s),
        )


