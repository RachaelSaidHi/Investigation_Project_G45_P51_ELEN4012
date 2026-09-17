from dataclasses import dataclass
from time import perf_counter
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
    preprocess_s: float         # mel-spectrogram extraction (CPU) and transfer to model device
    inference_s: float          # model.generate, the part that scales with model size
    decode_s: float             # token ids to text (CPU)
    elapsed_s: float            # sum of the three (how long the full transcription took)
    real_time_factor: float     # elapsed_s / duration_s (below 1.0 is faster than real time)


# TODO: check this with Rachael's manifest because she has a duration_s row (maybe compare the two to make sure they match)
# resampling belongs in preprocessing, so this will eventually take already-prepared audio.
def load_audio(filepath: str):
    """
    Load an audio file as mono 16 kHz audio and return it with its duration
    """

    audio, _ = librosa.load(
        filepath,
        sr = SAMPLE_RATE,
        mono = True,
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
        """
        Args:
            model_name: Hugging Face checkpoint, e.g. "openai/whisper-small".
            device:     "cuda" or "cpu". Detected automatically if not given.
            use_fp16:   half precision on GPU. Can affect the transcript as well as speed, so it must match across the model sweep.
        """
        
        if device is None:
            device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device(device)

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
        }


    def _synchronise(self):
        """
        Wait for outstanding/queued CUDA work to finish before timing.
        """

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # TODO: decide if only ever using 30s or less audio else have some sort of chunking strategy 
    def transcribe(self, filepath: str) -> TranscriptionResult:
        """
        Transcribe one audio file.

        Audio longer than 30 seconds is rejected so that every sample in an analysis goes through the same decoding path.
        """

        if not filepath:
            raise ValueError("Audio filepath cannot be None or empty.")

        audio, duration_s = load_audio(filepath)

        if duration_s > MAX_SINGLE_WINDOW_S:
            return TranscriptionResult(
                        text="",
                        duration_s=0,
                        preprocess_s=0,
                        inference_s=0,
                        decode_s=0,
                        elapsed_s=0,
                        real_time_factor=0,
                    )
            # raise ValueError(
            #     f"Audio is {duration_s:.1f} seconds long. "
            #     f"Maximum supported duration is {MAX_SINGLE_WINDOW_S:.1f} seconds."
            # )

        # Greedy decoding with no stochastic sampling. The same decoding policy is used for every model in the sweep. so the output is deterministic for a given model, precision and device.
        generate_kwargs = {
            "num_beams": 1,
            "do_sample": False,
            "return_timestamps": False,
        }

        # Multilingual Whisper models are explicitly constrained to English speech recognition.
        if not self.is_english_only:
            generate_kwargs["language"] = "english"
            generate_kwargs["task"] = "transcribe"


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

        # clean_up_tokenization_spaces is off so that spacing is not altered before normalisation sees the text.
        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        self._synchronise()
        t3 = perf_counter()

        preprocess_s = t1 - t0
        inference_s = t2 - t1
        decode_s = t3 - t2
        elapsed_s = t3 - t0

        return TranscriptionResult(
            text=text,
            duration_s=float(duration_s),
            preprocess_s=float(preprocess_s),
            inference_s=float(inference_s),
            decode_s=float(decode_s),
            elapsed_s=float(elapsed_s),
            real_time_factor=float(elapsed_s / duration_s),
        )


