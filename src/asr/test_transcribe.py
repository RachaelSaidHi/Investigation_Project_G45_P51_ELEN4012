from src.asr.transcribe import WhisperTranscriber
from src.asr.normalize import normalize_text
from src.asr.metrics import compute_wer

MODELS = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
]

REFERENCE_RAW = (
    "Please call Stella. Ask her to bring these things with her from the "
    "store: Six spoons of fresh snow peas, five thick slabs of blue cheese, "
    "and maybe a snack for her brother Bob. We also need a small plastic "
    "snake and a big toy frog for the kids. She can scoop these things into "
    "three red bags, and we will go meet her Wednesday at the train station."
)

AUDIO = "afrikaans1.mp3"

for model_name in MODELS:
    transcriber = WhisperTranscriber(model_name)

    # First run includes CUDA warm-up, so discard it.
    transcriber.transcribe(AUDIO)

    result = transcriber.transcribe(AUDIO)

    reference, _ = normalize_text(REFERENCE_RAW, "speech_accent_archive")
    hypothesis, _ = normalize_text(result.text, "speech_accent_archive")
    scores = compute_wer(reference, hypothesis)

    print(model_name)
    print("Text:", result.text)
    print("WER: ", round(scores.wer, 4),
          "S:", scores.substitutions,
          "D:", scores.deletions,
          "I:", scores.insertions)
    print("Audio: ", round(result.duration_s, 2), "s")
    print("Feature:", round(result.feature_s, 3), "s")
    print("Infer:  ", round(result.inference_s, 3), "s")
    print("Decode: ", round(result.decode_s, 3), "s")
    print("RTF:    ", round(result.real_time_factor, 3))
    print()
    print()

    del transcriber



# print("Text:  ", result.text)
# print("Audio: ", round(result.duration_s, 2), "s")
# print("Feature:", round(result.feature_s, 3), "s")
# print("Infer:  ", round(result.inference_s, 3), "s")
# print("Decode: ", round(result.decode_s, 3), "s")
# print("RTF:    ", round(result.real_time_factor, 3))
# print("WER:    ", compute_wer(reference, hypothesis))