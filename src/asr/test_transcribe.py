# TODO: Rename file to "transcribe_experiment.py" so it will be gitignored at some point, here for understanding how to the WhisperTranscriber class

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

# tiny
# Text: Please call Stala, ask her to bring these things with her from the store 6 spoons of freshener peas, 5 thick slabs of blue cheese and maybe a snack for her brother-bop. We also need a small plastic snake and a big to a frog for the kids. She can scoop these things into free-dread bags and we will go meet a wind-stad train station.
# WER:  0.1739 S: 9 D: 2 I: 1
# Audio:  20.77 s
# Feature: 0.008 s
# Infer:   0.505 s
# Decode:  0.0 s
# RTF:     0.025

# base 
# Text: Please call Stella, ask her to bring these things with her from the store. 6 spoons of fresh snow peas, 5 thick slabs of blue cheese and maybe a snack for her brother bob. We also need a small plastic snake and a big twig for the kids. She can scoop these things into free-read bags and we will go meet a windstair train station.
# WER:  0.1159 S: 5 D: 3 I: 0
# Audio:  20.77 s
# Feature: 0.008 s
# Infer:   0.661 s
# Decode:  0.0 s
# RTF:     0.032

# small
# Text: Please call Stella. Ask her to bring these things with her from the store. Six spoons of fresh snow peas, five thick slabs of blue cheese and maybe a snack for her brother Bob. We also need a small plastic snake and a big two-way frog for the kids. She can scoop these things into free red bags and we will go meet her when she's at the train station.
# WER:  0.087 S: 3 D: 0 I: 3
# Audio:  20.77 s
# Feature: 0.009 s
# Infer:   1.122 s
# Decode:  0.0 s
# RTF:     0.054


# medium
# Text: Please call Stella. Ask her to bring these things with her from the store. Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big two-way frog for the kids. She can scoop these things into free red bags and we will go meet her Wednesday at the train station.
# WER:  0.0435 S: 2 D: 0 I: 1
# Audio:  20.77 s
# Feature: 0.01 s
# Infer:   2.125 s
# Decode:  0.0 s
# RTF:     0.103
