from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

MODEL_NAME = "openai/whisper-medium"

device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Not using GPU")

# load model and processor
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.generation_config.forced_decoder_ids = None
# model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# load the audio file
audio, sr = librosa.load(
    "afrikaans1.mp3",
    sr=16000,
    mono=True
)

# convert waveform into required Whisper log-mel input features
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)
input_features = inputs.input_features.to(device)

print("Model:", next(model.parameters()).device)
print("Input:", input_features.device)

# set decoder_ids to force whisper to do english transcription
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe"
)


# execute the transcription
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids
    )

transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(transcription)


from jiwer import process_words

reference_raw = """
Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.
"""

hypothesis_raw = transcription

reference_norm = processor.tokenizer.normalize(reference_raw)
hypothesis_norm = processor.tokenizer.normalize(hypothesis_raw)

result = process_words(reference_norm, hypothesis_norm)

print("Reference:", reference_norm)
print("Hypothesis:", hypothesis_norm)

print("WER:", result.wer)
print("Substitutions:", result.substitutions)
print("Deletions:", result.deletions)
print("Insertions:", result.insertions)


# Whisper results for different models:
# Tiny: Please call Stala, ask her to bring these things with her from the store 6 spoons of freshener peas, 5 thick slabs of blue cheese and maybe a snack for her brother-bop. We also need a small plastic snake and a big to a frog for the kids. She can scoop these things into free-dread bags and we will go meet a wind-stad train station.
# Base: Please call Stella, ask her to bring these things with her from the store. 6 spoons of fresh snow peas, 5 thick slabs of blue cheese and maybe a snack for her brother bob. We also need a small plastic snake and a big twig for the kids. She can scoop these things into free-read bags and we will go meet a windstair train station.
# Small: Please call Stella. Ask her to bring these things with her from the store. Six spoons of fresh snow peas, five thick slabs of blue cheese and maybe a snack for her brother Bob. We also need a small plastic snake and a big two-way frog for the kids. She can scoop these things into free red bags and we will go meet her when she's at the train station.
# Medium: Please call Stella. Ask her to bring these things with her from the store. Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big two-way frog for the kids. She can scoop these things into free red bags and we will go meet her Wednesday at the train station.

# Tiny:
# Reference: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big toy frog for the kids she can scoop these things into 3 red bags and we will go meet her wednesday at the train station
# Hypothesis: please call stala ask her to bring these things with her from the store 6 spoons of freshener peas 5 thick slabs of blue cheese and maybe a snack for her brother bop we also need a small plastic snake and a big to a frog for the kids she can scoop these things into free dread bags and we will go meet a wind stad train station
# WER: 0.17391304347826086
# Substitutions: 9
# Deletions: 2
# Insertions: 1

# Base:
# Reference: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big toy frog for the kids she can scoop these things into 3 red bags and we will go meet her wednesday at the train station
# Hypothesis: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big twig for the kids she can scoop these things into free read bags and we will go meet a windstair train station
# WER: 0.11594202898550725
# Substitutions: 5
# Deletions: 3
# Insertions: 0

# Small:
# Reference: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big toy frog for the kids she can scoop these things into 3 red bags and we will go meet her wednesday at the train station
# Hypothesis: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big 2 way frog for the kids she can scoop these things into free red bags and we will go meet her when she is at the train station
# WER: 0.08695652173913043
# Substitutions: 3
# Deletions: 0
# Insertions: 3

# Medium:
# Reference: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big toy frog for the kids she can scoop these things into 3 red bags and we will go meet her wednesday at the train station
# Hypothesis: please call stella ask her to bring these things with her from the store 6 spoons of fresh snow peas 5 thick slabs of blue cheese and maybe a snack for her brother bob we also need a small plastic snake and a big 2 way frog for the kids she can scoop these things into free red bags and we will go meet her wednesday at the train station
# WER: 0.043478260869565216
# Substitutions: 2
# Deletions: 0
# Insertions: 1

