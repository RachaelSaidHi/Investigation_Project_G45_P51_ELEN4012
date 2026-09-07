from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

MODEL_NAME = "openai/whisper-tiny"

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# from transformers import pipeline
# import torch

# transcriber = pipeline(
#     "automatic-speech-recognition",
#     model="openai/whisper-base",
#     device=0
# )

# result = transcriber("afrikaans1.mp3")
# print(result["text"])



