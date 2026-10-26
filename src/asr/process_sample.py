
from pathlib import Path
import pandas as pd

from src.asr.transcribe import WhisperTranscriber
from src.asr.normalize import normalize_text
from src.asr.metrics import compute_wer

DURATION_TOLERANCE_S = 0.5

DATA_ROOT = Path("data/raw")


def resolve_audio_path(row) -> Path:
    """
    Build the absolute audio path from the manifest's relative filepath.
    """
 
    audio_path = DATA_ROOT / row["filepath"]
 
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
 
    return audio_path


def process_sample(transcriber: WhisperTranscriber, row) -> dict:
    """
    Transcribe, normalise and score one manifest row.
 
    Args:
        transcriber: an already-loaded WhisperTranscriber. Loading is slow,
                     so one instance is reused across the whole run.
        row:         one row of the manifest, as a pandas Series.
 
    Returns:
        A flat dictionary holding the raw transcript, the normalized text,
        the WER results, the timing breakdown and the run configuration.
 
    Raises:
        FileNotFoundError: if the audio file is missing.
        ValueError:        if the audio is unusable or exceeds the window.
    """
 
    sample_id = row["sample_id"]
    dataset = row["dataset"]
 
    audio_path = resolve_audio_path(row)
 
    # The raw transcript is the only artefact here that cannot be cheaply regenerated, so it is carried through into the record unchanged.

    transcription = transcriber.transcribe(str(audio_path))
 
    duration_mismatch = abs(transcription.duration_s - float(row["duration_s"])) > DURATION_TOLERANCE_S
 

    # Both sides go through the same normalization call, which is what guarantees they are compared under identical rules.
 
    reference_norm, reference_punctuation = normalize_text(
        row["reference_transcript"],
        dataset,
    )
 
    hypothesis_norm, hypothesis_punctuation = normalize_text(
        transcription.text,
        dataset,
    )
 
    # An empty reference has no defined WER, so it is flagged rather than scored. 
    # This could happen if an AfriSpeech reference consists only of punctuation.
 
    reference_empty = reference_norm.strip() == ""
 
    if reference_empty:
        scores = None
        status = "empty_reference"
    else:
        scores = compute_wer(reference_norm, hypothesis_norm)
        status = "ok"
 
    return build_record(
        sample_id=sample_id,
        dataset=dataset,
        status=status,
        transcription=transcription,
        reference_norm=reference_norm,
        hypothesis_norm=hypothesis_norm,
        reference_punctuation=reference_punctuation,
        hypothesis_punctuation=hypothesis_punctuation,
        scores=scores,
        config=transcriber.describe(),
        duration_mismatch=duration_mismatch,
    )


def process_by_id(transcriber, manifest, sample_id, data_root):
    """
    Convenience wrapper for testing and spot checks.
    """
    return process_sample(transcriber, manifest.loc[sample_id], data_root)
 
 
def build_record(
    sample_id,
    dataset,
    status,
    transcription,
    reference_norm,
    hypothesis_norm,
    reference_punctuation,
    hypothesis_punctuation,
    scores,
    config,
    duration_mismatch,
) -> dict:
    """
    Assemble one flat results row.
 
    Only sample_id and dataset are carried over from the manifest. Accent,
    country and speaker are joined back from the manifest at analysis time
    rather than duplicated here, so there is one source of truth for them.
    """
 
    record = {
        # Join key. The DSP feature table uses the same one.
        "sample_id": sample_id,
        "dataset": dataset,
        "status": status,
 
        # Run configuration. Stored per row so that results from several models can be stacked in one table and still be told apart.
        "model_name": config["model_name"],
        "device": config["device"],
        "hardware": config["hardware"],
        "precision": config["precision"],
        "decoding": config.get("decoding", "short_form"),
 
        # The irreplaceable artefact. Everything below it can be recomputed from this without re-running inference.
        "hypothesis_raw": transcription.text,
 
        "reference_norm": reference_norm,
        "hypothesis_norm": hypothesis_norm,
 
        # Punctuation diagnostic. Comparing these per accent group shows
        # whether Whisper fails to transcribe spoken punctuation for
        # particular accents, which would otherwise inflate WER in a way
        # that correlates with acoustic distance.
        # TODO: I don't think this is good to have since the reference has no punctuation?
        "n_punctuation_reference": reference_punctuation,
        "n_punctuation_hypothesis": hypothesis_punctuation,
 
        # Timing results
        "duration_s": transcription.duration_s,
        "preprocess_s": transcription.preprocess_s,
        "inference_s": transcription.inference_s,
        "decode_s": transcription.decode_s,
        "elapsed_s": transcription.elapsed_s,
        "real_time_factor": transcription.real_time_factor,
 
        "duration_mismatch": duration_mismatch,
    }
 
    # S, D, I and N are stored rather than WER alone, because corpus WER cannot be reconstructed from per-sample WER values.
    if scores is None:
        record["wer"] = None
        record["substitutions"] = None
        record["deletions"] = None
        record["insertions"] = None
        record["hits"] = None
        record["n_reference_words"] = None
    else:
        record["wer"] = scores.wer
        record["substitutions"] = scores.substitutions
        record["deletions"] = scores.deletions
        record["insertions"] = scores.insertions
        record["hits"] = scores.hits
        record["n_reference_words"] = scores.n_reference_words
 
    return record


manifest = pd.read_csv("data/manifests/dev_manifest.csv")
transcriber = WhisperTranscriber("openai/whisper-tiny")

for _, row in manifest.iterrows():
    record = process_sample(transcriber, row)
    print(row["accent_raw"])
    print(record["wer"])
    print()