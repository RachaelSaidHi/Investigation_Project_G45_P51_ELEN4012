"""
Resumable Whisper transcription pipeline for the expensive ASR inference stage.

Normalization and WER computation are deliberately performed separately 
so that changes to scoring do not require Whisper inference to be repeated.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.asr.transcribe import WhisperTranscriber


REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "filepath",
}

DEFAULT_DURATION_TOLERANCE_S = 0.5


RESULT_COLUMNS = [
    # Identification
    "sample_id",
    "dataset",
    "filepath",
    "manifest_index",

    # Status
    "status",
    "error_type",
    "error_message",

    # Whisper configuration
    "model_name",
    "decode_mode",
    "device",
    "hardware",
    "precision",

    # Raw transcription
    "transcription_raw",

    # Duration validation
    "manifest_duration_s",
    "processed_duration_s",
    "duration_difference_s",
    "duration_mismatch",

    # Timing
    "preprocess_s",
    "inference_s",
    "decode_s",
    "elapsed_s",
    "real_time_factor",

    # Time of run for auditting
    "completed_at_utc",
]

# TODO: can make a helper function that can be called for DSP or transcription pipeline to check correct columns are there
# Only keep any transcription specific validation needed
def validate_manifest(manifest: pd.DataFrame) -> None:
    """
    Validate the minimum requirements for transcription.
    """

    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)

    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if manifest["sample_id"].isna().any():
        raise ValueError("Manifest contains missing sample_id values.")

    if manifest["filepath"].isna().any():
        raise ValueError("Manifest contains missing filepath values.")

    sample_ids = manifest["sample_id"].astype(str)

    duplicated = sample_ids[sample_ids.duplicated()].unique()

    if len(duplicated) > 0:
        raise ValueError(f"Manifest contains duplicate sample_id values: {duplicated.tolist()}")


# TODO: check if this duplicates def load_audio(filepath: str) error checks in transcribe.py
def validate_audio_path(row, data_root: Path) -> Path:
    """
    Validate a manifest filepath relative to data/raw.
    """

    audio_path = data_root / str(row["filepath"])

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")

    return audio_path


def get_manifest_duration(row) -> float | None:
    """
    Get the manifest duration if available to compare with the audio duration after loading the audio.

    If any manifests do not contain duration_s, transcription does not require the column.
    """

    if "duration_s" not in row:
        return None

    value = row["duration_s"]

    if pd.isna(value):
        return None

    return float(value)


def build_success_record(
    row,
    manifest_index: int,
    transcription,
    config: dict,
    duration_tolerance_s: float,
) -> dict:
    """
    Build one successful transcription result row.
    """

    manifest_duration_s = get_manifest_duration(row)

    if manifest_duration_s is None:
        duration_difference_s = None
        duration_mismatch = None

    else:
        duration_difference_s = abs(transcription.duration_s - manifest_duration_s)
        duration_mismatch = (duration_difference_s > duration_tolerance_s)

    return {
        "sample_id": str(row["sample_id"]),
        "dataset": row.get("dataset"),
        "filepath": str(row["filepath"]),
        "manifest_index": manifest_index,

        "status": "ok",
        "error_type": None,
        "error_message": None,

        "model_name": config["model_name"],
        "decode_mode": config["decode_mode"],
        "device": config["device"],
        "hardware": config["hardware"],
        "precision": config["precision"],

        "transcription_raw": transcription.text,

        "manifest_duration_s": manifest_duration_s,
        "processed_duration_s": transcription.duration_s,
        "duration_difference_s": duration_difference_s,
        "duration_mismatch": duration_mismatch,

        "preprocess_s": transcription.preprocess_s,
        "inference_s": transcription.inference_s,
        "decode_s": transcription.decode_s,
        "elapsed_s": transcription.elapsed_s,
        "real_time_factor": transcription.real_time_factor,

        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_error_record(
    row,
    manifest_index: int,
    config: dict,
    error: Exception,
) -> dict:
    """
    Build a result row for a sample that could not be transcribed.
    """

    return {
        "sample_id": str(row["sample_id"]),
        "dataset": row.get("dataset"),
        "filepath": str(row["filepath"]),
        "manifest_index": manifest_index,

        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),

        "model_name": config["model_name"],
        "decode_mode": config["decode_mode"],
        "device": config["device"],
        "hardware": config["hardware"],
        "precision": config["precision"],

        "transcription_raw": None,

        "manifest_duration_s": get_manifest_duration(row),
        "processed_duration_s": None,
        "duration_difference_s": None,
        "duration_mismatch": None,

        "preprocess_s": None,
        "inference_s": None,
        "decode_s": None,
        "elapsed_s": None,
        "real_time_factor": None,

        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_transcription_manifest(
    manifest: pd.DataFrame,
    transcriber: WhisperTranscriber,
    data_root: Path,
    fail_fast: bool = False,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    ) -> pd.DataFrame:
        """
        Transcribe an entire manifest.

        Args:
            manifest:
                Manifest containing at least sample_id and filepath.
    
            transcriber:
                Already-loaded WhisperTranscriber. The model is loaded once
                outside this function and reused across all samples.
    
            data_root:
                Root directory to which manifest filepaths are relative.
    
            duration_tolerance_s:
                Allowed difference between manifest duration and the duration
                of the audio actually processed by Whisper.
    
        Returns:
            The complete current transcription results DataFrame.
        """
    
        validate_manifest(manifest)
    
        manifest = manifest.copy()
        manifest["sample_id"] = manifest["sample_id"].astype(str)
    
        config = transcriber.describe()

        total_samples = len(manifest)

        results = []
    
    
        print(f"Model: {config['model_name']}")
        print(f"Decode mode: {config['decode_mode']}")
        print(f"Device: {config['device']}")
        print(f"Hardware: {config['hardware']}")
        print(f"Precision: {config['precision']}")
        print()
        print(f"Manifest samples: {total_samples}")
        print()
    
        for manifest_index, (_, row) in enumerate(manifest.iterrows()):
            sample_id = str(row["sample_id"])
    
            print(f"[{manifest_index + 1}/{total_samples}] Transcribing {sample_id}...")
    
            try:
                audio_path = validate_audio_path(row, data_root)
    
                transcription = transcriber.transcribe(str(audio_path))
    
                record = build_success_record(
                    row=row,
                    manifest_index=manifest_index,
                    transcription=transcription,
                    config=config,
                    duration_tolerance_s=duration_tolerance_s,
                )
    
                print(
                    f"\tOK | "
                    f"{transcription.duration_s:.1f}s audio | "
                    f"{transcription.elapsed_s:.2f}s transcription | "
                    f"RTF = {transcription.real_time_factor:.3f}"
                )
    
            except Exception as error:
                record = build_error_record(
                    row=row,
                    manifest_index=manifest_index,
                    config=config,
                    error=error,
                )
    
                print(f"\tERROR | {type(error).__name__}: {error}")

            results.append(record)

            if record["status"] == "error" and fail_fast:
                raise RuntimeError(f"Transcription failed for sample {sample_id}: {record['error_message']}")

        results_df = pd.DataFrame(results)

        print()
        print("Transcription run complete.")
        print(f"Successful: {(results_df['status'] == 'ok').sum()}")
        print(f"Errors: {(results_df['status'] == 'error').sum()}")
    
        return results_df
