"""
Normalization and WER scoring pipeline.

This stage uses saved Whisper transcriptions and the corresponding manifest. 
It does not run Whisper inference.

The scoring stage is intentionally separate from transcription so that
normalization rules and metrics can be changed without repeating expensive
ASR inference.
"""

from datetime import datetime, timezone

import pandas as pd

from src.asr.normalize import normalize_text
from src.asr.metrics import compute_wer


REQUIRED_MANIFEST_COLUMNS = {
    "sample_id",
    "dataset",
    "reference_transcript",
}

REQUIRED_TRANSCRIPTION_COLUMNS = {
    "sample_id",
    "status",
    "transcription_raw",
    "model_name",
    "decode_mode",
}


RESULT_COLUMNS = [
    # Identification
    "sample_id",
    "dataset",

    # ASR experiment
    "model_name",
    "decode_mode",

    # Status
    "transcription_status",
    "scoring_status",
    "scoring_error_type",
    "scoring_error_message",

    # Normalized text
    "reference_norm",
    "hypothesis_norm",

    # TODO: I don't think this is good to have since the reference has no written out punctuation
    # Dataset-specific punctuation diagnostics
    "n_punctuation_reference",
    "n_punctuation_hypothesis",

    # WER metrics
    "wer",
    "substitutions",
    "deletions",
    "insertions",
    "hits",
    "n_reference_words",

    # Time of run for auditting
    "scored_at_utc",
]


# TODO: remove duplicate code between this and transcription_pipeline, make a helper
def validate_manifest(manifest: pd.DataFrame) -> None:
    """
    Validate the minimum manifest fields required for scoring.
    """

    missing = (REQUIRED_MANIFEST_COLUMNS- set(manifest.columns))

    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if manifest["sample_id"].isna().any():
        raise ValueError("Manifest contains missing sample_id values.")

    sample_ids = manifest["sample_id"].astype(str)

    if sample_ids.str.strip().eq("").any():
        raise ValueError("Manifest contains empty sample_id values.")

    duplicated = (sample_ids[sample_ids.duplicated()].unique())

    if len(duplicated) > 0:
        raise ValueError("Manifest contains duplicate sample_id values: "f"{duplicated.tolist()}")


def validate_transcriptions(transcriptions: pd.DataFrame) -> None:
    """
    Validate the minimum transcription fields required for scoring.
    """

    missing = (REQUIRED_TRANSCRIPTION_COLUMNS- set(transcriptions.columns))

    if missing:
        raise ValueError(f"Transcription results are missing required columns: {sorted(missing)}")

    if transcriptions["sample_id"].isna().any():
        raise ValueError("Transcription results contain missing sample_id values.")

    sample_ids = (transcriptions["sample_id"].astype(str))

    if sample_ids.str.strip().eq("").any():
        raise ValueError("Manifest contains empty sample_id values.")

    duplicated = (sample_ids[sample_ids.duplicated()].unique())

    if len(duplicated) > 0:
        raise ValueError(f"Transcription results contain duplicate sample_id values: {duplicated.tolist()}")


def validate_sample_ids(manifest: pd.DataFrame,transcriptions: pd.DataFrame) -> None:
    """
    Ensure transcription results do not refer to samples absent from the manifest.

    Missing transcription rows are allowed.
    A transcription run may be incomplete in future resumable experiments.
    """

    manifest_ids = set(manifest["sample_id"].astype(str))

    transcription_ids = set(transcriptions["sample_id"].astype(str))

    unknown_ids = (transcription_ids - manifest_ids)

    if unknown_ids:
        raise ValueError(
            "Transcription results contain sample IDs that are "
            f"not present in the manifest: {sorted(unknown_ids)}"
        )


def empty_metrics() -> dict:
    """
    Return empty metric fields for a sample that cannot be scored.
    (empty reference transcript, pending transcriptions, errored transcriptions)
    """

    return {
        "wer": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "hits": None,
        "n_reference_words": None,
    }


def build_pending_record(row) -> dict:
    """
    Build a scoring row for a sample that has not yet been transcribed.
    """

    return {
        "sample_id": str(row["sample_id"]),
        "dataset": row["dataset"],

        "model_name": None,
        "decode_mode": None,

        "transcription_status": "pending",
        "scoring_status": "pending_transcription",
        "scoring_error_type": None,
        "scoring_error_message": None,

        "reference_norm": None,
        "hypothesis_norm": None,

        "n_punctuation_reference": None,
        "n_punctuation_hypothesis": None,

        **empty_metrics(),

        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_error_record(row, error: Exception = None) -> dict:
    """
    Build a scoring row when Whisper inference previously failed or scoring failed.
    """

    record = {
        "sample_id": str(row["sample_id"]),
        "dataset": row["dataset"],

        "model_name": row["model_name"],
        "decode_mode": row["decode_mode"],

        "transcription_status": row["status"],
        "scoring_status": None,
        "scoring_error_type": None,
        "scoring_error_message": None,

        "reference_norm": None,
        "hypothesis_norm": None,

        "n_punctuation_reference": None,
        "n_punctuation_hypothesis": None,

        **empty_metrics(),

        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if error is None:
        record["scoring_status"] = "transcription_error"
        return record

    record["scoring_status"] = "scoring_error"
    record["scoring_error_type"] = type(error).__name__
    record["scoring_error_message"] = str(error)
    
    return record
    

def score_sample(row) -> dict:
    """
    Normalize and score one successfully transcribed sample.

    The input row is expected to contain fields from both the manifest and the transcription results.
    """

    reference_norm, reference_punctuation = (
        normalize_text(row["reference_transcript"], row["dataset"])
        )

    hypothesis_norm, hypothesis_punctuation = (
        normalize_text(row["transcription_raw"], row["dataset"],)
    )

    base_record = {
        "sample_id": str(row["sample_id"]),
        "dataset": row["dataset"],

        "model_name": row["model_name"],
        "decode_mode": row["decode_mode"],

        "transcription_status": row["status"],

        "scoring_error_type": None,
        "scoring_error_message": None,

        "reference_norm": reference_norm,
        "hypothesis_norm": hypothesis_norm,

        "n_punctuation_reference": reference_punctuation,
        "n_punctuation_hypothesis": hypothesis_punctuation,

        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # WER is undefined if the normalized reference contains no words.
    if reference_norm.strip() == "":
        return {
            **base_record, 
            "scoring_status": "empty_reference", 
            **empty_metrics()
        }

    scores = compute_wer(reference_norm,hypothesis_norm)

    return {
        **base_record,

        "scoring_status": "ok",

        "wer": scores.wer,
        "substitutions": scores.substitutions,
        "deletions": scores.deletions,
        "insertions": scores.insertions,
        "hits": scores.hits,
        "n_reference_words": scores.n_reference_words,
    }


def run_scoring(
    manifest: pd.DataFrame,
    transcriptions: pd.DataFrame,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """
    Normalize and score transcription results against the manifest.
    Samples without a transcription result are marked as pending_transcription. 
    Samples with failed transcription are marked as transcription_error.

    Args:
        manifest:
            Manifest containing reference transcripts and dataset names.
            
        transcriptions:
            Saved output from the Whisper transcription pipeline.

        fail_fast:
            If True, stop immediately if normalization or metric
            computation fails for a sample.

    Returns:
        DataFrame containing one scoring result row per manifest sample.
    """

    validate_manifest(manifest)

    validate_transcriptions(transcriptions)

    manifest = manifest.copy()
    transcriptions = transcriptions.copy()

    manifest["sample_id"] = (manifest["sample_id"].astype(str))
    transcriptions["sample_id"] = (transcriptions["sample_id"].astype(str))

    validate_sample_ids(manifest, transcriptions)

    # The manifest is the source of truth for dataset/sample metadata 
    # Duplicated metadata is dropped fom the transcription results before joining
    transcriptions = transcriptions.drop(columns=["dataset", "filepath"], errors="ignore")

    # Join manifest to transcription results
    # Left join so every manifest sample remains present even if it has not yet been transcribed.
    merged = manifest.merge(
        transcriptions,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    results = []

    total_samples = len(merged)

    print(f"Samples to score: {total_samples}")
    print()

    for position, (_, row) in enumerate(merged.iterrows(), start=1):
        sample_id = str(row["sample_id"])

        print(f"[{position}/{total_samples}] Scoring {sample_id}...")

        # No transcription row exists yet.
        if pd.isna(row["status"]):
            record = build_pending_record(row)

            print("\tPENDING | No transcription available.")
            results.append(record)
            continue

        # Whisper attempted the recording but failed.
        if row["status"] != "ok":
            record = (build_error_record(row))

            print(f"\tSKIPPED | Transcription status = {row['status']}")
            results.append(record)
            continue

        # Successful Whisper output: normalize and calculate WER.
        try:
            record = score_sample(row)

            if record["scoring_status"] == "ok":
                print(f"\tOK | WER = {record['wer']:.4f}")

            elif (record["scoring_status"]== "empty_reference"):
                print("\tEMPTY REFERENCE | WER undefined.")

        except Exception as error:
            record = build_error_record(row, error)

            print(f"\tERROR | {type(error).__name__}: {error}")

            if fail_fast:
                raise RuntimeError(
                    f"Scoring failed for sample {sample_id}: {error}"
                ) from error

        results.append(record)

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    print()
    print("Scoring complete.")

    print(f"Successfully scored: {(results_df['scoring_status'] == 'ok').sum()}")
    print(f"Empty references: {(results_df['scoring_status'] == 'empty_reference').sum()}")
    print(f"Transcription errors: {(results_df['scoring_status'] == 'transcription_error').sum()}")
    print(f"Pending transcription: {(results_df['scoring_status'] == 'pending_transcription').sum()}")
    print(f"Scoring errors: {(results_df['scoring_status'] == 'scoring_error').sum()}")

    return results_df