"""
Resumable DSP feature-extraction pipeline.

Runs one feature family (e.g. MFCC, pitch, formants) over every sample in a
manifest and writes one results row per sample_id to a checkpoint CSV.

The pipeline deliberately knows nothing about MFCCs, pitch, formants, or
the speech masking step. It loads audio, calls the feature module it is 
given and handles all csv logic and validation.
"""

import math
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
 
import numpy as np
import pandas as pd
 
from src.common.manifest_checks import (
    require_columns, 
    validate_sample_ids
)

from src.common.run_management import (
    load_existing_results,
    save_results_atomic,
    upsert_result
)
from src.preprocessing.audio import TARGET_SAMPLE_RATE



AudioLoader = Callable[[Path, str], np.ndarray]

# Columns written before the feature columns
ENVELOPE_COLUMNS = [
    "sample_id", 
    "dataset", 
    "manifest_index",
    "status", 
    "error_type", 
    "error_message",
    "feature_name", 
    "feature_version"
]

# Columns written after the feature columns
TRAILING_COLUMNS = [
    "audio_duration_s", 
    "elapsed_s", 
    "completed_at_utc"
]

REQUIRED_MANIFEST_COLUMNS = [
    "sample_id", 
    "dataset", 
    "processed_filepath", 
    "sex", 
    "accent_group"
]

REQUIRED_MODULE_ATTRIBUTES = [
    "FEATURE_NAME",
    "FEATURE_VERSION",
    "feature_columns",
    "extract",
]


class FeatureSchemaError(Exception):
    """
    A feature module's output does not match the columns it declared.
 
    This is a bug in the feature code, not a problem with a sample, so it
    stops the run rather than being recorded as an error row.
    """


# Validation
def validate_feature_module(feature_module) -> None:
    """
    Check that a feature module provides everything the pipeline needs.
 
    Done once before the run starts, so a missing function fails immediately
    rather than on the first sample.
    """
 
    missing = [
        attribute
        for attribute in REQUIRED_MODULE_ATTRIBUTES
        if not hasattr(feature_module, attribute)
    ]
 
    if missing:
        raise TypeError(
            f"Feature module is missing required attributes: {missing}"
        )


def get_result_columns(feature_module, config: dict) -> list[str]:
    """
    Return the full, ordered column list of this feature's results file.
 
    Envelope columns, then the feature's own columns, then timing columns.
    Also checks that the feature's columns are unique and do not reuse an
    envelope or timing column name.
    """
 
    feature_columns = list(feature_module.feature_columns(config))
 
    if len(feature_columns) != len(set(feature_columns)):
        raise FeatureSchemaError(
            f"{feature_module.FEATURE_NAME}: feature_columns() contains duplicates."
        )
 
    common_feature = set(feature_columns) & set(ENVELOPE_COLUMNS + TRAILING_COLUMNS)
 
    if common_feature:
        raise FeatureSchemaError(
            f"{feature_module.FEATURE_NAME}: feature columns clash with "
            f"reserved columns: {sorted(common_feature)}"
        )
 
    return ENVELOPE_COLUMNS + feature_columns + TRAILING_COLUMNS


def check_feature_columns(
    values,
    expected_columns: list[str],
    feature_name: str,
) -> None:
    """
    Check that extract() returned a dict with exactly the declared columns.
 
    Called outside the per-sample try/except in process_sample, so a
    FeatureSchemaError is not turned into an error row. It stops the run,
    because the feature code is wrong and every sample would fail the same way.
    """
 
    if not isinstance(values, dict):
        raise FeatureSchemaError(
            f"{feature_name}: extract() must return a dict, "
            f"got {type(values).__name__}."
        )
 
    returned = set(values)
    expected = set(expected_columns)
 
    if returned != expected:
        raise FeatureSchemaError(
            f"{feature_name}: extract() returned the wrong columns. "
            f"Missing: {sorted(expected - returned)}. "
            f"Unexpected: {sorted(returned - expected)}."
        )
 
 
def is_finite_number(value) -> bool:
    """
    Return True if value converts to a float that is not NaN or infinite.
    """
 
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
 

# Result records
def build_success_record(
    row,
    manifest_index: int,
    feature_module,
    feature_values: dict,
    audio_duration_s: float,
    elapsed_s: float,
) -> dict:
    """
    Build one successful results row.
    """
 
    return {
        "sample_id": str(row["sample_id"]),
        "dataset": row["dataset"],
        "manifest_index": manifest_index,
 
        "status": "ok",
        "error_type": None,
        "error_message": None,
 
        "feature_name": feature_module.FEATURE_NAME,
        "feature_version": feature_module.FEATURE_VERSION,
 
        **feature_values,
 
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
 
 
def build_error_record(
    row,
    manifest_index: int,
    feature_module,
    error: Exception,
    audio_duration_s: float | None,
    elapsed_s: float,
) -> dict:
    """
    Build a results row for a sample that could not be processed.
 
    Feature columns are left out of the record, so upsert_result writes
    them as empty (NaN) values.
    """
 
    return {
        "sample_id": str(row["sample_id"]),
        "dataset": row["dataset"],
        "manifest_index": manifest_index,
 
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
 
        "feature_name": feature_module.FEATURE_NAME,
        "feature_version": feature_module.FEATURE_VERSION,
 
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }



# Processing
def process_sample(
    row,
    manifest_index,
    feature_module,
    config: dict,
    feature_columns: list[str],
    processed_root: Path,
    load_audio: AudioLoader,
) -> dict:
    """
    Load one sample's audio, extract its features and return a results row.
 
    Any error caused by the sample itself (unreadable audio, silence,
    non-finite feature values) is returned as an error row. A feature
    module returning the wrong columns raises FeatureSchemaError instead,
    stopping the run.
    """
 
    start = perf_counter()
    audio_duration_s = None
 
    # Anything that goes wrong while loading or extracting is a problem
    # with this sample, so it becomes an error row and the run continues.
    try:
        audio = load_audio(
            processed_root,
            str(row["processed_filepath"]),
        )
 
        audio_duration_s = len(audio) / TARGET_SAMPLE_RATE
 
        values = feature_module.extract(
            audio,
            TARGET_SAMPLE_RATE,
            row,
            config,
        )
 
    except Exception as error:
        return build_error_record(
            row=row,
            manifest_index=manifest_index,
            feature_module=feature_module,
            error=error,
            audio_duration_s=audio_duration_s,
            elapsed_s=perf_counter() - start,
        )
 
    # Deliberately outside the try since a FeatureSchemaError is a code bug and stops the run.
    check_feature_columns(
        values,
        feature_columns,
        feature_module.FEATURE_NAME,
    )
 
    # NaN, infinite or non-numeric values make this sample an error row, so
    # every "ok" row is guaranteed to have a complete, finite set of features.
    bad_columns = [
        column
        for column in feature_columns
        if not is_finite_number(values[column])
    ]
 
    if bad_columns:
        return build_error_record(
            row=row,
            manifest_index=manifest_index,
            feature_module=feature_module,
            error=ValueError(f"Non-finite feature values in: {bad_columns}"),
            audio_duration_s=audio_duration_s,
            elapsed_s=perf_counter() - start,
        )
 
    feature_values = {
        column: float(values[column])
        for column in feature_columns
    }
 
    return build_success_record(
        row=row,
        manifest_index=manifest_index,
        feature_module=feature_module,
        feature_values=feature_values,
        audio_duration_s=audio_duration_s,
        elapsed_s=perf_counter() - start,
    )


def run_feature_manifest(
    manifest: pd.DataFrame,
    feature_module,
    config: dict,
    processed_root: Path,
    results_path: Path,
    load_audio: AudioLoader,
    retry_errors: bool = False,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """
    Extract one feature family for every sample in a manifest, with resumability.
 
    Samples that already succeeded are skipped. Samples that previously
    failed are skipped by default and reprocessed with retry_errors=True.
    Results are saved after every sample.
 
    Args:
        manifest:
            The run's manifest snapshot. Row order defines manifest_index.
 
        feature_module:
            Module/object providing FEATURE_NAME, FEATURE_VERSION,
            feature_columns() and extract(). 
 
        config:
            Configuration passed unchanged to feature_columns() and extract().
 
        processed_root:
            Root folder that processed_filepath values are relative to
 
        results_path:
            Path to this feature's checkpoint CSV.
 
        load_audio:
            Function (processed_root, processed_filepath) -> audio array.
            Passed in so tests can supply fake audio without touching disk.
 
        retry_errors:
            If True, samples previously recorded with status="error" are
            attempted again.
 
        fail_fast:
            If True, stop after the first sample error. The error row is
            saved before stopping.
 
    Returns:
        The complete current results table.
    """
 
    validate_feature_module(feature_module)
 
    require_columns(manifest, REQUIRED_MANIFEST_COLUMNS)
    validate_sample_ids(manifest)
 
    manifest = manifest.copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
 
    result_columns = get_result_columns(feature_module, config)
    feature_columns = result_columns[len(ENVELOPE_COLUMNS):-len(TRAILING_COLUMNS)]
 
    results = load_existing_results(results_path, result_columns)
 
    manifest_ids = set(manifest["sample_id"])
    existing_ids = set(results["sample_id"])
 
    # Results for samples absent from this manifest suggest the wrong
    # manifest or results file is being used.
    unknown_ids = existing_ids - manifest_ids
 
    if unknown_ids:
        raise ValueError(
            "Existing results contain samples that are not present in "
            f"this manifest: {sorted(unknown_ids)}"
        )

    valid_statuses = {"ok", "error"}

    unexpected_statuses = set(results["status"].dropna()) - valid_statuses

    if unexpected_statuses:
        raise ValueError(
            "Existing results contain invalid status values: "
            f"{sorted(unexpected_statuses)}"
        )
 
    successful_ids = set(results.loc[results["status"] == "ok", "sample_id"])
    error_ids = set(results.loc[results["status"] == "error", "sample_id"])
 
    # Successful samples are never repeated.
    # Failed samples are only repeated when explicitly requested.
    if retry_errors:
        skipped_ids = successful_ids
    else:
        skipped_ids = successful_ids | error_ids
 
    total_samples = len(manifest)
 
    print(f"Feature: {feature_module.FEATURE_NAME} (version {feature_module.FEATURE_VERSION})")
    print()
    print(f"Manifest samples: {total_samples}")
    print(f"Already successful: {len(successful_ids & manifest_ids)}")
    print(f"Previous errors: {len(error_ids & manifest_ids)}")
    print(f"Samples to process: {total_samples - len(skipped_ids & manifest_ids)}")
    print()
 
    for manifest_index, (_, row) in enumerate(manifest.iterrows()):
        sample_id = row["sample_id"]
 
        if sample_id in skipped_ids:
            continue
 
        print(f"[{manifest_index + 1}/{total_samples}] {sample_id}...")
 
        record = process_sample(
            row=row,
            manifest_index=manifest_index,
            feature_module=feature_module,
            config=config,
            feature_columns=feature_columns,
            processed_root=processed_root,
            load_audio=load_audio,
        )
 
        if record["status"] == "ok":
            print(f"\tOK | {record['elapsed_s']:.2f}s")
        else:
            print(f"\tERROR | {record['error_type']}: {record['error_message']}")
 
        # Save immediately so an interrupted run loses at most one sample.
        results = upsert_result(results, record, result_columns)
        save_results_atomic(results, results_path, result_columns)
 
        if record["status"] == "error" and fail_fast:
            raise RuntimeError(
                f"Feature extraction failed for sample {sample_id}: "
                f"{record['error_message']}"
            )
 
    n_ok = int((results["status"] == "ok").sum())
    n_errors = int((results["status"] == "error").sum())
 
    print()
    print("Feature extraction complete.")
    print(f"Successful: {n_ok}")
    print(f"Errors: {n_errors}")
    print(f"Results: {results_path}")
 
    return results
