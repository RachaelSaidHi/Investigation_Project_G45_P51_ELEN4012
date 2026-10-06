"""
Shared run-management helpers for resumable, per-sample pipelines.

- Run directories: each experiment lives in its own folder containing
  run_config.json (the settings used) and manifest_snapshot.csv (a frozen
  copy of the input manifest). An existing run is only resumed if its
  configuration matches the current one.
- Checkpoints: results are stored as a CSV with one row per sample_id.
  Rows are inserted or replaced (upserted) as samples are processed and the
  file is saved atomically, so an interrupted run never leaves a half-written
  checkpoint.
 
Nothing in this module is specific to ASR or DSP. Callers pass in their own
result column list and run configuration. The module deliberately avoids
heavy imports (torch, transformers, librosa) so that any pipeline can use it
on any machine.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
 
import pandas as pd
from src.common.manifest_checks import validate_sample_ids

RUN_CONFIG_NAME = "run_config.json"
MANIFEST_SNAPSHOT_NAME = "manifest_snapshot.csv"
 
ID_COLUMN = "sample_id"
ORDER_COLUMN = "manifest_index"

# Config keys that are expected to differ between the original run and a
# resumed run, so they are never compared. A key also ignores everything
# nested under it (e.g. "paths" ignores "paths.data_root").
DEFAULT_IGNORED_CONFIG_KEYS = (
    "created_at_utc",
)


# Checkpoint files

def load_existing_results(results_path: Path, columns: list[str]) -> pd.DataFrame:
    """
    Load an existing results checkpoint, or return an empty table if none 
    exists.
    """

    if not results_path.exists():
        return pd.DataFrame(columns = columns)

    # sample_id is read as a string from the start. Converting afterwards
    # is too late: an ID such as "007" would already have become 7.
    results = pd.read_csv(
        results_path,
        dtype={ID_COLUMN: str},
    )

    missing = (set(columns)- set(results.columns))

    if missing:
        raise ValueError(
            "Existing results are "
            f"missing columns: {sorted(missing)}"
        )

    validate_sample_ids(results, label="Existing results")

    # There should be no duplicated sample ids in the existing results
    # duplicated = results.loc[
    #     results[ID_COLUMN].duplicated(),
    #     ID_COLUMN,
    # ].unique()

    # if len(duplicated) > 0:
    #     raise ValueError(
    #         "Existing results contain "
    #         f"duplicate sample IDs: {duplicated.tolist()}"
    #     )

    return results[columns]


def upsert_result(results: pd.DataFrame, record: dict, columns: list[str]) -> pd.DataFrame:
    """
    Insert or replace one sample result depending on its existance.

    Replacement happens when a previously failed sample is retried, so the
    table always holds exactly one row per sample_id (the latest attempt).
    """

    sample_id = str(record[ID_COLUMN])

    remaining = results[
        results[ID_COLUMN].astype(str) != sample_id
    ]

    new_row = pd.DataFrame(
        [{column: record.get(column) for column in columns}],
        columns=columns,
    )

    # Concatenating with an empty table can change column dtypes in
    # pandas, so the first row is returned directly instead.
    if remaining.empty:
        updated = new_row
    else:
        updated = pd.concat(
            [
                remaining, 
                new_row
            ],
            ignore_index=True,
        )

    if ORDER_COLUMN in columns:
        updated = updated.sort_values(
            ORDER_COLUMN,
            kind="stable",
        )

    return updated.reset_index(drop=True)


def save_results_atomic(results: pd.DataFrame, results_path: Path, columns: list[str]) -> None:
    """
    Save the results checkpoint without risking a half-written file.

    The CSV is first written to a temporary file next to the real one and
    then swapped into place with os.replace, which is atomic. If the run is
    killed mid-write, the previous checkpoint is still intact.
    """

    results_path.parent.mkdir(parents = True, exist_ok = True)

    temp_path = results_path.with_name(results_path.name + ".tmp")

    results.to_csv(temp_path, index=False, columns=columns)

    os.replace(temp_path, results_path)




# Run directories and configuration

def file_sha256(path: Path) -> str:
    """
    Returns the SHA-256 hash of a file.

    Used to fingerprint the input manifest so a run is never resumed, or
    combined with another run, using a manifest that has since changed.
    """

    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def prepare_run_directory(
    run_dir: Path,
    manifest_path: Path,
    run_config: dict,
    ignored_keys: tuple[str, ...] = DEFAULT_IGNORED_CONFIG_KEYS
) -> Path:
    """
    Create a new run directory, or check that an existing one can be 
    resumed.

    New run:
        Creates run_dir, writes run_config.json and copies the manifest to
        manifest_snapshot.csv.
 
    Existing run:
        Checks both files are present and that the saved configuration
        matches `run_config` (see validate_resume_config). Nothing is
        overwritten.
 
    The pipeline should always read the returned snapshot rather than the
    original manifest, so a resumed run processes exactly the same samples
    even if the manifest file has been rebuilt in the meantime.

    Returns:
        Path to the manifest snapshot belonging to the run.
    """

    config_path = run_dir / RUN_CONFIG_NAME
    manifest_snapshot_path = run_dir / MANIFEST_SNAPSHOT_NAME

    # Resume an existing run
    if run_dir.exists():

        if not config_path.exists():
            raise FileNotFoundError(
                f"Existing run directory has no {RUN_CONFIG_NAME}: "
                f"{run_dir}"
            )

        if not manifest_snapshot_path.exists():
            raise FileNotFoundError(
                "Existing run directory has no "
                f"{MANIFEST_SNAPSHOT_NAME}: {run_dir}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            previous_config = json.load(file)
            
        validate_resume_config(
            previous_config,
            run_config,
            ignored_keys=ignored_keys,
        )

        print(f"Resuming run: {run_dir}")

        return manifest_snapshot_path

    run_dir.mkdir(parents=True, exist_ok=False)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    shutil.copy2(manifest_path, manifest_snapshot_path)

    print(f"Created run: {run_dir}")

    return manifest_snapshot_path


def _flatten_config(config: dict, prefix: str = "") -> dict:
    """
    Flatten a nested config into dotted keys.
 
    Example:
        {"feature": {"name": "mfcc", "n_mfcc": 13}}
        -> {"feature.name": "mfcc", "feature.n_mfcc": 13}
 
    Lists and other values are kept as they are and compared whole.
    """
 
    flat = {}
 
    for key, value in config.items():
        dotted_key = f"{prefix}{key}"
 
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=f"{dotted_key}."))
        else:
            flat[dotted_key] = value
 
    return flat


def _is_ignored(key: str, ignored_keys: tuple[str, ...]) -> bool:
    """
    Return True if `key` is an ignored key or is nested under one.
    """
 
    return any(
        key == ignored or key.startswith(f"{ignored}.")
        for ignored in ignored_keys
    )


def validate_resume_config(
    previous: dict,
    current: dict,
    ignored_keys: tuple[str, ...] = DEFAULT_IGNORED_CONFIG_KEYS
) -> None:
    """
    Refuse to resume a run whose configuration has changed.
    
    Every setting in the saved and current configs is compared, including
    nested settings, except keys listed in `ignored_keys`. A setting that
    exists in only one of the two configs also counts as a change.
 
    Because the comparison is generic, whatever a pipeline puts in its
    run_config is automatically protected. Values that legitimately differ
    between machines (e.g. absolute paths) should be placed under a key such
    as "paths" and that key passed in `ignored_keys`.
 
    Raises:
        ValueError: listing every setting that differs.
    """

    missing = "<missing>"

    previous_flat = _flatten_config(previous)
    current_flat = _flatten_config(current)

    mismatches = []

    for key in sorted(set(previous_flat) | set(current_flat)):
 
        if _is_ignored(key, ignored_keys):
            continue
 
        previous_value = previous_flat.get(key, missing)
        current_value = current_flat.get(key, missing)
 
        if previous_value != current_value:
            mismatches.append(
                f"{key}: "
                f"previous={previous_value!r}, "
                f"current={current_value!r}"
            )

    if mismatches:
        differences = "\n".join(mismatches)
 
        raise ValueError(
            "Cannot resume this run because its configuration has changed:\n"
            f"{differences}\n\n"
            "Use a different --run-name for the new experiment."
        )
