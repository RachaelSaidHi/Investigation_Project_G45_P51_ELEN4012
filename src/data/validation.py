"""Validate a manifest before it is written or used.

Checks that the columns match MANIFEST_COLUMNS exactly, including their
order; that every sample_id is unique; that required columns contain no
missing or blank values; and that every filepath exists under the given
raw-data root. Processed-audio paths are not checked here.
"""

from pathlib import Path

import pandas as pd

from src.data.schema import MANIFEST_COLUMNS, REQUIRED_COLUMNS

# make sure manifest is all good before writing to disk
def validate_manifest(manifest: pd.DataFrame, data_root: Path) -> None:

    # Check the schema matches the expected columns
    if manifest.columns.tolist() != MANIFEST_COLUMNS:
        raise ValueError("Manifest columns do not match MANIFEST_COLUMNS.")

    # sample_id must uniquely identify every recording
    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(keep=False), "sample_id",].tolist()
        raise ValueError(f"Duplicate sample_id values found: {duplicates}")

    # Required columns must contain a value for every row
    for column in REQUIRED_COLUMNS:
        if manifest[column].isna().any():
            raise ValueError(f"Required column contains missing values: {column}")

        blank_values = manifest[column].map(lambda value: isinstance(value, str) and value.strip() == "")

        if blank_values.any():
            raise ValueError(f"Required column contains empty values: {column}")

    # Every filepath in the manifest must correspond to a real file
    missing_paths = []

    for filepath in manifest["filepath"]:
        full_path = data_root / filepath

        if not full_path.exists():
            missing_paths.append(filepath)

    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} manifest filepaths do not exist. First few missing: {missing_paths[:5]}"
        )
