"""
Shared checks for manifests and results tables.

Any per-sample pipeline (ASR transcription, ASR scoring, DSP feature
extraction) can use these to fail early, before any processing starts,
if its input table is missing columns, has empty values, or has missing
or duplicate sample IDs.

TODO:
Optionally refactor ASR (transcription and scoring) to use these functions.
"""

import pandas as pd


def _is_blank(value) -> bool:
    """Return True for missing values (NaN/None) and empty 
    or whitespace-only strings."""

    if pd.isna(value):
        return True

    return isinstance(value, str) and value.strip() == ""


def require_columns(df: pd.DataFrame, columns: list[str], label: str = "Manifest") -> None:
    """
        Check that every required column exists and has a value in every row.
     """

    columns = set(columns)

    missing = columns - set(df.columns)
    
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    for column in sorted(columns):
        blank_rows = df[column].map(_is_blank)

        if blank_rows.any():
            raise ValueError(
                f"{label} has {int(blank_rows.sum())} missing or blank "
                f"values in required column: {column}"
            )


def validate_sample_ids(df: pd.DataFrame, label = "Manifest") -> None:
    """
    Check that sample_id is present, non-empty and unique in every row.

    sample_id is the key used to join the manifest and results, so a missing or repeated 
    ID would silently drop or duplicate rows when the tables are combined.
    """

    if "sample_id" not in df.columns:
        raise ValueError(f"{label} has no sample_id column.")

    blank_ids = df["sample_id"].map(_is_blank)

    if blank_ids.any():
        raise ValueError(
            f"{label} contains {int(blank_ids.sum())} missing or empty sample_id values."
        )

    sample_ids = df["sample_id"].astype(str)

    duplicated = sample_ids[sample_ids.duplicated()].unique()

    if len(duplicated) > 0:
        raise ValueError(
            f"{label} contains duplicate sample_id values: {duplicated.tolist()}"
        )