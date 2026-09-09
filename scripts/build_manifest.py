from pathlib import Path

import pandas as pd

from src.data.loaders import load_saa
from src.data.schema import MANIFEST_COLUMNS, REQUIRED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_PATH = MANIFEST_DIR / "full_manifest.csv"

# make sure manifest is all good before writing to disk
def validate_manifest(manifest: pd.DataFrame, data_root: Path) -> None:

    # Check the schema is exactly what we expect.
    if manifest.columns.tolist() != MANIFEST_COLUMNS:
        raise ValueError("Manifest columns do not match MANIFEST_COLUMNS.")

    # sample_id must uniquely identify every recording.
    if manifest["sample_id"].duplicated().any():
        duplicates = manifest.loc[manifest["sample_id"].duplicated(keep=False), "sample_id",].tolist()
        raise ValueError(f"Duplicate sample_id values found: {duplicates}")

    # Required columns must contain a value for every row.
    for column in REQUIRED_COLUMNS:
        if manifest[column].isna().any():
            raise ValueError(f"Required column contains missing values: {column}")

        if manifest[column].dtype == object:
            empty_mask = manifest[column].astype(str).str.strip() == ""

            if empty_mask.any():
                raise ValueError(f"Required column contains empty values: {column}")

    # Every filepath in the manifest must correspond to a real file.
    missing_paths = []

    for filepath in manifest["filepath"]:
        full_path = data_root / filepath

        if not full_path.exists():
            missing_paths.append(filepath)

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest filepaths do not exist. "
            f"First few: {missing_paths[:5]}"
        )


def main() -> None:
    # Each loader returns the same schema.
    manifests = [
        load_saa(DATA_ROOT),

        # TODO:
        # load_afrispeech(DATA_ROOT),
        # load_edacc(DATA_ROOT),
    ]

    full_manifest = pd.concat(manifests, ignore_index=True,)

    validate_manifest(full_manifest, DATA_ROOT)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True,)

    full_manifest.to_csv(OUTPUT_PATH, index=False,)

    print(f"Manifest written to: {OUTPUT_PATH}")
    print(f"Samples: {len(full_manifest)}")
    print()
    print("Samples by dataset:")
    print(full_manifest["dataset"].value_counts())


if __name__ == "__main__":
    main()