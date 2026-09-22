"""Build the study manifest from the full manifest.

Dataset-specific selection is handled by the corresponding selection
modules. This script combines the selected study data and writes the
resulting manifest.

Run from project root:
    python -m scripts.build_study_manifest
"""

from pathlib import Path

import pandas as pd

from src.data.select_saa import select_saa_study
from src.data.validation import validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
FULL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "full_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_PATH = OUTPUT_DIR / "study_manifest.csv"


def load_full_manifest() -> pd.DataFrame:
    """Load the full manifest from disk."""

    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Full manifest not found: {FULL_MANIFEST}\n"
            "Run 'python -m scripts.build_full_manifest' first."
        )

    return pd.read_csv(FULL_MANIFEST)


def print_summary(manifest: pd.DataFrame) -> None:
    """Print a summary of the final study manifest."""

    summary = (
        manifest
        .groupby(["dataset", "accent_group"])
        .agg(
            speakers=("speaker_id", "nunique"),
            samples=("sample_id", "count"),
        )
        .sort_index()
    )

    print("Accent group counts:")
    print(summary.to_string())
    print()

    print("Sex breakdown by accent group:")
    print(
        pd.crosstab(
            manifest["accent_group"],
            manifest["sex"],
        ).to_string()
    )
    print()

    print("Age summary by accent group:")
    print(
        manifest
        .groupby("accent_group")["age"]
        .agg(["min", "median", "max"])
        .to_string()
    )

    print(f"\nTotal samples: {len(manifest)}")
    print(f"Total speakers: {manifest['speaker_id'].nunique()}\n")


def main() -> None:
    full = load_full_manifest()

    # Each dataset has its own selection logic because the available
    # metadata and eligibility rules differ between datasets.
    selections = [
        select_saa_study(full),

        # TODO:
        # select_afrispeech_study(full),
    ]

    study = pd.concat(selections, ignore_index=True)

    validate_manifest(study, DATA_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study.to_csv(OUTPUT_PATH, index=False)

    print(f"Study manifest written to: {OUTPUT_PATH}\n")
    print_summary(study)


if __name__ == "__main__":
    main()