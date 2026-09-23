"""Build a small development manifest from the SAA study manifest.

Selects a configurable number of female and male speakers from each
accent group. The development manifest is intended for pipeline
development and testing, not for statistical analysis.

Run from project root:
    python -m scripts.build_dev_manifest
"""

from pathlib import Path

import pandas as pd
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDY_MANIFEST = (PROJECT_ROOT / "data" / "manifests" / "study_manifest.csv")
OUTPUT_PATH = (PROJECT_ROOT / "data" / "manifests" / "dev_manifest.csv")

SPEAKERS_PER_SEX = 1


def build_dev_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select a balanced development sample from each accent group."""

    selected = []

    for (dataset, accent_group), group_df in manifest.groupby(["dataset", "accent_group"]):

        for sex in ["female", "male"]:

            candidates = (group_df[group_df["sex"] == sex]
                            .sort_values(["speaker_id", "sample_id"])
                            .drop_duplicates(subset="speaker_id",keep="first"))

            if len(candidates) < SPEAKERS_PER_SEX:
                warnings.warn(
                    f"{dataset}/{accent_group} has only "
                    f"{len(candidates)} {sex} speaker(s); "
                    f"requested up to {SPEAKERS_PER_SEX}."
                )

            selected.append(candidates.head(SPEAKERS_PER_SEX))

    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["dataset", "accent_group", "sex", "sample_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    if not STUDY_MANIFEST.exists():
        raise FileNotFoundError(
            f"Study manifest not found: {STUDY_MANIFEST}\n"
            "Run 'python -m scripts.build_study_manifest' first."
        )

    study_manifest = pd.read_csv(STUDY_MANIFEST, keep_default_na=False)

    dev_manifest = build_dev_manifest(
        study_manifest
    )

    dev_manifest.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Development manifest written to: "
        f"{OUTPUT_PATH}\n"
    )

    print(
        dev_manifest[
            [
                "sample_id",
                "dataset",
                "accent_group",
                "sex",
                "age",
                "age_group",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTotal samples: {len(dev_manifest)}"
    )


if __name__ == "__main__":
    main()