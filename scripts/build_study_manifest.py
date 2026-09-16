"""Build the study manifest from the full manifest.

Selects the accent groups chosen for the investigation, assigns
the harmonised accent_group label, and applies eligibility filters.

Run from project root:
    python -m scripts.build_study_manifest
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "full_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_PATH = OUTPUT_DIR / "study_manifest.csv"


# decrease likelihood of age related confounds
MIN_AGE = 18
MAX_AGE = 65

# Accent group definitions: 
# Each entry maps an accent_group label to a selection rule defined by
# native_language (and country for american english)
ACCENT_GROUPS = {
    "american_english": {
        "native_language": "english",
        "country": "usa",
    },
    "amharic": {
        "native_language": "amharic",
    },
    "hausa": {
        "native_language": "hausa",
    },
    "kiswahili": {
        "native_language": "kiswahili",
    },
    "tigrigna": {
        "native_language": "tigrigna",
    },
    "yoruba": {
        "native_language": "yoruba",
    },
    "hindi": {
        "native_language": "hindi",
    },
}


def load_full_manifest() -> pd.DataFrame:
    """Load the full manifest from disk."""

    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Full manifest not found: {FULL_MANIFEST}\n"
            "Run 'python -m scripts.build_full_saa_manifest' first."
        )

    return pd.read_csv(FULL_MANIFEST)


def select_accent_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select SAA rows belonging to the defined accent groups and assign labels."""

    saa = manifest[manifest["dataset"] == "saa"].copy()

    selected = []

    for group_label, rules in ACCENT_GROUPS.items():

        mask = pd.Series(True, index=saa.index) # first assumes every SAA row could belong to this group

        # then applies every rule defined for the group
        for column, value in rules.items():
            mask = mask & (saa[column] == value)

        group_df = saa[mask].copy()

        if group_df.empty:
            raise ValueError(f"No SAA samples found for accent group '{group_label}'.")

        group_df["accent_group"] = group_label
        selected.append(group_df)

    return pd.concat(selected, ignore_index=True)



def filter_adults(manifest: pd.DataFrame) -> pd.DataFrame:
    """Remove speakers under MIN_AGE, over MAX_AGE or with an unknown age."""

    age = pd.to_numeric(manifest["age"], errors="coerce")
    eligible = (age.notna() & (age >= MIN_AGE) & (age <= MAX_AGE))

    return manifest[eligible].copy()



def print_summary(manifest: pd.DataFrame) -> None:
    """Print a summary of the final study manifest."""

    summary = (
        manifest
        .groupby("accent_group")
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

    print(
        f"\nTotal samples: {len(manifest)}"
    )

    print(
        f"Total speakers: {manifest['speaker_id'].nunique()}\n"
    )


def main() -> None:
    full = load_full_manifest()

    study = select_accent_groups(full)
    study = filter_adults(study)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    study.to_csv(OUTPUT_PATH, index=False)

    print(f"Study manifest written to: {OUTPUT_PATH}\n")
    print_summary(study)



if __name__ == "__main__":
    main()