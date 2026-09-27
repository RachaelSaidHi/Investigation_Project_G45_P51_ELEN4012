"""AfriSpeech-specific study sample selection."""

import pandas as pd


# Provisional AfriSpeech study groups.
# Update after the Level 1 SAA analysis when the external-validation accents are finalised.
AFRISPEECH_GROUPS = {
    "hausa": {
        "accent_raw": "hausa",
        "source_split": "test",
    },
    "yoruba": {
        "accent_raw": "yoruba",
        "source_split": "test",
    },
}

# AfriSpeech provides age ranges rather than exact ages.
# Include only known adult age groups and exclude minors or unknown age.
# Excludes the '<18yrs' category and rows with missing age data.
ADULT_AGE_GROUPS = {
    "19-25",
    "26-40",
    "41-55",
}

# AfriSpeech allows gender to be marked as other.
# Include only known sexes.
# Exclude other/NaN etc due to downstream sex-specific analysis.
STUDY_SEXES = {
    "male",
    "female",
}


def _select_accent_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select the defined AfriSpeech study groups and assign study labels."""
    
    afrispeech = manifest[manifest["dataset"] == "afrispeech"].copy()

    selected = []

    for group_label, rules in AFRISPEECH_GROUPS.items():

        mask = pd.Series(True, index=afrispeech.index)

        for column, value in rules.items():
            mask = mask & (afrispeech[column] == value)

        group_df = afrispeech[mask].copy()

        if group_df.empty:
            raise ValueError(
                f"No AfriSpeech samples found for "
                f"study group '{group_label}'."
            )

        group_df["accent_group"] = group_label
        selected.append(group_df)

    return pd.concat(selected, ignore_index=True)


def _filter_adults(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep AfriSpeech samples belonging to known adult age groups."""

    eligible = manifest["age_group"].isin(ADULT_AGE_GROUPS)

    return manifest[eligible].copy()

def _filter_known_sex(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep samples with known male or female sex metadata."""

    sex = (manifest["sex"].astype("string").str.strip().str.lower())
    eligible = sex.isin(STUDY_SEXES)

    return manifest[eligible].copy()

def _filter_consistent_speaker_sex(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Remove speakers with conflicting male and female sex labels."""

    # Sex should be a speaker-level property. AfriSpeech contains some
    # speaker IDs whose recordings are labelled both male and female, so
    # their sex cannot be determined reliably from the source metadata.
    sex = (
        manifest["sex"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    known_sex = manifest[
        sex.isin(STUDY_SEXES)
    ].copy()

    known_sex["sex"] = sex[
        sex.isin(STUDY_SEXES)
    ]

    sex_counts = (
        known_sex
        .groupby("speaker_id")["sex"]
        .nunique()
    )

    inconsistent_speakers = sex_counts[
        sex_counts > 1
    ].index

    return manifest[
        ~manifest["speaker_id"].isin(
            inconsistent_speakers
        )
    ].copy()


def select_afrispeech_study(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the AfriSpeech portion of the study manifest."""

    study = _select_accent_groups(manifest)
    study = _filter_adults(study)
    study = _filter_known_sex(study)
    study = _filter_consistent_speaker_sex(study)

    return study