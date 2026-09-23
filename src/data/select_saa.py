"""SAA-specific study sample selection."""

import pandas as pd


# Age range chosen to reduce age-related confounds on pitch and
# formant measurements while retaining the majority of speakers.
MIN_AGE = 18
MAX_AGE = 65

# Each entry maps an accent_group label to a selection rule on
# native_language, with country added where needed to distinguish
# subsets (e.g. American English vs other English L1 speakers).
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

# Include only known sexes.
# Exclude other/NaN etc due to downstream sex-specific analysis.
STUDY_SEXES = {
    "male",
    "female",
}


def _filter_adults(manifest: pd.DataFrame) -> pd.DataFrame:
    """Remove speakers under MIN_AGE, over MAX_AGE, or with unknown age.

    Uses pd.to_numeric so that non-numeric age values (e.g. blanks
    or strings) are coerced to NaN and excluded rather than causing
    a type error.
    """

    age = pd.to_numeric(manifest["age"], errors="coerce")
    eligible = age.notna() & (age >= MIN_AGE) & (age <= MAX_AGE)

    return manifest[eligible].copy()


def _select_accent_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select SAA rows matching the defined accent groups and assign group labels."""

    saa = manifest[manifest["dataset"] == "saa"].copy()

    selected = []

    for group_label, rules in ACCENT_GROUPS.items():

        mask = pd.Series(True, index=saa.index)

        for column, value in rules.items():
            mask = mask & (saa[column] == value)

        group_df = saa[mask].copy()

        if group_df.empty:
            raise ValueError(
                f"No SAA samples found for accent group '{group_label}'."
            )

        group_df["accent_group"] = group_label
        selected.append(group_df)

    return pd.concat(selected, ignore_index=True)

def _filter_known_sex(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep samples with known male or female sex metadata."""

    sex = (manifest["sex"].astype("string").str.strip().str.lower())
    eligible = sex.isin(STUDY_SEXES)

    return manifest[eligible].copy()

def select_saa_study(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the SAA portion of the study manifest."""

    study = _select_accent_groups(manifest)
    study = _filter_adults(study)
    study = _filter_known_sex(study)

    return study