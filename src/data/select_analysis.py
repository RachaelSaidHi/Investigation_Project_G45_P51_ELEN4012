"""Select the final analysis sample from QC-passing recordings.

All SAA recordings are retained. One recording is randomly selected per
AfriSpeech source user ID, since recordings sharing an ID cannot reliably
be assumed to belong to the same speaker.

Selection takes place after audio QC so that only eligible recordings
can be chosen.
"""

import pandas as pd


RANDOM_SEED = 42


def select_analysis_sample(
    manifest: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Retain all SAA recordings and one recording per AfriSpeech source ID."""

    required_columns = {
        "sample_id",
        "dataset",
        "speaker_id",
    }

    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    unknown_datasets = set(manifest["dataset"]) - {
        "saa",
        "afrispeech",
    }

    if unknown_datasets:
        raise ValueError(
            f"Unexpected datasets: {sorted(map(str, unknown_datasets))}"
        )

    saa = manifest[
        manifest["dataset"] == "saa"
    ].copy()

    afrispeech = manifest[
        manifest["dataset"] == "afrispeech"
    ].copy()

    if not afrispeech.empty:

        if (
            afrispeech["speaker_id"].isna().any()
            or afrispeech["speaker_id"].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(
                "AfriSpeech recordings contain missing source user IDs."
            )

        # Sorting makes selection reproducible even if the input
        # manifest rows are in a different order.
        afrispeech = afrispeech.sort_values(
            ["speaker_id", "sample_id"]
        )

        # AfriSpeech speaker_id contains the original source user_ids.
        # One recording is selected per ID without using gender labels,
        # acoustic features or ASR performance to influence the choice.
        afrispeech = (
            afrispeech
            .groupby("speaker_id")
            .sample(
                n=1,
                random_state=random_seed,
            )
        )

    selected = pd.concat(
        [saa, afrispeech],
        ignore_index=True,
    )

    return (
        selected
        .sort_values(["dataset", "accent_group", "sample_id"])
        .reset_index(drop=True)
    )


# run using:
# python -m scripts.data.build_analysis_manifest