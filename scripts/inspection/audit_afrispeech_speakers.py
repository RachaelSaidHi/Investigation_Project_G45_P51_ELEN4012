"""Select AfriSpeech speaker IDs for manual voice-consistency inspection.

The script samples retained Hausa and Yoruba speaker IDs from the study
manifest and prints several recordings for each. The recordings can be
listened to manually to assess whether each source speaker ID appears to
represent one consistent voice.

Run from project root:
    python -m scripts.inspection.audit_afrispeech_speakers
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STUDY_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "study_manifest.csv"
)

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

ACCENT_GROUPS = [
    "hausa",
    "yoruba",
]

SPEAKERS_PER_GROUP = 10
RECORDINGS_PER_SPEAKER = 3

# A fixed seed makes the audit reproducible.
RANDOM_SEED = 42


def select_audit_speakers(
    manifest: pd.DataFrame,
    accent_group: str,
) -> pd.DataFrame:
    """Select retained speakers with enough recordings for manual inspection."""

    group = manifest[
        (manifest["dataset"] == "afrispeech")
        & (manifest["accent_group"] == accent_group)
    ].copy()

    recording_counts = (
        group
        .groupby("speaker_id")
        .size()
    )

    eligible_speakers = recording_counts[
        recording_counts >= RECORDINGS_PER_SPEAKER
    ].index

    if len(eligible_speakers) < SPEAKERS_PER_GROUP:
        raise ValueError(
            f"Only {len(eligible_speakers)} {accent_group} speakers "
            f"have at least {RECORDINGS_PER_SPEAKER} recordings."
        )

    selected_speakers = (
        pd.Series(
            eligible_speakers,
            dtype="string",
        )
        .sample(
            n=SPEAKERS_PER_GROUP,
            random_state=RANDOM_SEED,
        )
        .tolist()
    )

    return group[
        group["speaker_id"].isin(
            selected_speakers
        )
    ].copy()


def print_audit_recordings(
    selected: pd.DataFrame,
    accent_group: str,
) -> None:
    """Print several recordings for each selected speaker ID."""

    print(
        f"\n{'=' * 70}"
        f"\n{accent_group.upper()}"
        f"\n{'=' * 70}"
    )

    for speaker_id, speaker_df in selected.groupby(
        "speaker_id",
        sort=True,
    ):
        recordings = (
            speaker_df
            .sort_values("sample_id")
            .head(RECORDINGS_PER_SPEAKER)
        )

        print(f"\nSpeaker: {speaker_id}")

        # Speaker-level metadata is shown so inconsistencies can be noticed
        # alongside any audible change in voice identity.
        print(
            "Metadata:"
            f" sex={recordings['sex'].iloc[0]},"
            f" age_group={recordings['age_group'].iloc[0]},"
            f" country={recordings['country'].iloc[0]}"
        )

        for row in recordings.itertuples(index=False):
            audio_path = (
                Path("data")
                / "raw"
                / row.filepath
            )

            print(
                f"  {row.sample_id}\n"
                f"    {audio_path}"
            )


def main() -> None:
    manifest = pd.read_csv(
        STUDY_MANIFEST,
        keep_default_na=False,
    )

    for accent_group in ACCENT_GROUPS:
        selected = select_audit_speakers(
            manifest,
            accent_group,
        )

        print_audit_recordings(
            selected,
            accent_group,
        )


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.inspection.audit_afrispeech_speakers