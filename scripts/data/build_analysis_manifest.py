"""Build the final analysis manifest from the study manifest and QC results.

The current audio exclusion rules are reapplied to the stored QC measurements.
Only QC-passing recordings are eligible for analysis selection.

All QC-passing SAA recordings are retained, while one recording is randomly
selected per AfriSpeech source user ID.

Run from project root:
    python -m scripts.data.build_analysis_manifest
"""

from pathlib import Path

import pandas as pd

from src.data.schema import MANIFEST_COLUMNS
from src.data.select_analysis import (
    RANDOM_SEED,
    select_analysis_sample,
)
from src.data.validation import validate_manifest
from src.preprocessing.exclusion import (
    REASON_SEPARATOR,
    apply_exclusion_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MANIFEST_ROOT = PROJECT_ROOT / "data" / "manifests"

STUDY_MANIFEST = MANIFEST_ROOT / "study_manifest.csv"
QC_TABLE = PROCESSED_ROOT / "audio_qc_study_manifest.csv"
OUTPUT_PATH = MANIFEST_ROOT / "analysis_manifest.csv"

GROUP_COLUMNS = [
    "dataset",
    "accent_group",
]


def build_analysis_manifest(
    study: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """Apply QC decisions and select the final analysis recordings."""

    if study.columns.tolist() != MANIFEST_COLUMNS:
        raise ValueError(
            "Study manifest columns do not match MANIFEST_COLUMNS."
        )

    required_qc_columns = {
        "sample_id",
        "processed_filepath",
        "error",
        "exclude",
    }

    missing_columns = required_qc_columns - set(quality.columns)

    if missing_columns:
        raise ValueError(
            f"QC table is missing columns: {sorted(missing_columns)}"
        )

    if study["sample_id"].duplicated().any():
        raise ValueError(
            "Study manifest contains duplicate sample_id values."
        )

    if quality["sample_id"].duplicated().any():
        raise ValueError(
            "QC table contains duplicate sample_id values."
        )

    # The QC table must contain exactly one result for every recording
    # in the study manifest. A mismatch may indicate outdated QC results.
    study_ids = set(study["sample_id"])
    quality_ids = set(quality["sample_id"])

    if study_ids != quality_ids:
        raise ValueError(
            "Study manifest and QC table contain different sample IDs. "
            f"Missing QC: {len(study_ids - quality_ids)}; "
            f"unexpected QC: {len(quality_ids - study_ids)}."
        )

    # Failed preprocessing means that QC measurements may be incomplete.
    # The analysis sample must not be built until these errors are resolved.
    failed = quality[
        quality["error"].fillna("").astype(str).str.strip() != ""
    ]

    if not failed.empty:
        raise ValueError(
            f"{len(failed)} recording(s) failed preprocessing. "
            "Resolve these before building the analysis manifest."
        )

    # Validate the exclusion values explicitly. Converting strings
    # directly with astype(bool) would incorrectly treat "False" as True.
    exclude = (
        quality["exclude"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
        })
    )

    if exclude.isna().any():
        raise ValueError(
            "QC table contains invalid exclusion values."
        )

    # QC exclusions are applied before sampling, ensuring that each
    # AfriSpeech source ID is selected from its passing recordings.
    passed = quality.loc[
        ~exclude,
        ["sample_id", "processed_filepath"],
    ].copy()

    if passed.empty:
        raise ValueError(
            "No recordings passed audio QC."
        )

    if (
        passed["processed_filepath"].isna().any()
        or passed["processed_filepath"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(
            "QC-passing recordings contain missing processed filepaths."
        )

    # The study manifest supplies the original metadata, while the
    # QC table supplies the path to each processed WAV file.
    eligible = (
        study
        .drop(columns=["processed_filepath"])
        .merge(
            passed,
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
    )

    eligible = eligible[MANIFEST_COLUMNS]

    # SAA recordings are retained and AfriSpeech contributes one
    # randomly selected recording per source user ID.
    analysis = select_analysis_sample(
        eligible,
        random_seed=RANDOM_SEED,
    )

    return analysis[MANIFEST_COLUMNS]


def check_processed_files_exist(
    manifest: pd.DataFrame,
) -> None:
    """Verify that every selected processed audio file exists."""

    missing_paths = [
        filepath
        for filepath in manifest["processed_filepath"]
        if not (PROCESSED_ROOT / filepath).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} processed audio files are missing. "
            f"First few: {missing_paths[:5]}"
        )


def print_summary(
    study: pd.DataFrame,
    quality: pd.DataFrame,
    analysis: pd.DataFrame,
) -> None:
    """Print final recording counts and dataset composition."""

    # Use study metadata for all three stages so that the same
    # dataset and accent labels are used throughout the summary.
    passing_ids = set(
        quality.loc[
            ~quality["exclude"],
            "sample_id",
        ]
    )

    passed = study[
        study["sample_id"].isin(passing_ids)
    ]

    print("\nRecordings at each stage:")

    counts = pd.DataFrame({
        "study": (
            study
            .groupby(GROUP_COLUMNS)
            .size()
        ),
        "passing_qc": (
            passed
            .groupby(GROUP_COLUMNS)
            .size()
        ),
        "analysis": (
            analysis
            .groupby(GROUP_COLUMNS)
            .size()
        ),
    }).fillna(0).astype(int)

    print(counts.to_string())

    # AfriSpeech speaker_id stores the source user_ids value.
    # These counts do not represent independently verified speakers.
    print("\nAfriSpeech source user IDs at each stage:")

    source_id_counts = pd.DataFrame({
        "study": (
            study[study["dataset"] == "afrispeech"]
            .groupby("accent_group")["speaker_id"]
            .nunique()
        ),
        "passing_qc": (
            passed[passed["dataset"] == "afrispeech"]
            .groupby("accent_group")["speaker_id"]
            .nunique()
        ),
        "analysis": (
            analysis[analysis["dataset"] == "afrispeech"]
            .groupby("accent_group")["speaker_id"]
            .nunique()
        ),
    }).fillna(0).astype(int)

    print(source_id_counts.to_string())

    print("\nQC exclusions by reason:")

    excluded = quality[
        quality["exclude"]
    ]

    if excluded.empty:
        print("No recordings excluded.")

    else:
        # A recording may fail multiple rules, so each reason is
        # counted separately rather than only recording exclusions.
        reasons = excluded.assign(
            reason=excluded["exclude_reason"].str.split(
                REASON_SEPARATOR
            )
        ).explode("reason")

        print(
            reasons["reason"]
            .value_counts()
            .to_string()
        )

    # These are source-provided labels, not independently verified
    # biological sex measurements.
    print("\nFinal recordings by dataset, accent and sex label:")

    print(
        analysis
        .groupby(["dataset", "accent_group", "sex"])
        .size()
        .rename("recordings")
        .to_string()
    )

    afrispeech = analysis[
        analysis["dataset"] == "afrispeech"
    ]

    if not afrispeech.empty:
        print("\nAfriSpeech domain distribution:")

        print(
            afrispeech
            .groupby(["accent_group", "domain"])
            .size()
            .rename("recordings")
            .to_string()
        )


def main() -> None:
    if not STUDY_MANIFEST.exists():
        raise FileNotFoundError(
            f"Study manifest not found: {STUDY_MANIFEST}"
        )

    if not QC_TABLE.exists():
        raise FileNotFoundError(
            f"QC table not found: {QC_TABLE}\n"
            "Run preprocessing on the study manifest first."
        )

    study = pd.read_csv(
        STUDY_MANIFEST,
        keep_default_na=False,
    )

    quality = pd.read_csv(
        QC_TABLE,
        keep_default_na=False,
    )

    # Check that the original study recordings still exist.
    validate_manifest(
        study,
        RAW_ROOT,
    )

    # Preprocessing failures must be checked before reapplying QC,
    # since failed recordings may have incomplete measurements.
    if "error" not in quality.columns:
        raise ValueError(
            "QC table is missing the error column."
        )

    if quality["error"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError(
            "QC table contains preprocessing errors. "
            "Resolve these before rebuilding the analysis manifest."
        )

    # Reapply the current rules to existing measurements. Changing a
    # QC threshold does not require regenerating the processed audio.
    quality = apply_exclusion_rules(
        quality
    )

    analysis = build_analysis_manifest(
        study,
        quality,
    )

    # Validate the final manifest before writing it.
    validate_manifest(
        analysis,
        RAW_ROOT,
    )

    check_processed_files_exist(
        analysis
    )

    afrispeech = analysis[
        analysis["dataset"] == "afrispeech"
    ]

    if afrispeech["speaker_id"].duplicated().any():
        raise ValueError(
            "Multiple AfriSpeech recordings share a source user ID."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Analysis manifest written to: {OUTPUT_PATH}"
        f"\nTotal recordings: {len(analysis)}"
    )

    print_summary(
        study,
        quality,
        analysis,
    )


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.data.build_analysis_manifest