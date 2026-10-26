"""Preprocess audio and calculate quality-control measurements.

Each recording is converted to 16 kHz mono audio and written to
data/processed/audio/.  Quality measurements and exclusion decisions are
written to a QC table named after the input manifest.

The input manifest is not modified. The final analysis manifest is built
later from recordings that pass the quality-control rules.

Run from project root:
    python -m scripts.preprocessing.preprocess_audio
    python -m scripts.preprocessing.preprocess_audio --manifest data/manifests/study_manifest.csv
    python -m scripts.preprocessing.preprocess_audio --manifest data/manifests/study_manifest.csv --overwrite

Run options:
- With no arguments, the study manifest is used.
- Supplying --manifest allows a different manifest to be processed.
- Existing processed WAV files are reused by default. Supplying --overwrite
  regenerates them from the raw audio and should be used if the audio
  preprocessing method itself has changed.
"""

import argparse
from pathlib import Path, PurePosixPath

import pandas as pd

from src.data.validation import validate_manifest
from src.preprocessing.audio import (
    load_and_prepare_audio,
    preprocess_audio_file,
)
from src.preprocessing.exclusion import (
    REASON_SEPARATOR,
    apply_exclusion_rules,
)
from src.preprocessing.quality import measure_audio_quality


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

# The study manifest is the default input for preprocessing and QC.
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "study_manifest.csv"
)

# Processed paths are stored relative to data/processed rather than as
# absolute paths so that the resulting files remain portable between machines.
PROCESSED_AUDIO_DIR = "audio"

# Only metadata needed for QC inspection and grouping is copied into the
# quality table. The full manifest remains the source of all other metadata.
QC_METADATA_COLUMNS = [
    "sample_id",
    "dataset",
    "accent_group",
    "speaker_id",
    "sex",
    "domain",
]

GROUP_COLUMNS = [
    "dataset",
    "accent_group",
]

PROGRESS_INTERVAL = 50


def resolve_manifest_path(
    manifest_path: Path,
) -> Path:
    """Resolve a manifest path relative to the project root."""

    if manifest_path.is_absolute():
        return manifest_path

    return PROJECT_ROOT / manifest_path


def processed_relative_path(
    sample_id: str,
) -> str:
    """Return the processed audio path relative to data/processed/."""

    # PurePosixPath keeps forward slashes in the stored path even when the
    # preprocessing is run on Windows.
    return PurePosixPath(
        PROCESSED_AUDIO_DIR,
        f"{sample_id}.wav",
    ).as_posix()


def process_recording(
    row,
    overwrite: bool,
) -> dict:
    """Preprocess one recording and return its quality-control record."""

    relative_processed_path = processed_relative_path(
        row.sample_id
    )

    raw_path = (
        RAW_ROOT
        / row.filepath
    )

    output_path = (
        PROCESSED_ROOT
        / relative_processed_path
    )

    record = {
        column: getattr(row, column)
        for column in QC_METADATA_COLUMNS
    }

    record["processed_filepath"] = relative_processed_path
    record["error"] = ""

    try:
        # Existing processed audio can be reused because changing a QC
        # threshold does not require the waveform itself to be regenerated.
        # --overwrite is only needed when the audio preprocessing changes.
        if output_path.exists() and not overwrite:
            audio = load_and_prepare_audio(
                output_path
            )

        else:
            audio = preprocess_audio_file(
                raw_path,
                output_path,
            )

        # QC is measured from the processed waveform so that the measurements
        # describe the exact audio later supplied to the DSP and ASR pipelines.
        record.update(
            measure_audio_quality(
                audio
            )
        )

    except Exception as error:
        # Failures are recorded rather than raised immediately so that one
        # full run can identify every recording that could not be processed.
        record["error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    return record


def build_quality_table(
    manifest: pd.DataFrame,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Preprocess all manifest recordings and build the quality table."""

    records = []
    total = len(manifest)

    for count, row in enumerate(
        manifest.itertuples(index=False),
        start=1,
    ):
        records.append(
            process_recording(
                row,
                overwrite=overwrite,
            )
        )

        # Progress is printed occasionally rather than once per file so that
        # the full study run remains readable.
        if (
            count % PROGRESS_INTERVAL == 0
            or count == total
        ):
            print(
                f"  {count}/{total}"
            )

    return pd.DataFrame(
        records
    )


def section(
    title: str,
) -> None:
    """Print a section heading."""

    print(
        f"\n{'=' * 70}"
        f"\n{title}"
        f"\n{'=' * 70}"
    )


def print_summary(
    quality: pd.DataFrame,
) -> None:
    """Print preprocessing and quality-control summaries."""

    pd.set_option(
        "display.width",
        200,
    )

    pd.set_option(
        "display.max_rows",
        200,
    )

    included = quality[
        ~quality["exclude"]
    ]

    excluded = quality[
        quality["exclude"]
    ]

    section(
        "Recordings and speakers before and after current QC rules"
    )

    # Both recording and speaker counts are needed because AfriSpeech can
    # contain several recordings from the same speaker. A large file loss
    # therefore does not necessarily imply the same loss of participants.
    counts = pd.DataFrame({
        "files_before": (
            quality
            .groupby(GROUP_COLUMNS)
            .size()
        ),
        "files_after": (
            included
            .groupby(GROUP_COLUMNS)
            .size()
        ),
        "speakers_before": (
            quality
            .groupby(GROUP_COLUMNS)["speaker_id"]
            .nunique()
        ),
        "speakers_after": (
            included
            .groupby(GROUP_COLUMNS)["speaker_id"]
            .nunique()
        ),
    }).fillna(0).astype(int)

    print(
        counts.to_string()
    )

    section(
        "Exclusions by reason"
    )

    if excluded.empty:
        print(
            "No recordings excluded."
        )

    else:
        # A recording can fail more than one rule. Splitting and exploding
        # the stored reason string allows each failure reason to be counted.
        reasons = (
            excluded.assign(
                reason=excluded[
                    "exclude_reason"
                ].str.split(
                    REASON_SEPARATOR
                )
            )
            .explode(
                "reason"
            )
            .reset_index(
                drop=True
            )
        )

        print(
            pd.crosstab(
                [
                    reasons["dataset"],
                    reasons["accent_group"],
                ],
                reasons["reason"],
            ).to_string()
        )

    section(
        "Speakers lost entirely"
    )

    lost_speakers = (
        set(quality["speaker_id"])
        - set(included["speaker_id"])
    )

    if not lost_speakers:
        print(
            "None."
        )

    else:
        print(
            quality[
                quality["speaker_id"].isin(
                    lost_speakers
                )
            ]
            .groupby(
                GROUP_COLUMNS
            )["speaker_id"]
            .nunique()
            .rename(
                "speakers_lost"
            )
            .to_string()
        )

    afrispeech = quality[
        quality["dataset"] == "afrispeech"
    ]

    if not afrispeech.empty:
        section(
            "AfriSpeech domain counts before and after current QC rules"
        )

        # AfriSpeech contains different speech domains. Comparing their
        # counts before and after QC helps detect whether an exclusion rule
        # changes the composition of an accent group disproportionately.
        domain_counts = pd.DataFrame({
            "before": (
                afrispeech
                .groupby(
                    [
                        "accent_group",
                        "domain",
                    ]
                )
                .size()
            ),
            "after": (
                afrispeech[
                    ~afrispeech["exclude"]
                ]
                .groupby(
                    [
                        "accent_group",
                        "domain",
                    ]
                )
                .size()
            ),
        }).fillna(0).astype(int)

        print(
            domain_counts.to_string()
        )

    section(
        "WADA-SNR of recordings passing current QC rules"
    )

    print(
        included
        .groupby(
            GROUP_COLUMNS
        )
        .agg(
            files=(
                "sample_id",
                "size",
            ),
            minimum=(
                "wada_snr_db",
                "min",
            ),
            median=(
                "wada_snr_db",
                "median",
            ),
            maximum=(
                "wada_snr_db",
                "max",
            ),
            saturated=(
                "wada_snr_saturated",
                "sum",
            ),
        )
        .round(2)
        .to_string()
    )

    section(
        "Noise-floor gap of recordings passing current QC rules"
    )

    print(
        included
        .groupby(
            GROUP_COLUMNS
        )
        .agg(
            files=(
                "sample_id",
                "size",
            ),
            minimum=(
                "floor_gap_db",
                "min",
            ),
            median=(
                "floor_gap_db",
                "median",
            ),
            maximum=(
                "floor_gap_db",
                "max",
            ),
        )
        .round(2)
        .to_string()
    )

    section(
        "Lowest 10 WADA-SNR estimates passing current QC rules"
    )

    # The lowest estimates are printed for manual inspection before any SNR
    # threshold is considered. A low WADA value alone is not sufficient
    # justification for excluding a recording.
    print(
        included
        .nsmallest(
            10,
            "wada_snr_db",
        )[
            [
                "sample_id",
                "dataset",
                "accent_group",
                "speaker_id",
                "wada_snr_db",
                "wada_snr_saturated",
                "processed_filepath",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and calculate "
            "quality-control measurements."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "Manifest to preprocess. "
            "Defaults to data/manifests/study_manifest.csv."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate processed WAV files even if "
            "they already exist."
        ),
    )

    args = parser.parse_args()

    manifest_path = resolve_manifest_path(
        args.manifest
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        keep_default_na=False,
    )

    # The manifest is checked before processing so that problems such as
    # duplicate IDs or missing raw files fail before any outputs are created.
    validate_manifest(
        manifest,
        RAW_ROOT,
    )

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_ROOT
        / f"audio_qc_{manifest_path.stem}.csv"
    )

    print(
        f"Preprocessing {len(manifest)} recordings "
        f"from {manifest_path}"
    )

    quality = build_quality_table(
        manifest,
        overwrite=args.overwrite,
    )

    failed = quality[
        quality["error"] != ""
    ]

    if not failed.empty:
        # The incomplete QC table is retained because its error column shows
        # every failed recording. Exclusion rules cannot be applied reliably
        # until every recording has a valid set of measurements.
        quality.to_csv(
            output_path,
            index=False,
        )

        print(
            "\nPreprocessing failures:"
        )

        print(
            failed[
                [
                    "sample_id",
                    "error",
                ]
            ].to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{len(failed)} recording(s) failed preprocessing. "
            f"QC table with errors written to: {output_path}\n"
            "Exclusion rules were not applied."
        )

    # Apply the QC policy to the measured recording properties. Measurements
    # are retained so that every exclusion decision remains auditable.
    quality = apply_exclusion_rules(
        quality
    )

    quality.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nQC table written to: "
        f"{output_path}"
    )

    print_summary(
        quality
    )


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.preprocessing.preprocess_audio
# python -m scripts.preprocessing.preprocess_audio --manifest data/manifests/dev_manifest.csv
# python -m scripts.preprocessing.preprocess_audio --overwrite