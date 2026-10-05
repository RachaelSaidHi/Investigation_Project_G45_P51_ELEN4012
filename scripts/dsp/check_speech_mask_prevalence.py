"""Inspect persistent-noise floor gap and speech-mask behaviour.

The production floor-gap metric is calculated by
src.preprocessing.quality.measure_noise_floor_gap(). This script uses that
canonical implementation rather than maintaining a separate copy.

For each recording this script measures:
- floor_gap_db: the production preprocessing-QC floor-gap measurement;
- energy_dynamic_range_db: the speech-mask reference minus P10 over the
  whole recording, retained for comparison;
- low_floor_gap: True when floor_gap_db is below the selected cutoff
  (the production QC cutoff by default);
- speech_ratio: the fraction marked as speech by the configured speech mask.

The script also prints a calibration table for recordings previously
classified during speech-mask investigation
(speech_mask_floor_labels.csv: clean, noisy or no_pause).

Each run folder stores run_parameters.json with the settings used.

Run from project root:
    python -m scripts.dsp.check_speech_mask_prevalence
    python -m scripts.dsp.check_speech_mask_prevalence --overwrite
    python -m scripts.dsp.check_speech_mask_prevalence --cutoff-db 28

Run options:
- With no arguments, all analysis-manifest recordings are measured using
  the production floor-gap metric and speech_mask.json. Recordings are
  flagged when floor_gap_db is below the production QC cutoff, and results
  are written to a new numbered folder under
  results/dsp/speech_mask_tuning/ (prevalence_01, prevalence_02, ...).
- Supplying --overwrite replaces run_name_01 instead of creating a new
  numbered folder.
- Supplying --run-name changes the folder name prefix.
- Supplying --cutoff-db applies a different diagnostic floor-gap cutoff
  without changing the production QC rule.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.audio_io import (
    SAMPLE_RATE,
    load_processed_audio,
)
from src.common.frame_energy import frame_energy_db
from src.dsp.speech_mask import (
    _seconds_to_samples,
    detect_speech_frames,
    frames_to_intervals,
)
from src.preprocessing.exclusion import MIN_FLOOR_GAP_DB
from src.preprocessing.quality import measure_noise_floor_gap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "analysis_manifest.csv"
)

QC_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "audio_qc_study_manifest.csv"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dsp"
    / "speech_mask.json"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dsp"
    / "speech_mask_floor_labels.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dsp"
    / "speech_mask_tuning"
)


def prepare_run_directory(
    run_name: str,
    overwrite: bool,
) -> Path:
    """
    Create an empty numbered results folder for one run.

    Runs are stored as run_name_01, run_name_02, and so on.
    If --overwrite is supplied, run_name_01 is replaced.
    """

    if overwrite:
        run_root = (
            RESULTS_ROOT
            / f"{run_name}_01"
        )

        if run_root.exists():
            shutil.rmtree(
                run_root
            )

    else:
        run_number = 1

        while (
            RESULTS_ROOT
            / f"{run_name}_{run_number:02d}"
        ).exists():
            run_number += 1

        run_root = (
            RESULTS_ROOT
            / f"{run_name}_{run_number:02d}"
        )

    run_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_root


def measure_recording(
    audio: np.ndarray,
    config: dict,
) -> dict:
    """Return production floor-gap and speech-mask diagnostic measurements."""

    frame_length = _seconds_to_samples(
        duration_s=config["frame_length_s"],
        sample_rate=SAMPLE_RATE,
        parameter_name="frame_length_s",
    )

    hop_length = _seconds_to_samples(
        duration_s=config["hop_length_s"],
        sample_rate=SAMPLE_RATE,
        parameter_name="hop_length_s",
    )

    energy_db, all_zero = frame_energy_db(
        audio=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # Digital silence is excluded because exact zeros do not describe the
    # background energy present during the recording.
    usable_energy = energy_db[
        ~all_zero
    ]

    if usable_energy.size == 0:
        raise ValueError(
            "Audio contains only digital silence."
        )

    # This reference is used only for the diagnostic dynamic-range measure.
    # The production floor-gap calculation uses its own frozen QC settings.
    reference_db = np.percentile(
        usable_energy,
        config["reference_percentile"],
    )

    # Use the canonical preprocessing-QC implementation rather than
    # maintaining a second floor-gap calculation in this diagnostic script.
    floor_gap_db = measure_noise_floor_gap(
        audio
    )

    duration_s = len(audio) / SAMPLE_RATE

    # Run the real mask as well, so the speech ratio shows what it actually
    # does to this recording at the configured threshold.
    active = detect_speech_frames(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        config=config,
    )

    intervals = frames_to_intervals(
        active=active,
        hop_length_s=hop_length / SAMPLE_RATE,
        duration_s=duration_s,
    )

    detected_speech_s = sum(
        end_s - start_s
        for start_s, end_s in intervals
    )

    return {
        "duration_s": duration_s,
        "floor_gap_db": floor_gap_db,
        "energy_dynamic_range_db": float(
            reference_db
            - np.percentile(
                usable_energy,
                10,
            )
        ),
        "speech_ratio": detected_speech_s / duration_s,
        "all_zero_pct": (
            100.0
            * all_zero.sum()
            / len(all_zero)
        ),
    }


def measure_rows(
    rows: pd.DataFrame,
    config: dict,
    cutoff_db: float,
    progress: bool,
) -> pd.DataFrame:
    """Measure every row; failures are recorded rather than stopping the run."""

    results = []

    for number, row in enumerate(
        rows.itertuples(index=False),
        start=1,
    ):
        if progress and number % 50 == 0:
            print(
                f"[{number}/{len(rows)}]"
            )

        base = {
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "accent_group": row.accent_group,
            "sex": row.sex,
        }

        try:
            audio = load_processed_audio(
                PROCESSED_ROOT,
                row.processed_filepath,
            )

            measured = measure_recording(
                audio,
                config,
            )

            results.append(
                {
                    **base,
                    **measured,
                    "low_floor_gap": (
                        measured["floor_gap_db"]
                        < cutoff_db
                    ),
                    "status": "ok",
                    "error": "",
                }
            )

        except ValueError as exc:
            results.append(
                {
                    **base,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return pd.DataFrame(
        results
    )


def summarise_groups(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise flagged recordings per dataset and accent group."""

    ok = measurements[
        measurements["status"] == "ok"
    ]

    summary = (
        ok
        .groupby(
            [
                "dataset",
                "accent_group",
            ],
            sort=True,
        )
        .agg(
            n_files=(
                "sample_id",
                "size",
            ),
            n_flagged=(
                "low_floor_gap",
                "sum",
            ),
            median_floor_gap_db=(
                "floor_gap_db",
                "median",
            ),
            median_speech_ratio=(
                "speech_ratio",
                "median",
            ),
        )
        .reset_index()
    )

    summary["pct_flagged"] = (
        100.0
        * summary["n_flagged"]
        / summary["n_files"]
    )

    # How many recordings each group would keep if flagged ones were excluded.
    summary["n_kept_if_excluded"] = (
        summary["n_files"]
        - summary["n_flagged"]
    )

    return summary


def calibration_table(
    measurements: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict,
    cutoff_db: float,
) -> pd.DataFrame:
    """
    Measure the hand-labelled recordings, including any outside the analysis
    manifest, and return them sorted by floor gap.
    """

    labels = pd.read_csv(
        LABELS_PATH
    )

    in_analysis = labels.merge(
        measurements,
        on="sample_id",
        how="inner",
    )

    # Labelled recordings outside the analysis manifest are measured from
    # the QC table, which lists every processed study recording.
    outside_ids = (
        set(labels["sample_id"])
        - set(manifest["sample_id"])
    )

    if outside_ids:
        qc = pd.read_csv(
            QC_MANIFEST_PATH
        )

        outside_rows = qc[
            qc["sample_id"].isin(
                outside_ids
            )
        ]

        missing = (
            outside_ids
            - set(outside_rows["sample_id"])
        )

        if missing:
            raise ValueError(
                "Labelled recordings not found in the QC table: "
                f"{sorted(missing)}"
            )

        outside = labels.merge(
            measure_rows(
                outside_rows,
                config,
                cutoff_db,
                progress=False,
            ),
            on="sample_id",
            how="inner",
        )

        in_analysis = pd.concat(
            [
                in_analysis,
                outside,
            ],
            ignore_index=True,
        )

    return in_analysis.sort_values(
        "floor_gap_db"
    )[
        [
            "sample_id",
            "label",
            "floor_gap_db",
            "energy_dynamic_range_db",
            "speech_ratio",
            "low_floor_gap",
        ]
    ]


# method used for blind validation during tuning to test noise_floor_gap
# see method at bottom of script
# def write_validation_sheet(


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect persistent-noise floor gap and "
            "energy speech-mask behaviour."
        )
    )

    parser.add_argument(
        "--cutoff-db",
        type=float,
        default=None,
        help=(
            "Diagnostic floor-gap cutoff for flagging. "
            "Default: production QC cutoff."
        ),
    )

    parser.add_argument(
        "--run-name",
        default="prevalence",
        help=(
            "Results folder prefix. "
            "Default: prevalence."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace run_name_01 instead of "
            "creating a new numbered folder."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Measure every analysis recording, summarise by group and calibrate."""

    args = parse_args()

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(
            file
        )

    cutoff_db = (
        args.cutoff_db
        if args.cutoff_db is not None
        else MIN_FLOOR_GAP_DB
    )

    run_root = prepare_run_directory(
        run_name=args.run_name,
        overwrite=args.overwrite,
    )

    manifest = pd.read_csv(
        ANALYSIS_MANIFEST_PATH
    )

    with (
        run_root
        / "run_parameters.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "speech_mask_config": config,
                "cutoff_db": cutoff_db,
                "production_floor_gap_cutoff_db": MIN_FLOOR_GAP_DB,
                "analysis_manifest": str(
                    ANALYSIS_MANIFEST_PATH
                ),
                "n_recordings": len(
                    manifest
                ),
            },
            file,
            indent=2,
        )

    print(
        f"Measuring {len(manifest)} recordings | "
        "production floor-gap metric | "
        f"flag cutoff {cutoff_db:g} dB"
    )

    measurements = measure_rows(
        manifest,
        config,
        cutoff_db,
        progress=True,
    )

    summary = summarise_groups(
        measurements
    )

    measurements.to_csv(
        run_root / "recordings.csv",
        index=False,
    )

    summary.to_csv(
        run_root / "group_summary.csv",
        index=False,
    )

    # used for blind validation tests during tuning
    # labelled_ids = (
    #     set(pd.read_csv(LABELS_PATH)["sample_id"])
    #     if LABELS_PATH.exists()
    #     else set()
    # )
    #
    # write_validation_sheet(
    #     measurements,
    #     labelled_ids,
    #     run_root,
    # )


    # Check the cutoff against recordings already classified during
    # speech-mask investigation.
    if LABELS_PATH.exists():
        calibration = calibration_table(
            measurements,
            manifest,
            config,
            cutoff_db,
        )

        calibration.to_csv(
            run_root / "calibration.csv",
            index=False,
        )

        print(
            "\nCalibration: inspected recordings sorted by floor gap. "
            f"Noisy ones should fall below {cutoff_db:g} dB; clean and "
            "no_pause ones above it.\n"
        )

        print(
            calibration
            .round(2)
            .to_string(
                index=False
            )
        )

    print(
        f"\nRecordings flagged "
        f"(floor gap below {cutoff_db:g} dB) by group:\n"
    )

    print(
        summary
        .round(
            {
                "median_floor_gap_db": 1,
                "median_speech_ratio": 3,
                "pct_flagged": 1,
            }
        )
        .to_string(
            index=False
        )
    )

    ok = measurements[
        measurements["status"] == "ok"
    ]

    print(
        f"\nOverall: "
        f"{int(ok['low_floor_gap'].sum())} "
        f"of {len(ok)} recordings flagged "
        f"({100.0 * ok['low_floor_gap'].mean():.1f}%)."
    )

    print(
        "Recordings with errors: "
        f"{(measurements['status'] == 'error').sum()}"
    )

    print(
        f"Wrote results to: "
        f"{run_root}"
    )


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.dsp.check_speech_mask_prevalence
# python -m scripts.dsp.check_speech_mask_prevalence --overwrite
# python -m scripts.dsp.check_speech_mask_prevalence --cutoff-db 28










### used for blind validation test of noise floor gap during tuning

# def write_validation_sheet(
#     measurements: pd.DataFrame,
#     labelled_ids: set,
#     run_root: Path,
# ) -> None:
#     """
#     Write a blind validation sheet of borderline recordings and a separate
#     key holding their floor-gap values.
#     """

#     candidates = measurements[
#         (measurements["status"] == "ok")
#         & ~measurements["sample_id"].isin(labelled_ids)
#     ]

#     chosen = []

#     for low_db, high_db in VALIDATION_BANDS:

#         band = candidates[
#             (candidates["floor_gap_db"] >= low_db)
#             & (candidates["floor_gap_db"] < high_db)
#         ].sample(
#             frac=1,
#             random_state=VALIDATION_SEED,
#         )

#         # Take the first recording from each accent group before any second
#         # one, so a band is not filled from a single large group.
#         band = band.assign(
#             group_rank=band.groupby(["dataset", "accent_group"]).cumcount()
#         ).sort_values("group_rank", kind="stable")

#         picked = band.head(N_VALIDATION_PER_BAND)

#         if len(picked) < N_VALIDATION_PER_BAND:
#             print(
#                 f"Note: only {len(picked)} recordings in the "
#                 f"{low_db:g}-{high_db:g} dB band."
#             )

#         chosen.append(
#             picked.assign(band_db=f"{low_db:g}-{high_db:g}")
#         )

#     chosen = pd.concat(chosen).sample(
#         frac=1,
#         random_state=VALIDATION_SEED,
#     )

#     # The sheet holds no measurements, so labels are not influenced by them.
#     chosen[["sample_id"]].assign(
#         label="",
#         notes="",
#     ).to_csv(
#         run_root / "validation_sheet.csv",
#         index=False,
#     )

#     chosen[
#         [
#             "sample_id",
#             "dataset",
#             "accent_group",
#             "band_db",
#             "floor_gap_db",
#             "low_floor_gap",
#         ]
#     ].to_csv(
#         run_root / "validation_key.csv",
#         index=False,
#     )
