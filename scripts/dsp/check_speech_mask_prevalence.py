"""Estimate how many analysis recordings have background noise too loud for
the energy speech mask, and calibrate a noise-floor measure for exclusion.

The energy mask cannot separate speech from background noise when the noise
floor sits within threshold_db of the recording's reference level. For each
recording this script measures:
- floor_gap_db: reference level minus the P{floor_percentile} frame energy
  (P2 by default), the gap between loud speech and the quietest frames. The
  reference uses the mask's reference percentile from the config. The floor
  is measured only inside the speech span (from the first to the last
  sustained loud run), so near-silent lead-ins and endings, recorded before
  any background noise starts, cannot pose as the noise floor. A recording
  with steady hum never gets quiet, so its gap stays small. A recording with
  continuous speech but brief gaps between some words still dips low, so its
  gap stays large;
- energy_dynamic_range_db: reference minus P10 over the whole recording,
  kept for comparison. It cannot tell pause-free speech from hum, because
  both have few quiet frames;
- low_floor_gap: True when floor_gap_db is below the cutoff (the config
  threshold_db by default), i.e. the noise floor is likely to sit above the
  mask threshold;
- speech_ratio: the fraction marked as speech at the configured threshold.

The script also prints a calibration table for recordings already classified
by inspecting their plots (speech_mask_floor_labels.csv: clean, noisy or
no_pause). Those recordings were used to design the measure, so the script
also writes a blind validation sheet: borderline recordings near the cutoff,
not among the labelled ones, to be labelled by listening before their
floor-gap values (in validation_key.csv) are looked at.

Each run folder also stores run_parameters.json with the settings used.

Run from project root:
    python -m scripts.dsp.check_speech_mask_prevalence
    python -m scripts.dsp.check_speech_mask_prevalence --overwrite
    python -m scripts.dsp.check_speech_mask_prevalence --cutoff-db 28
    python -m scripts.dsp.check_speech_mask_prevalence --floor-percentile 1
    python -m scripts.dsp.check_speech_mask_prevalence --run-name prevalence_p1 --floor-percentile 1

Run options:
- With no arguments, all analysis-manifest recordings are measured using
  speech_mask.json, recordings are flagged when floor_gap_db is below the
  config threshold_db, and results are written to a new numbered folder
  under results/dsp/speech_mask_tuning/ (prevalence_01, prevalence_02, ...).
- Supplying --overwrite replaces run_name_01 instead of creating a new
  numbered folder.
- Supplying --run-name changes the folder name prefix.
- Supplying --cutoff-db flags recordings against a different floor-gap
  cutoff instead of the config threshold_db.
- Supplying --floor-percentile measures the noise floor at a different low
  percentile of frame energies (default 2).
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
from src.dsp.speech_mask import (
    _runs,
    _seconds_to_samples,
    detect_speech_frames,
    frame_energy_db,
    frames_to_intervals,
)


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

DEFAULT_FLOOR_PERCENTILE = 2

# The speech span runs from the first to the last run of at least
# MIN_SPAN_RUN_FRAMES consecutive frames within SPAN_DB of the reference.
# Requiring a run rather than one frame stops a single pop setting the span.
SPAN_DB = 20.0
MIN_SPAN_RUN_FRAMES = 5

# Blind validation: recordings per floor-gap band (dB), drawn around the
# default 30 dB cutoff.
VALIDATION_BANDS = (
    (27.0, 30.0),
    (30.0, 33.0),
    (33.0, 36.0),
)
N_VALIDATION_PER_BAND = 4
VALIDATION_SEED = 0


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


def speech_span(
    energy_db: np.ndarray,
    all_zero: np.ndarray,
    reference_db: float,
) -> slice:
    """Return the frames from the first to the last sustained loud run."""

    loud = (
        (energy_db >= reference_db - SPAN_DB)
        & ~all_zero
    )

    sustained = [
        (start, end)
        for start, end in _runs(loud, target=True)
        if end - start >= MIN_SPAN_RUN_FRAMES
    ]

    if not sustained:
        raise ValueError(
            "No sustained loud run found to define the speech span."
        )

    return slice(sustained[0][0], sustained[-1][1])


def measure_recording(
    audio: np.ndarray,
    config: dict,
    floor_percentile: float,
) -> dict:
    """Return noise-floor measures and the mask speech ratio for one recording."""

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

    # Digital silence is excluded: exact zeros say nothing about the
    # background noise level during the recording.
    usable_energy = energy_db[~all_zero]

    if usable_energy.size == 0:
        raise ValueError(
            "Audio contains only digital silence."
        )

    # Same reference as the mask, so the floor gap compares directly with
    # threshold_db.
    reference_db = np.percentile(
        usable_energy,
        config["reference_percentile"],
    )

    span_energy = energy_db[
        speech_span(energy_db, all_zero, reference_db)
    ]

    floor_db = np.percentile(
        span_energy[np.isfinite(span_energy)],
        floor_percentile,
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

    # The floor gap uses a low percentile so that brief dips between words
    # count as quiet frames; a steady hum keeps even these frames high.
    return {
        "duration_s": duration_s,
        "floor_gap_db": float(reference_db - floor_db),
        "energy_dynamic_range_db": float(
            reference_db
            - np.percentile(usable_energy, 10)
        ),
        "speech_ratio": detected_speech_s / duration_s,
        "all_zero_pct": 100.0 * all_zero.sum() / len(all_zero),
    }


def measure_rows(
    rows: pd.DataFrame,
    config: dict,
    floor_percentile: float,
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
            print(f"[{number}/{len(rows)}]")

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
                floor_percentile,
            )

            results.append(
                {
                    **base,
                    **measured,
                    "low_floor_gap": measured["floor_gap_db"] < cutoff_db,
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

    return pd.DataFrame(results)


def summarise_groups(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise flagged recordings per dataset and accent group."""

    ok = measurements[measurements["status"] == "ok"]

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
            n_files=("sample_id", "size"),
            n_flagged=("low_floor_gap", "sum"),
            median_floor_gap_db=("floor_gap_db", "median"),
            median_speech_ratio=("speech_ratio", "median"),
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
    floor_percentile: float,
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
    outside_ids = set(labels["sample_id"]) - set(manifest["sample_id"])

    if outside_ids:
        qc = pd.read_csv(QC_MANIFEST_PATH)
        outside_rows = qc[qc["sample_id"].isin(outside_ids)]

        missing = outside_ids - set(outside_rows["sample_id"])

        if missing:
            raise ValueError(
                f"Labelled recordings not found in the QC table: {sorted(missing)}"
            )

        outside = labels.merge(
            measure_rows(
                outside_rows,
                config,
                floor_percentile,
                cutoff_db,
                progress=False,
            ),
            on="sample_id",
            how="inner",
        )

        in_analysis = pd.concat(
            [in_analysis, outside],
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


def write_validation_sheet(
    measurements: pd.DataFrame,
    labelled_ids: set,
    run_root: Path,
) -> None:
    """
    Write a blind validation sheet of borderline recordings and a separate
    key holding their floor-gap values.
    """

    candidates = measurements[
        (measurements["status"] == "ok")
        & ~measurements["sample_id"].isin(labelled_ids)
    ]

    chosen = []

    for low_db, high_db in VALIDATION_BANDS:

        band = candidates[
            (candidates["floor_gap_db"] >= low_db)
            & (candidates["floor_gap_db"] < high_db)
        ].sample(
            frac=1,
            random_state=VALIDATION_SEED,
        )

        # Take the first recording from each accent group before any second
        # one, so a band is not filled from a single large group.
        band = band.assign(
            group_rank=band.groupby(["dataset", "accent_group"]).cumcount()
        ).sort_values("group_rank", kind="stable")

        picked = band.head(N_VALIDATION_PER_BAND)

        if len(picked) < N_VALIDATION_PER_BAND:
            print(
                f"Note: only {len(picked)} recordings in the "
                f"{low_db:g}-{high_db:g} dB band."
            )

        chosen.append(
            picked.assign(band_db=f"{low_db:g}-{high_db:g}")
        )

    chosen = pd.concat(chosen).sample(
        frac=1,
        random_state=VALIDATION_SEED,
    )

    # The sheet holds no measurements, so labels are not influenced by them.
    chosen[["sample_id"]].assign(
        label="",
        notes="",
    ).to_csv(
        run_root / "validation_sheet.csv",
        index=False,
    )

    chosen[
        [
            "sample_id",
            "dataset",
            "accent_group",
            "band_db",
            "floor_gap_db",
            "low_floor_gap",
        ]
    ].to_csv(
        run_root / "validation_key.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate how many analysis recordings are too noisy for the "
            "energy speech mask."
        )
    )

    parser.add_argument(
        "--cutoff-db",
        type=float,
        default=None,
        help="Floor-gap cutoff for flagging. Default: config threshold_db.",
    )

    parser.add_argument(
        "--floor-percentile",
        type=float,
        default=DEFAULT_FLOOR_PERCENTILE,
        help="Low percentile used as the noise floor. Default: 2.",
    )

    parser.add_argument(
        "--run-name",
        default="prevalence",
        help="Results folder prefix. Default: prevalence.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace run_name_01 instead of creating a new numbered folder.",
    )

    return parser.parse_args()


def main() -> None:
    """Measure every analysis recording, summarise by group and calibrate."""

    args = parse_args()

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    cutoff_db = (
        args.cutoff_db
        if args.cutoff_db is not None
        else config["threshold_db"]
    )

    run_root = prepare_run_directory(
        run_name=args.run_name,
        overwrite=args.overwrite,
    )

    manifest = pd.read_csv(
        ANALYSIS_MANIFEST_PATH
    )

    with (run_root / "run_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "speech_mask_config": config,
                "floor_percentile": args.floor_percentile,
                "cutoff_db": cutoff_db,
                "span_db": SPAN_DB,
                "min_span_run_frames": MIN_SPAN_RUN_FRAMES,
                "validation_bands_db": VALIDATION_BANDS,
                "n_validation_per_band": N_VALIDATION_PER_BAND,
                "validation_seed": VALIDATION_SEED,
                "analysis_manifest": str(ANALYSIS_MANIFEST_PATH),
                "n_recordings": len(manifest),
            },
            file,
            indent=2,
        )

    print(
        f"Measuring {len(manifest)} recordings | "
        f"floor P{args.floor_percentile:g} | "
        f"flag cutoff {cutoff_db:g} dB"
    )

    measurements = measure_rows(
        manifest,
        config,
        args.floor_percentile,
        cutoff_db,
        progress=True,
    )

    summary = summarise_groups(measurements)

    measurements.to_csv(
        run_root / "recordings.csv",
        index=False,
    )

    summary.to_csv(
        run_root / "group_summary.csv",
        index=False,
    )

    labelled_ids = (
        set(pd.read_csv(LABELS_PATH)["sample_id"])
        if LABELS_PATH.exists()
        else set()
    )

    write_validation_sheet(
        measurements,
        labelled_ids,
        run_root,
    )

    # Check the cutoff against recordings already classified from their plots.
    if LABELS_PATH.exists():
        calibration = calibration_table(
            measurements,
            manifest,
            config,
            args.floor_percentile,
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
        print(calibration.round(2).to_string(index=False))

    print(f"\nRecordings flagged (floor gap below {cutoff_db:g} dB) by group:\n")
    print(
        summary.round(
            {
                "median_floor_gap_db": 1,
                "median_speech_ratio": 3,
                "pct_flagged": 1,
            }
        ).to_string(index=False)
    )

    ok = measurements[measurements["status"] == "ok"]

    print(
        f"\nOverall: {int(ok['low_floor_gap'].sum())} of {len(ok)} "
        f"recordings flagged ({100.0 * ok['low_floor_gap'].mean():.1f}%)."
    )
    print(f"Recordings with errors: {(measurements['status'] == 'error').sum()}")
    print(
        "\nBlind validation: label the recordings in validation_sheet.csv by "
        "listening, before opening validation_key.csv."
    )
    print(f"Wrote results to: {run_root}")


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.dsp.check_speech_mask_prevalence
# python -m scripts.dsp.check_speech_mask_prevalence --overwrite
# python -m scripts.dsp.check_speech_mask_prevalence --cutoff-db 28
# python -m scripts.dsp.check_speech_mask_prevalence --floor-percentile 1
# python -m scripts.dsp.check_speech_mask_prevalence --run-name prevalence_p1 --floor-percentile 1