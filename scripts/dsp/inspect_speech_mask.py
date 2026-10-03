"""Inspect and tune the shared DSP speech mask on real processed recordings.

Inspection recordings come from three sources:
1. every recording in dev_manifest.csv,
2. difficult analysis-manifest recordings selected automatically from QC
   information (never WER), excluding dev recordings:
   - per dataset and accent group: the lowest-WADA and the shortest
     recording, so every accent group is represented;
   - per dataset: the lowest high-frequency-energy recording and one
     recording whose WADA estimate is saturated at the upper limit,
3. manual extras listed in speech_mask_inspection_extras.csv, which may be
   any QC-passing recording, including ones outside the analysis manifest.

For each recording and reference percentile, the script writes:
- one plot comparing every candidate threshold: spectrogram, waveform
  envelope, frame energy with each threshold line, and one mask strip per
  threshold;
- compact "kept" and "removed" listening clips per threshold (unless
  --no-audio), joined with short silences and peak-scaled for listening;
- one summary.csv row per recording and setting.

Every run writes to its own folder under results/dsp/speech_mask_tuning/,
so different experiments never mix outputs.

Run from project root:
    python -m scripts.dsp.inspect_speech_mask
    python -m scripts.dsp.inspect_speech_mask --overwrite
    python -m scripts.dsp.inspect_speech_mask --thresholds 30 35 --no-audio
    python -m scripts.dsp.inspect_speech_mask --run-name percentile_sensitivity --thresholds 30 --percentiles 90 95 99

Run options:
- With no arguments, thresholds of 25, 30, 35 and 40 dB are compared at the
  reference percentile in speech_mask.json, and results are written to the
  threshold_sweep run folder.
- An existing run folder is never reused. Supplying --overwrite deletes and
  replaces it; supplying --run-name writes to a different folder instead.
- Supplying --thresholds compares a different set of candidate thresholds.
- Supplying --percentiles runs each listed reference percentile.
- Supplying --no-audio skips the listening clips, which is faster when only
  the plots and summary are needed.
"""

import argparse
import json
import shutil
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from src.common.audio_io import (
    SAMPLE_RATE,
    load_processed_audio,
)
from src.dsp.speech_mask import (
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

DEV_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dev_manifest.csv"
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

MANUAL_EXTRAS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dsp"
    / "speech_mask_inspection_extras.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dsp"
    / "speech_mask_tuning"
)


DEFAULT_THRESHOLDS_DB = (
    25.0,
    30.0,
    35.0,
    40.0,
)

# Automatic difficult-recording selection. Group-level rules guarantee every
# accent group is represented; dataset-level rules add stress cases without
# multiplying the inspection workload.
N_LOWEST_WADA_PER_GROUP = 1
N_SHORTEST_PER_GROUP = 1
N_LOWEST_HF_PER_DATASET = 1
N_SATURATED_HIGH_PER_DATASET = 1

QC_COLUMNS = [
    "sample_id",
    "processed_filepath",
    "dataset",
    "accent_group",
    "sex",
    "processed_duration_s",
    "hf_energy_pct",
    "wada_snr_db",
    "wada_snr_saturated",
]

# Listening clips join the selected regions with short silences so each cut
# is audible, and are peak-scaled so quiet material can be heard. This
# scaling affects only the listening clips, never the analysis.
LISTENING_PEAK = 0.5
CLIP_GAP_S = 0.15

# The waveform is drawn as a min/max envelope over blocks of samples. Unlike
# plotting every nth sample, an envelope cannot hide short pops or clicks.
MAX_WAVEFORM_BLOCKS = 4000


def load_config() -> dict:
    """Load the speech-mask configuration."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_valid_qc() -> pd.DataFrame:
    """Load QC rows for recordings that passed preprocessing QC."""

    qc = pd.read_csv(
        QC_MANIFEST_PATH
    )

    # Only QC-passing recordings have valid processed audio for inspection.
    valid = qc[
        qc["error"].isna()
        & ~qc["exclude"].astype(bool)
        & qc["processed_filepath"].notna()
    ]

    return valid[QC_COLUMNS].copy()


def _first_new(
    ordered: pd.DataFrame,
    n: int,
    chosen: set,
) -> list[str]:
    """Return the first n sample_ids in order that were not already chosen."""

    taken = []

    for sample_id in ordered["sample_id"]:

        if len(taken) == n:
            break

        if sample_id not in chosen:
            taken.append(
                sample_id
            )

    return taken


def select_difficult_recordings(
    analysis_ids: set,
    dev_ids: set,
    valid_qc: pd.DataFrame,
) -> list[tuple[str, str]]:
    """
    Select difficult analysis recordings from QC information.

    Returns (sample_id, reason) pairs. Ties are broken by sample_id so the
    selection is identical on every run.
    """

    candidates = valid_qc[
        valid_qc["sample_id"].isin(analysis_ids)
        & ~valid_qc["sample_id"].isin(dev_ids)
    ].sort_values(
        "sample_id"
    )

    selected = []
    chosen = set()

    def add(
        ordered: pd.DataFrame,
        n: int,
        reason: str,
    ) -> None:
        for sample_id in _first_new(ordered, n, chosen):
            chosen.add(sample_id)
            selected.append(
                (sample_id, reason)
            )

    # Group-level rules: every dataset/accent group contributes its noisiest
    # (by WADA estimate) and shortest recording.
    for _, group in candidates.groupby(
        [
            "dataset",
            "accent_group",
        ],
        sort=True,
    ):
        add(
            group.sort_values(["wada_snr_db", "sample_id"]),
            N_LOWEST_WADA_PER_GROUP,
            "lowest_wada_in_dataset_accent",
        )

        add(
            group.sort_values(["processed_duration_s", "sample_id"]),
            N_SHORTEST_PER_GROUP,
            "shortest_in_dataset_accent",
        )

    # Dataset-level stress cases.
    for _, group in candidates.groupby(
        "dataset",
        sort=True,
    ):
        add(
            group.sort_values(["hf_energy_pct", "sample_id"]),
            N_LOWEST_HF_PER_DATASET,
            "lowest_hf_energy_in_dataset",
        )

        # Saturated at the upper limit only. A recording saturated at the
        # lower limit is already covered by the lowest-WADA rule.
        saturated_high = group[
            group["wada_snr_saturated"].astype(bool)
            & (group["wada_snr_db"] > 0)
        ].sort_values(
            "sample_id"
        )

        add(
            saturated_high,
            N_SATURATED_HIGH_PER_DATASET,
            "wada_saturated_high_in_dataset",
        )

    return selected


def select_inspection_rows(
    valid_qc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the inspection set: dev recordings, automatic difficult analysis
    recordings and manual extras, one row per recording with every reason.
    """

    analysis = pd.read_csv(
        ANALYSIS_MANIFEST_PATH
    )

    dev = pd.read_csv(
        DEV_MANIFEST_PATH
    )

    reasons: dict[str, list[str]] = {}

    def add_reason(
        sample_id: str,
        reason: str,
    ) -> None:
        sample_reasons = reasons.setdefault(sample_id, [])

        if reason not in sample_reasons:
            sample_reasons.append(reason)

    for sample_id in dev["sample_id"]:
        add_reason(
            sample_id,
            "dev_manifest",
        )

    for sample_id, reason in select_difficult_recordings(
        analysis_ids=set(analysis["sample_id"]),
        dev_ids=set(dev["sample_id"]),
        valid_qc=valid_qc,
    ):
        add_reason(
            sample_id,
            reason,
        )

    # Manual extras cover unusual recordings found by listening that the
    # automatic QC metrics did not flag.
    if MANUAL_EXTRAS_PATH.exists():

        extras = pd.read_csv(
            MANUAL_EXTRAS_PATH
        )

        missing_columns = {"sample_id", "reason"} - set(extras.columns)

        if missing_columns:
            raise ValueError(
                "Manual inspection extras are missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for row in extras.itertuples(index=False):
            add_reason(
                row.sample_id,
                f"manual: {row.reason}",
            )

    missing_from_qc = sorted(
        set(reasons)
        - set(valid_qc["sample_id"])
    )

    if missing_from_qc:
        raise ValueError(
            "Selected samples are not QC-passing rows in "
            "audio_qc_study_manifest.csv: "
            + ", ".join(missing_from_qc)
        )

    selected = valid_qc[
        valid_qc["sample_id"].isin(reasons)
    ].copy()

    selected["selection_reason"] = selected["sample_id"].map(
        lambda sample_id: "; ".join(reasons[sample_id])
    )

    return selected.sort_values(
        [
            "dataset",
            "accent_group",
            "sample_id",
        ]
    ).reset_index(
        drop=True
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


def _label(value: float) -> str:
    """Format a setting for filenames, e.g. 30.0 -> '30', 32.5 -> '32p5'."""

    return f"{value:g}".replace(".", "p")


def complement_intervals(
    intervals: list[tuple[float, float]],
    duration_s: float,
) -> list[tuple[float, float]]:
    """Return the non-speech regions between and around speech intervals."""

    gaps = []
    previous_end = 0.0

    for start_s, end_s in intervals:

        if start_s > previous_end:
            gaps.append((previous_end, start_s))

        previous_end = end_s

    if previous_end < duration_s:
        gaps.append((previous_end, duration_s))

    return gaps


def build_listening_clip(
    audio: np.ndarray,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    """
    Join the given regions into one compact clip for listening.

    Short silences separate the pieces so each cut is audible. The clip is
    peak-scaled to LISTENING_PEAK. One loud pop can dominate this scaling,
    so a removed clip containing a pop may sound quiet apart from the pop.
    """

    gap = np.zeros(
        int(CLIP_GAP_S * SAMPLE_RATE),
        dtype=np.float32,
    )

    pieces = []

    for start_s, end_s in intervals:
        pieces.append(
            audio[
                int(round(start_s * SAMPLE_RATE)):
                int(round(end_s * SAMPLE_RATE))
            ]
        )
        pieces.append(gap)

    if not pieces:
        return np.zeros(0, dtype=np.float32)

    clip = np.concatenate(pieces)
    peak = np.max(np.abs(clip))

    if peak > 0:
        clip = clip * (LISTENING_PEAK / peak)

    return clip.astype(np.float32)


def _waveform_envelope(
    audio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return block times, minima and maxima for drawing the waveform."""

    block = max(
        1,
        int(np.ceil(len(audio) / MAX_WAVEFORM_BLOCKS)),
    )

    n_blocks = int(np.ceil(len(audio) / block))

    # Pad with the final sample so padding cannot create a false extreme.
    padded = np.pad(
        audio,
        (0, n_blocks * block - len(audio)),
        mode="edge",
    ).reshape(n_blocks, block)

    times = np.arange(n_blocks) * block / SAMPLE_RATE

    return times, padded.min(axis=1), padded.max(axis=1)


def save_inspection_plot(
    row,
    audio: np.ndarray,
    energy_db: np.ndarray,
    frame_times: np.ndarray,
    hop_length: int,
    reference_db: float,
    reference_percentile: float,
    intervals_by_threshold: dict,
    output_path: Path,
) -> None:
    """Save one figure comparing every candidate threshold for one recording."""

    thresholds = list(intervals_by_threshold)
    duration_s = len(audio) / SAMPLE_RATE

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2.5, 1.5, 2, 0.4 * len(thresholds) + 0.4],
        },
    )

    # Spectrogram: weak consonants such as fricatives show as high-frequency
    # energy, which helps judge whether the mask removes real speech.
    spectrogram_db = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                audio,
                n_fft=512,
                hop_length=hop_length,
            )
        ),
        ref=np.max,
    )

    librosa.display.specshow(
        spectrogram_db,
        sr=SAMPLE_RATE,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=axes[0],
    )
    axes[0].set_ylabel("Hz")
    axes[0].set_xlabel("")

    # Waveform envelope.
    times, minima, maxima = _waveform_envelope(audio)
    axes[1].fill_between(times, minima, maxima, color="black", linewidth=0)
    axes[1].set_ylabel("Amplitude")

    # Frame energy with each candidate threshold line. -inf values from
    # all-zero frames are hidden so they do not stretch the scale.
    display_energy = np.where(
        np.isfinite(energy_db),
        energy_db,
        np.nan,
    )

    axes[2].plot(frame_times, display_energy, linewidth=0.8, color="black")
    axes[2].axhline(reference_db, color="grey", linestyle=":", label="reference")

    for index, threshold in enumerate(thresholds):
        axes[2].axhline(
            reference_db - threshold,
            linewidth=0.9,
            color=f"C{index}",
            label=f"-{threshold:g} dB",
        )

    axes[2].set_ylim(
        reference_db - max(thresholds) - 20,
        reference_db + 10,
    )
    axes[2].set_ylabel("Energy (dB)")
    axes[2].legend(loc="upper right", fontsize=7, ncol=len(thresholds) + 1)

    # One strip per candidate showing its final speech mask.
    for index, threshold in enumerate(thresholds):
        axes[3].broken_barh(
            [
                (start_s, end_s - start_s)
                for start_s, end_s in intervals_by_threshold[threshold]
            ],
            (index - 0.4, 0.8),
            color=f"C{index}",
        )

    axes[3].set_yticks(range(len(thresholds)))
    axes[3].set_yticklabels([f"{t:g} dB" for t in thresholds])
    axes[3].set_xlim(0.0, duration_s)
    axes[3].set_xlabel("Time (s)")

    figure.suptitle(
        f"{row.sample_id} | {row.dataset} | {row.accent_group} | {row.sex} | "
        f"P{reference_percentile:g}\nselected: {row.selection_reason}",
        fontsize=10,
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110)
    plt.close(figure)


def inspect_recording(
    row,
    base_config: dict,
    percentiles: list[float],
    thresholds: list[float],
    run_root: Path,
    write_audio: bool,
) -> list[dict]:
    """Run every candidate setting on one recording and write its outputs."""

    audio = load_processed_audio(
        PROCESSED_ROOT,
        row.processed_filepath,
    )

    duration_s = len(audio) / SAMPLE_RATE

    # Use the same fail-loudly conversion as the production speech mask, so
    # the inspection runs on exactly the production frame grid.
    frame_length = _seconds_to_samples(
        duration_s=base_config["frame_length_s"],
        sample_rate=SAMPLE_RATE,
        parameter_name="frame_length_s",
    )

    hop_length = _seconds_to_samples(
        duration_s=base_config["hop_length_s"],
        sample_rate=SAMPLE_RATE,
        parameter_name="hop_length_s",
    )

    actual_hop_length_s = hop_length / SAMPLE_RATE

    # Energies are recomputed here only for plots and diagnostics. The mask
    # decisions themselves come from detect_speech_frames().
    energy_db, all_zero = frame_energy_db(
        audio=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    frame_times = librosa.frames_to_time(
        np.arange(len(energy_db)),
        sr=SAMPLE_RATE,
        hop_length=hop_length,
    )

    usable_energy = energy_db[~all_zero]

    if usable_energy.size == 0:
        raise ValueError(
            "Audio contains only digital silence."
        )

    # Frame-energy dynamic range: a diagnostic only, not an SNR estimate.
    # A small value can mean noise near speech level, little silence or
    # quiet speech.
    dynamic_range_db = float(
        np.percentile(usable_energy, 95)
        - np.percentile(usable_energy, 10)
    )

    all_zero_frames = int(all_zero.sum())

    rows = []

    # Save the original processed recording alongside the kept/removed clips
    # so the mask outputs can be compared directly with the input audio.
    if write_audio:
        clip_root = (
            run_root
            / "audio"
            / row.sample_id
        )

        clip_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            PROCESSED_ROOT
            / row.processed_filepath,
            clip_root
            / "original.wav",
        )

    for reference_percentile in percentiles:

        reference_db = float(
            np.percentile(usable_energy, reference_percentile)
        )

        intervals_by_threshold = {}

        for threshold_db in thresholds:

            config = dict(
                base_config,
                reference_percentile=reference_percentile,
                threshold_db=threshold_db,
            )

            # detect_speech_frames() rather than speech_intervals(), which
            # raises below min_total_speech_s. During tuning such cases should
            # be plotted and recorded, not abort the recording.
            active = detect_speech_frames(
                audio=audio,
                sample_rate=SAMPLE_RATE,
                config=config,
            )

            intervals = frames_to_intervals(
                active=active,
                hop_length_s=actual_hop_length_s,
                duration_s=duration_s,
            )

            intervals_by_threshold[threshold_db] = intervals

            detected_speech_s = sum(
                end_s - start_s
                for start_s, end_s in intervals
            )

            if write_audio:
                clip_root = (
                    run_root
                    / "audio"
                    / row.sample_id
                )
                clip_root.mkdir(parents=True, exist_ok=True)

                setting = (
                    f"p{_label(reference_percentile)}"
                    f"_{_label(threshold_db)}db"
                )

                sf.write(
                    clip_root / f"{setting}_kept.wav",
                    build_listening_clip(audio, intervals),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )

                sf.write(
                    clip_root / f"{setting}_removed.wav",
                    build_listening_clip(
                        audio,
                        complement_intervals(intervals, duration_s),
                    ),
                    SAMPLE_RATE,
                    subtype="FLOAT",
                )

            rows.append(
                {
                    "sample_id": row.sample_id,
                    "dataset": row.dataset,
                    "accent_group": row.accent_group,
                    "sex": row.sex,
                    "selection_reason": row.selection_reason,
                    "reference_percentile": reference_percentile,
                    "threshold_db": threshold_db,
                    "duration_s": duration_s,
                    "detected_speech_s": detected_speech_s,
                    "speech_ratio": detected_speech_s / duration_s,
                    "n_intervals": len(intervals),
                    "passes_min_total_speech": (
                        detected_speech_s
                        >= base_config["min_total_speech_s"]
                    ),
                    "reference_db": reference_db,
                    "energy_dynamic_range_db": dynamic_range_db,
                    "all_zero_frames": all_zero_frames,
                    "all_zero_pct": 100.0 * all_zero_frames / len(all_zero),
                    "wada_snr_db": row.wada_snr_db,
                    "wada_snr_saturated": row.wada_snr_saturated,
                    "status": "ok",
                    "error": "",
                }
            )

        save_inspection_plot(
            row=row,
            audio=audio,
            energy_db=energy_db,
            frame_times=frame_times,
            hop_length=hop_length,
            reference_db=reference_db,
            reference_percentile=reference_percentile,
            intervals_by_threshold=intervals_by_threshold,
            output_path=(
                run_root
                / "plots"
                / f"{row.sample_id}__p{_label(reference_percentile)}.png"
            ),
        )

    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect candidate speech-mask settings on dev and difficult audio."
        )
    )

    parser.add_argument(
        "--run-name",
        default="threshold_sweep",
        help=(
            "Subfolder under results/dsp/speech_mask_tuning/. "
            "Default: threshold_sweep."
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS_DB),
        help="Candidate threshold_db values. Default: 25 30 35 40.",
    )

    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=None,
        help="Reference percentiles. Default: the value in speech_mask.json.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and replace an existing run folder.",
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip writing the kept/removed listening clips.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the speech-mask inspection."""

    args = parse_args()

    base_config = load_config()

    percentiles = (
        args.percentiles
        if args.percentiles is not None
        else [base_config["reference_percentile"]]
    )

    valid_qc = load_valid_qc()
    selected = select_inspection_rows(valid_qc)

    run_root = prepare_run_directory(
        run_name=args.run_name,
        overwrite=args.overwrite,
    )

    run_settings = {
        "base_config": base_config,
        "percentiles": percentiles,
        "thresholds_db": args.thresholds,
        "write_audio": not args.no_audio,
    }

    with (
        run_root
        / "run_settings.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_settings,
            file,
            indent=2,
        )

    # Record exactly which recordings were inspected and why.
    selected.to_csv(
        run_root / "selection.csv",
        index=False,
    )

    print(
        f"Inspecting {len(selected)} recordings | "
        f"percentiles {percentiles} | thresholds {args.thresholds} dB"
    )

    summary_rows = []

    for number, row in enumerate(
        selected.itertuples(index=False),
        start=1,
    ):
        print(f"[{number}/{len(selected)}] {row.sample_id}")

        # A failing recording is recorded and skipped rather than stopping
        # the inspection of the others.
        try:
            summary_rows.extend(
                inspect_recording(
                    row=row,
                    base_config=base_config,
                    percentiles=percentiles,
                    thresholds=args.thresholds,
                    run_root=run_root,
                    write_audio=not args.no_audio,
                )
            )

        except ValueError as exc:
            summary_rows.append(
                {
                    "sample_id": row.sample_id,
                    "dataset": row.dataset,
                    "accent_group": row.accent_group,
                    "sex": row.sex,
                    "selection_reason": row.selection_reason,
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        run_root / "summary.csv",
        index=False,
    )

    ok = summary[summary["status"] == "ok"]

    if not ok.empty:
        print("\nMedian speech ratio by dataset, percentile and threshold:")
        print(
            ok.pivot_table(
                index=["dataset", "reference_percentile"],
                columns="threshold_db",
                values="speech_ratio",
                aggfunc="median",
            )
            .round(3)
            .to_string()
        )

    print(f"\nRecordings with errors: {(summary['status'] == 'error').sum()}")
    print(f"Wrote results to: {run_root}")


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.dsp.inspect_speech_mask
# python -m scripts.dsp.inspect_speech_mask --overwrite
# python -m scripts.dsp.inspect_speech_mask --thresholds 30 35 --no-audio
# python -m scripts.dsp.inspect_speech_mask --run-name percentile_sensitivity --thresholds 30 --percentiles 90 95 99





# inspection testing:
# python -c "import pandas as pd; s = pd.read_csv('results/dsp/speech_mask_tuning/threshold_sweep_01/summary.csv'); 
# s = s[s['threshold_db'] == 30]; print(s.sort_values('energy_dynamic_range_db')
# [['sample_id', 'dataset', 'accent_group', 'energy_dynamic_range_db', 'speech_ratio']].head(10).to_string(index=False))"

