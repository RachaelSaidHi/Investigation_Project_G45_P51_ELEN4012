"""Inspect pitch tracking on real recordings and compare candidate ceilings.

Used to settle the two provisional pitch settings, pitch_ceiling_hz and
min_voiced_s, before configs/dsp/pitch.json is frozen. WER is never read,
and no accent groups are compared.

One run does two things:

1. Table. Every analysis and dev recording is analysed at each candidate
   ceiling with the production pitch code. summary.csv has one row per
   recording and ceiling: the production pitch values plus inspection-only
   diagnostics. Praat is rerun for every ceiling, because the ceiling can
   change the pitch path Praat chooses throughout a recording.

2. Plots. A subset is plotted for inspection by eye:
   - every recording in dev_manifest.csv;
   - per dataset, the analysis recordings with the lowest floor gap, the
     lowest WADA-SNR and the shortest duration (QC information, so chosen
     independently of the pitch results);
   - the most suspicious recordings in the table at the configured ceiling:
     largest spread above the median, largest spread below it, most large
     jumps, most frames near the floor, most frames near the ceiling, most
     sensitive to the ceiling, and least voiced speech;
   - any recording that could not be measured at some ceiling;
   - fixed-seed random controls from the remaining analysis recordings of
     each dataset, so recordings that look normal in the table are checked
     too;
   - manual extras listed in configs/dsp/pitch_inspection_extras.csv
     (columns: sample_id, reason), if that file exists.

Each plot shows a low-frequency spectrogram with the pitch tracks drawn on
it, the pitch tracks against time with a histogram, and the speech mask.
A correct track lies on the lowest harmonic of the voice.

inspection_sheet.csv lists the plotted recordings with empty columns for a
verdict per ceiling (ok / suspicious / bad / unclear), a preferred ceiling
and notes.

Decision rule, fixed before inspection:
- keep the configured ceiling unless plots show sustained tracking of
  something other than the voice's fundamental;
- move to a lower ceiling only if it fixes those recordings without losing
  voiced frames or pushing frames against the ceiling in higher voices;
- keep min_voiced_s unless recordings come close to it.

Every run writes to its own numbered folder under
results/dsp/pitch_validation/, so a filled-in inspection sheet is never
overwritten by a later run.

Run from project root:
    python -m scripts.dsp.inspect_pitch
    python -m scripts.dsp.inspect_pitch --no-plots
    python -m scripts.dsp.inspect_pitch --ceilings 600 500
    python -m scripts.dsp.inspect_pitch --run-name ceiling_check --overwrite

Run options:
- With no arguments, ceilings of 600, 500 and 400 Hz are compared, together
  with the ceiling in pitch.json if it is not one of them.
- Supplying --ceilings compares a different set of candidate ceilings.
- Supplying --no-plots writes only the table, selection and sheet.
- An existing run folder is never reused. Supplying --overwrite replaces
  run_name_01; otherwise the next free number is used.
"""

import argparse
import json
import shutil
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from src.common.audio_io import (
    SAMPLE_RATE,
    load_processed_audio,
)
from src.common.run_management import file_sha256
from src.dsp.features.pitch import (
    FEATURE_VERSION,
    InsufficientVoicingError,
    extract_pitch_track,
    summarise_pitch,
)
from src.dsp.speech_mask import speech_intervals


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

PITCH_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dsp"
    / "pitch.json"
)

SPEECH_MASK_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dsp"
    / "speech_mask.json"
)

MANUAL_EXTRAS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "dsp"
    / "pitch_inspection_extras.csv"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dsp"
    / "pitch_validation"
)


DEFAULT_CEILINGS_HZ = (
    600.0,
    500.0,
    400.0,
)

# Size of the plot set. Raise or lower these to change the inspection
# workload; the table always covers every recording.
N_STRESS_PER_DATASET = 2
N_SUSPICIOUS_PER_DIAGNOSTIC = 3
N_RANDOM_CONTROLS_PER_DATASET = 5
RANDOM_SEED = 0

SEMITONES_PER_OCTAVE = 12

# Inspection aids only. These are not validity thresholds and are never
# used to exclude a recording. A change of more than 6 semitones between
# two frames 10 ms apart is far faster than a voice moves, so it points to
# a tracking jump.
LARGE_JUMP_ST = 6.0
NEAR_BOUNDARY_ST = 1.0

QC_COLUMNS = [
    "sample_id",
    "processed_filepath",
    "dataset",
    "accent_group",
    "sex",
    "processed_duration_s",
    "wada_snr_db",
    "floor_gap_db",
]

# A long analysis window gives narrow frequency bins, so the individual
# harmonics of low voices are separated in the spectrogram.
SPECTROGRAM_N_FFT = 2048
SPECTROGRAM_HOP = 160
SPECTROGRAM_MAX_HZ = 1000.0
SPECTROGRAM_RANGE_DB = 70.0

HISTOGRAM_BIN_ST = 0.5

PITCH_AXIS_TICKS_HZ = [75, 100, 150, 200, 300, 400, 500, 600]

def load_json(path: Path) -> dict:
    """Load one JSON configuration file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_valid_qc() -> pd.DataFrame:
    """Load QC rows for recordings that passed preprocessing QC."""

    qc = pd.read_csv(
        QC_MANIFEST_PATH
    )

    valid = qc[
        qc["error"].isna()
        & ~qc["exclude"].astype(bool)
        & qc["processed_filepath"].notna()
    ]

    return valid[QC_COLUMNS].copy()


def load_manual_extras() -> pd.DataFrame:
    """Load the optional list of extra recordings to plot."""

    if not MANUAL_EXTRAS_PATH.exists():
        return pd.DataFrame(
            columns=["sample_id", "reason"]
        )

    extras = pd.read_csv(
        MANUAL_EXTRAS_PATH
    )

    missing_columns = {"sample_id", "reason"} - set(extras.columns)

    if missing_columns:
        raise ValueError(
            "Manual inspection extras are missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    return extras


def load_recordings(
    valid_qc: pd.DataFrame,
    extras: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per recording to measure: analysis, dev and extras."""

    analysis_ids = set(
        pd.read_csv(ANALYSIS_MANIFEST_PATH)["sample_id"]
    )

    dev_ids = set(
        pd.read_csv(DEV_MANIFEST_PATH)["sample_id"]
    )

    wanted_ids = (
        analysis_ids
        | dev_ids
        | set(extras["sample_id"])
    )

    missing_from_qc = sorted(
        wanted_ids
        - set(valid_qc["sample_id"])
    )

    if missing_from_qc:
        raise ValueError(
            "Selected samples are not QC-passing rows in "
            "audio_qc_study_manifest.csv: "
            + ", ".join(missing_from_qc)
        )

    recordings = valid_qc[
        valid_qc["sample_id"].isin(wanted_ids)
    ].copy()

    recordings["in_analysis_manifest"] = recordings["sample_id"].isin(
        analysis_ids
    )
    recordings["in_dev_manifest"] = recordings["sample_id"].isin(
        dev_ids
    )

    return recordings.sort_values(
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


def _semitones(
    upper_hz,
    lower_hz,
):
    """Return the distance from lower_hz up to upper_hz in semitones."""

    return SEMITONES_PER_OCTAVE * np.log2(upper_hz / lower_hz)


def measure_track(
    track,
    pitch_config: dict,
) -> dict:
    """Return the production pitch values and inspection diagnostics."""

    # The statistics come from the production summarise_pitch(), so the
    # inspection cannot drift from the feature definition. A recording
    # below the voiced minimum is summarised with the minimum switched off,
    # so that it is still measured and can be examined.
    try:
        summary = summarise_pitch(
            track=track,
            pitch_config=pitch_config,
        )
        passes_min_voiced = True

    except InsufficientVoicingError:
        summary = summarise_pitch(
            track=track,
            pitch_config=dict(
                pitch_config,
                min_voiced_s=0.0,
            ),
        )
        passes_min_voiced = False

    used_f0_hz = track.f0_hz[track.used]

    near_boundary = 2 ** (NEAR_BOUNDARY_ST / SEMITONES_PER_OCTAVE)

    # Jumps are measured only between two used frames that are next to each
    # other in time. Comparing neighbours after removing unused frames would
    # treat a pitch change across a pause as a 10 ms jump.
    consecutive = track.used[:-1] & track.used[1:]

    jumps_st = np.abs(
        _semitones(
            track.f0_hz[1:][consecutive],
            track.f0_hz[:-1][consecutive],
        )
    )

    n_consecutive_pairs = int(consecutive.sum())
    n_large_jumps = int((jumps_st > LARGE_JUMP_ST).sum())

    return {
        "f0_median_hz": summary["f0_median_hz"],
        "f0_p10_hz": summary["f0_p10_hz"],
        "f0_p90_hz": summary["f0_p90_hz"],
        "f0_span_st": summary["f0_span_st"],
        "lower_spread_st": float(
            _semitones(summary["f0_median_hz"], summary["f0_p10_hz"])
        ),
        "upper_spread_st": float(
            _semitones(summary["f0_p90_hz"], summary["f0_median_hz"])
        ),
        "n_speech_frames": int(track.in_speech.sum()),
        "n_voiced_frames": int(summary["n_voiced_frames"]),
        "voiced_fraction": summary["voiced_fraction"],
        "voiced_duration_s": (
            summary["n_voiced_frames"]
            * track.time_step_s
        ),
        "passes_min_voiced": passes_min_voiced,
        "f0_min_hz": float(used_f0_hz.min()),
        "f0_max_hz": float(used_f0_hz.max()),
        "near_floor_fraction": float(
            np.mean(
                used_f0_hz
                <= pitch_config["pitch_floor_hz"] * near_boundary
            )
        ),
        "near_ceiling_fraction": float(
            np.mean(
                used_f0_hz
                >= pitch_config["pitch_ceiling_hz"] / near_boundary
            )
        ),
        "n_large_jumps": n_large_jumps,
        "large_jump_fraction": (
            n_large_jumps / n_consecutive_pairs
            if n_consecutive_pairs > 0
            else np.nan
        ),
    }


def measure_recording(
    row,
    pitch_config: dict,
    speech_mask_config: dict,
    ceilings: list[float],
) -> list[dict]:
    """Measure one recording at every candidate ceiling."""

    # An unreadable file is recorded as an error at every ceiling rather
    # than stopping the run.
    try:
        audio = load_processed_audio(
            PROCESSED_ROOT,
            row.processed_filepath,
        )
        load_error = ""

    except Exception as exc:
        audio = None
        load_error = f"{type(exc).__name__}: {exc}"

    rows = []

    for ceiling_hz in ceilings:

        candidate_config = dict(
            pitch_config,
            pitch_ceiling_hz=ceiling_hz,
        )

        result = {
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "accent_group": row.accent_group,
            "sex": row.sex,
            "in_analysis_manifest": row.in_analysis_manifest,
            "in_dev_manifest": row.in_dev_manifest,
            "processed_duration_s": row.processed_duration_s,
            "floor_gap_db": row.floor_gap_db,
            "wada_snr_db": row.wada_snr_db,
            "pitch_floor_hz": candidate_config["pitch_floor_hz"],
            "pitch_ceiling_hz": ceiling_hz,
        }

        if audio is None:
            result["status"] = "error"
            result["error"] = load_error
            rows.append(result)
            continue

        # A recording that cannot be measured is recorded and skipped
        # rather than stopping the inspection of the others.
        try:
            track = extract_pitch_track(
                audio=audio,
                sample_rate=SAMPLE_RATE,
                pitch_config=candidate_config,
                speech_mask_config=speech_mask_config,
            )

            result.update(
                measure_track(
                    track=track,
                    pitch_config=candidate_config,
                )
            )
            result["status"] = "ok"
            result["error"] = ""

        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(result)

    return rows


def add_ceiling_comparison(
    summary: pd.DataFrame,
    reference_ceiling_hz: float,
) -> pd.DataFrame:
    """Add how each ceiling's result differs from the configured ceiling."""

    reference = summary[
        (summary["status"] == "ok")
        & (summary["pitch_ceiling_hz"] == reference_ceiling_hz)
    ].set_index(
        "sample_id"
    )

    summary = summary.copy()

    summary["median_shift_st"] = _semitones(
        summary["f0_median_hz"],
        summary["sample_id"].map(reference["f0_median_hz"]),
    )

    summary["span_change_st"] = (
        summary["f0_span_st"]
        - summary["sample_id"].map(reference["f0_span_st"])
    )

    # A lower ceiling that removes suspicious high values but also loses
    # many voiced frames has not fixed the track.
    summary["voiced_frames_change_pct"] = 100.0 * (
        summary["n_voiced_frames"]
        / summary["sample_id"].map(reference["n_voiced_frames"])
        - 1.0
    )

    # One value per recording: the largest change in median or span that
    # any candidate ceiling causes.
    sensitivity = (
        summary[["median_shift_st", "span_change_st"]]
        .abs()
        .max(axis=1)
        .groupby(summary["sample_id"])
        .max()
    )

    summary["ceiling_sensitivity_st"] = summary["sample_id"].map(
        sensitivity
    )

    return summary


def select_plot_rows(
    recordings: pd.DataFrame,
    summary: pd.DataFrame,
    extras: pd.DataFrame,
    reference_ceiling_hz: float,
) -> pd.DataFrame:
    """
    Choose the recordings to plot and record why each was chosen.

    Ties are broken by sample_id so the selection is identical on every run.
    """

    reasons: dict[str, list[str]] = {}

    def add_reason(
        sample_id: str,
        reason: str,
    ) -> None:
        sample_reasons = reasons.setdefault(sample_id, [])

        if reason not in sample_reasons:
            sample_reasons.append(reason)

    def add_first(
        ordered: pd.DataFrame,
        n: int,
        reason: str,
    ) -> None:
        for sample_id in ordered["sample_id"].head(n):
            add_reason(sample_id, reason)

    for sample_id in recordings.loc[
        recordings["in_dev_manifest"],
        "sample_id",
    ]:
        add_reason(sample_id, "dev_manifest")

    # Stress cases from QC information, chosen independently of pitch.
    candidates = recordings[
        recordings["in_analysis_manifest"]
        & ~recordings["in_dev_manifest"]
    ]

    for _, group in candidates.groupby(
        "dataset",
        sort=True,
    ):
        for column, reason in [
            ("floor_gap_db", "lowest_floor_gap_in_dataset"),
            ("wada_snr_db", "lowest_wada_in_dataset"),
            ("processed_duration_s", "shortest_in_dataset"),
        ]:
            add_first(
                group.sort_values([column, "sample_id"]),
                N_STRESS_PER_DATASET,
                reason,
            )

    # The most suspicious recordings in the table at the configured ceiling.
    measured = summary[
        (summary["status"] == "ok")
        & (summary["pitch_ceiling_hz"] == reference_ceiling_hz)
    ]

    for column, ascending, reason in [
        ("upper_spread_st", False, "largest_upper_spread"),
        ("lower_spread_st", False, "largest_lower_spread"),
        ("large_jump_fraction", False, "most_large_jumps"),
        ("near_floor_fraction", False, "most_frames_near_floor"),
        ("near_ceiling_fraction", False, "most_frames_near_ceiling"),
        ("ceiling_sensitivity_st", False, "most_ceiling_sensitive"),
        ("voiced_duration_s", True, "least_voiced_speech"),
    ]:
        ranked = measured[measured[column].notna()]

        # A recording with no large jumps, no frames near a limit, or
        # unaffected by the ceiling must not be picked to fill the quota.
        if not ascending:
            ranked = ranked[ranked[column] > 0]

        add_first(
            ranked.sort_values(
                [column, "sample_id"],
                ascending=[ascending, True],
            ),
            N_SUSPICIOUS_PER_DIAGNOSTIC,
            reason,
        )

    for sample_id in summary.loc[
        summary["status"] == "error",
        "sample_id",
    ].unique():
        add_reason(sample_id, "measurement_error")

    # Controls: recordings no rule picked. Without them, errors hidden in
    # normal-looking recordings would never be seen. They are drawn within
    # each dataset so the larger corpus cannot supply all of them.
    rng = np.random.default_rng(RANDOM_SEED)

    for _, group in measured[
        measured["in_analysis_manifest"]
    ].groupby(
        "dataset",
        sort=True,
    ):
        unchosen = sorted(
            set(group["sample_id"])
            - set(reasons)
        )

        for sample_id in rng.choice(
            unchosen,
            size=min(N_RANDOM_CONTROLS_PER_DATASET, len(unchosen)),
            replace=False,
        ):
            add_reason(str(sample_id), "random_control")

    for row in extras.itertuples(index=False):
        add_reason(row.sample_id, f"manual: {row.reason}")

    selected = recordings[
        recordings["sample_id"].isin(reasons)
    ].copy()

    selected["selection_reason"] = selected["sample_id"].map(
        lambda sample_id: "; ".join(reasons[sample_id])
    )

    return selected.reset_index(
        drop=True
    )


def save_inspection_plot(
    row,
    audio: np.ndarray,
    tracks: dict,
    intervals: list[tuple[float, float]],
    measurements: dict,
    pitch_floor_hz: float,
    reference_ceiling_hz: float,
    output_path: Path,
) -> None:
    """Save one figure comparing every candidate ceiling for one recording."""

    duration_s = len(audio) / SAMPLE_RATE

    colours = {
        ceiling_hz: f"C{index}"
        for index, ceiling_hz in enumerate(tracks)
    }

    # The configured ceiling is drawn on top where the candidates agree.
    draw_order = {
        ceiling_hz: (3 if ceiling_hz == reference_ceiling_hz else 2)
        for ceiling_hz in tracks
    }

    # Long recordings get a wider figure so individual frames stay visible.
    figure = plt.figure(
        figsize=(
            min(max(14.0, 0.5 * duration_s), 30.0),
            9.5,
        ),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[3, 3, 0.3],
        width_ratios=[1, 0.07],
    )

    spectrogram_axis = figure.add_subplot(grid[0, 0])

    pitch_axis = figure.add_subplot(
        grid[1, 0],
        sharex=spectrogram_axis,
    )

    histogram_axis = figure.add_subplot(
        grid[1, 1],
        sharey=pitch_axis,
    )

    mask_axis = figure.add_subplot(
        grid[2, 0],
        sharex=spectrogram_axis,
    )

    # Spectrogram, limited to low frequencies where the first few harmonics
    # of the voice lie. A correct pitch track sits on the lowest harmonic.
    # Energy is drawn dark on white so the coloured tracks stay visible
    # where they lie on a harmonic.
    spectrogram_db = librosa.amplitude_to_db(
        np.abs(
            librosa.stft(
                audio,
                n_fft=SPECTROGRAM_N_FFT,
                hop_length=SPECTROGRAM_HOP,
            )
        ),
        ref=np.max,
    )

    bin_hz = SAMPLE_RATE / SPECTROGRAM_N_FFT
    n_bins = int(SPECTROGRAM_MAX_HZ / bin_hz) + 1

    spectrogram_axis.imshow(
        spectrogram_db[:n_bins],
        origin="lower",
        aspect="auto",
        extent=(
            0.0,
            duration_s,
            -0.5 * bin_hz,
            (n_bins - 0.5) * bin_hz,
        ),
        vmin=-SPECTROGRAM_RANGE_DB,
        vmax=0.0,
        cmap="gray_r",
    )

    for ceiling_hz, track in tracks.items():
        spectrogram_axis.scatter(
            track.times_s[track.used],
            track.f0_hz[track.used],
            s=3,
            color=colours[ceiling_hz],
            linewidths=0,
            zorder=draw_order[ceiling_hz],
        )

    spectrogram_axis.set_ylim(0.0, SPECTROGRAM_MAX_HZ)
    spectrogram_axis.set_ylabel("Hz")
    spectrogram_axis.tick_params(labelbottom=False)

    # Pitch against time on a logarithmic axis, so an octave is the same
    # height everywhere.
    reference_track = tracks[reference_ceiling_hz]
    voiced_outside_speech = (
        reference_track.voiced
        & ~reference_track.in_speech
    )

    pitch_axis.scatter(
        reference_track.times_s[voiced_outside_speech],
        reference_track.f0_hz[voiced_outside_speech],
        s=4,
        color="lightgrey",
        linewidths=0,
        label="voiced, outside speech",
    )

    for ceiling_hz, track in tracks.items():
        pitch_axis.scatter(
            track.times_s[track.used],
            track.f0_hz[track.used],
            s=5,
            color=colours[ceiling_hz],
            linewidths=0,
            zorder=draw_order[ceiling_hz],
            label=f"used, {ceiling_hz:g} Hz ceiling",
        )

        pitch_axis.axhline(
            ceiling_hz,
            color=colours[ceiling_hz],
            linestyle=":",
            linewidth=0.9,
        )

    pitch_axis.axhline(
        pitch_floor_hz,
        color="black",
        linestyle=":",
        linewidth=0.9,
    )

    reference = measurements[reference_ceiling_hz]

    for column, style in [
        ("f0_p10_hz", "--"),
        ("f0_median_hz", "-"),
        ("f0_p90_hz", "--"),
    ]:
        pitch_axis.axhline(
            reference[column],
            color="black",
            linestyle=style,
            linewidth=0.7,
        )

    pitch_axis.set_yscale("log")
    pitch_axis.set_ylim(
        0.9 * pitch_floor_hz,
        1.1 * max(tracks),
    )
    pitch_axis.set_yticks(PITCH_AXIS_TICKS_HZ)
    pitch_axis.yaxis.set_major_formatter(ScalarFormatter())
    pitch_axis.minorticks_off()
    pitch_axis.set_ylabel(
        f"F0 (Hz)\nlines: P10, median, P90 at {reference_ceiling_hz:g} Hz"
    )
    pitch_axis.tick_params(labelbottom=False)
    pitch_axis.legend(
        loc="upper right",
        fontsize=7,
        ncol=len(tracks) + 1,
        markerscale=2,
    )

    # Histogram of the used pitch values. Octave errors appear as a second
    # cluster an octave above or below the main one.
    n_histogram_bins = int(
        np.ceil(
            _semitones(max(tracks), pitch_floor_hz)
            / HISTOGRAM_BIN_ST
        )
    )

    histogram_bins = pitch_floor_hz * 2 ** (
        np.arange(n_histogram_bins + 1)
        * HISTOGRAM_BIN_ST
        / SEMITONES_PER_OCTAVE
    )

    for ceiling_hz, track in tracks.items():
        histogram_axis.hist(
            track.f0_hz[track.used],
            bins=histogram_bins,
            orientation="horizontal",
            histtype="step",
            color=colours[ceiling_hz],
            linewidth=0.9,
            zorder=draw_order[ceiling_hz],
        )

    histogram_axis.tick_params(
        labelleft=False,
        labelbottom=False,
    )
    histogram_axis.set_xlabel("frames")

    # Speech mask: shows why a voiced frame was or was not used.
    mask_axis.broken_barh(
        [
            (start_s, end_s - start_s)
            for start_s, end_s in intervals
        ],
        (0.0, 1.0),
        color="black",
    )
    mask_axis.set_yticks([])
    mask_axis.set_ylabel("speech")
    mask_axis.set_xlim(0.0, duration_s)
    mask_axis.set_xlabel("Time (s)")

    title_lines = [
        f"{row.sample_id} | {row.dataset} | {row.accent_group} | {row.sex}"
        f" | selected: {row.selection_reason}"
    ]

    for ceiling_hz in tracks:
        values = measurements[ceiling_hz]

        title_lines.append(
            f"{ceiling_hz:g} Hz: median {values['f0_median_hz']:.1f} Hz, "
            f"spread -{values['lower_spread_st']:.1f} / "
            f"+{values['upper_spread_st']:.1f} st, "
            f"voiced {values['voiced_duration_s']:.1f} s, "
            f"large jumps {100 * values['large_jump_fraction']:.1f}%, "
            f"near ceiling {100 * values['near_ceiling_fraction']:.1f}%"
        )

    figure.suptitle(
        "\n".join(title_lines),
        fontsize=9,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=110,
    )
    plt.close(figure)


def plot_recording(
    row,
    summary: pd.DataFrame,
    pitch_config: dict,
    speech_mask_config: dict,
    ceilings: list[float],
    reference_ceiling_hz: float,
    output_path: Path,
) -> None:
    """Recompute one recording's pitch tracks and save its plot."""

    audio = load_processed_audio(
        PROCESSED_ROOT,
        row.processed_filepath,
    )

    # Tracks are recomputed here rather than kept in memory for every
    # recording during the table pass. Pitch extraction is deterministic, so
    # these are the same tracks the table was measured from.
    tracks = {
        ceiling_hz: extract_pitch_track(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            pitch_config=dict(
                pitch_config,
                pitch_ceiling_hz=ceiling_hz,
            ),
            speech_mask_config=speech_mask_config,
        )
        for ceiling_hz in ceilings
    }

    intervals = speech_intervals(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        config=speech_mask_config,
    )

    rows = summary[summary["sample_id"] == row.sample_id]

    measurements = {
        measured.pitch_ceiling_hz: measured._asdict()
        for measured in rows.itertuples(index=False)
    }

    save_inspection_plot(
        row=row,
        audio=audio,
        tracks=tracks,
        intervals=intervals,
        measurements=measurements,
        pitch_floor_hz=pitch_config["pitch_floor_hz"],
        reference_ceiling_hz=reference_ceiling_hz,
        output_path=output_path,
    )


def build_inspection_sheet(
    selected: pd.DataFrame,
    ceilings: list[float],
) -> pd.DataFrame:
    """Return the sheet in which inspection verdicts are recorded by hand."""

    sheet = selected[
        [
            "sample_id",
            "dataset",
            "accent_group",
            "sex",
            "selection_reason",
            "processed_filepath",
        ]
    ].copy()

    sheet["plot"] = "plots/" + sheet["sample_id"] + ".png"

    # Filled in by hand: ok / suspicious / bad / unclear.
    for ceiling_hz in ceilings:
        sheet[f"tracking_{ceiling_hz:g}"] = ""

    sheet["preferred_ceiling"] = ""
    sheet["notes"] = ""

    return sheet


def print_overview(
    summary: pd.DataFrame,
    min_voiced_s: float,
) -> None:
    """Print how the tracking diagnostics are distributed at each ceiling."""

    ok = summary[
        (summary["status"] == "ok")
        & summary["in_analysis_manifest"]
    ]

    rows = []

    for ceiling_hz, group in ok.groupby(
        "pitch_ceiling_hz",
        sort=False,
    ):
        rows.append(
            {
                "ceiling_hz": ceiling_hz,
                "recordings": len(group),
                "below_min_voiced": int(
                    (~group["passes_min_voiced"].astype(bool)).sum()
                ),
                "least_voiced_s": group["voiced_duration_s"].min(),
                "upper_spread_p90": group["upper_spread_st"].quantile(0.9),
                "upper_spread_max": group["upper_spread_st"].max(),
                "lower_spread_p90": group["lower_spread_st"].quantile(0.9),
                "lower_spread_max": group["lower_spread_st"].max(),
                "large_jump_pct_p90": (
                    100 * group["large_jump_fraction"].quantile(0.9)
                ),
                "near_ceiling_pct_max": (
                    100 * group["near_ceiling_fraction"].max()
                ),
                "voiced_change_pct_median": (
                    group["voiced_frames_change_pct"].median()
                ),
            }
        )

    print(
        "\nAnalysis-manifest recordings by ceiling "
        f"(min_voiced_s = {min_voiced_s:g}):"
    )
    print(
        pd.DataFrame(rows)
        .round(2)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect pitch tracking and compare candidate pitch ceilings."
        )
    )

    parser.add_argument(
        "--run-name",
        default="ceiling_comparison",
        help=(
            "Folder name prefix under results/dsp/pitch_validation/. "
            "Default: ceiling_comparison."
        ),
    )

    parser.add_argument(
        "--ceilings",
        nargs="+",
        type=float,
        default=list(DEFAULT_CEILINGS_HZ),
        help="Candidate pitch_ceiling_hz values. Default: 600 500 400.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and replace run_name_01.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the table, selection and sheet but no plots.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the pitch inspection."""

    args = parse_args()

    pitch_config = load_json(PITCH_CONFIG_PATH)
    speech_mask_config = load_json(SPEECH_MASK_CONFIG_PATH)

    # The ceiling in pitch.json is the reference every candidate is compared
    # with, so it is always included.
    reference_ceiling_hz = float(pitch_config["pitch_ceiling_hz"])

    ceilings = list(
        dict.fromkeys(
            [reference_ceiling_hz]
            + [float(ceiling_hz) for ceiling_hz in args.ceilings]
        )
    )

    for ceiling_hz in ceilings:
        if ceiling_hz <= pitch_config["pitch_floor_hz"]:
            raise ValueError(
                f"Candidate ceiling {ceiling_hz:g} Hz is not above the "
                f"pitch floor of {pitch_config['pitch_floor_hz']:g} Hz."
            )

    valid_qc = load_valid_qc()
    extras = load_manual_extras()

    recordings = load_recordings(
        valid_qc=valid_qc,
        extras=extras,
    )

    run_root = prepare_run_directory(
        run_name=args.run_name,
        overwrite=args.overwrite,
    )

    with (
        run_root
        / "run_settings.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "pitch_feature_version": FEATURE_VERSION,
                "pitch_config": pitch_config,
                "speech_mask_config": speech_mask_config,
                "input_sha256": {
                    path.name: file_sha256(path)
                    for path in [
                        ANALYSIS_MANIFEST_PATH,
                        DEV_MANIFEST_PATH,
                        QC_MANIFEST_PATH,
                        PITCH_CONFIG_PATH,
                        SPEECH_MASK_CONFIG_PATH,
                    ]
                },
                "reference_ceiling_hz": reference_ceiling_hz,
                "ceilings_hz": ceilings,
                "large_jump_st": LARGE_JUMP_ST,
                "near_boundary_st": NEAR_BOUNDARY_ST,
                "n_stress_per_dataset": N_STRESS_PER_DATASET,
                "n_suspicious_per_diagnostic": N_SUSPICIOUS_PER_DIAGNOSTIC,
                "n_random_controls_per_dataset": N_RANDOM_CONTROLS_PER_DATASET,
                "random_seed": RANDOM_SEED,
                "write_plots": not args.no_plots,
            },
            file,
            indent=2,
        )

    print(
        f"Measuring {len(recordings)} recordings | "
        f"ceilings {ceilings} Hz | reference {reference_ceiling_hz:g} Hz"
    )

    summary_rows = []

    for number, row in enumerate(
        recordings.itertuples(index=False),
        start=1,
    ):
        print(f"[{number}/{len(recordings)}] {row.sample_id}")

        summary_rows.extend(
            measure_recording(
                row=row,
                pitch_config=pitch_config,
                speech_mask_config=speech_mask_config,
                ceilings=ceilings,
            )
        )

    summary = add_ceiling_comparison(
        summary=pd.DataFrame(summary_rows),
        reference_ceiling_hz=reference_ceiling_hz,
    )

    summary.to_csv(
        run_root / "summary.csv",
        index=False,
    )

    selected = select_plot_rows(
        recordings=recordings,
        summary=summary,
        extras=extras,
        reference_ceiling_hz=reference_ceiling_hz,
    )

    # Record exactly which recordings were chosen for inspection and why.
    selected.to_csv(
        run_root / "selection.csv",
        index=False,
    )

    build_inspection_sheet(
        selected=selected,
        ceilings=ceilings,
    ).to_csv(
        run_root / "inspection_sheet.csv",
        index=False,
    )

    n_plot_errors = 0

    if not args.no_plots:

        print(f"\nPlotting {len(selected)} recordings")

        for number, row in enumerate(
            selected.itertuples(index=False),
            start=1,
        ):
            print(f"[{number}/{len(selected)}] {row.sample_id}")

            # Plotting fails for a recording whose pitch track cannot be
            # extracted, for example one with too little speech. That
            # recording is already an error row in summary.csv.
            try:
                plot_recording(
                    row=row,
                    summary=summary,
                    pitch_config=pitch_config,
                    speech_mask_config=speech_mask_config,
                    ceilings=ceilings,
                    reference_ceiling_hz=reference_ceiling_hz,
                    output_path=(
                        run_root
                        / "plots"
                        / f"{row.sample_id}.png"
                    ),
                )

            except Exception as exc:
                n_plot_errors += 1
                print(f"\tNot plotted | {type(exc).__name__}: {exc}")

    print_overview(
        summary=summary,
        min_voiced_s=pitch_config["min_voiced_s"],
    )

    n_error_rows = int((summary["status"] == "error").sum())

    print(f"\nRows with errors in summary.csv: {n_error_rows}")
    print(f"Recordings selected for inspection: {len(selected)}")

    if not args.no_plots:
        print(f"Recordings not plotted: {n_plot_errors}")

    print(f"Wrote results to: {run_root}")


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.dsp.inspect_pitch
# python -m scripts.dsp.inspect_pitch --no-plots
# python -m scripts.dsp.inspect_pitch --ceilings 600 500
# python -m scripts.dsp.inspect_pitch --run-name ceiling_check --overwrite