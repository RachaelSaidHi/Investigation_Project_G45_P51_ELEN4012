"""Inspect Praat Burg formants on real recordings before freezing settings.

Compares five candidate ceilings and two frame-selection policies over the
analysis manifest. Both policies use the production formant track:
- all_voiced: every speech-active, voiced frame with valid F1-F3;
- loud_half: the loudest half of speech-active voiced frames, intersected
  with the same formant-validity mask. Loudness is not a vowel detector.

Records coverage, ceiling sensitivity, bandwidth/jump diagnostics and the
sensitivity of summaries to shorter contiguous recording segments. None of
these diagnostics automatically excludes a sample or selects a ceiling.
WER is not loaded and no accent-group result is used to tune parameters.

Writes summary.csv, duration_windows.csv, four overview CSVs, selection.csv,
inspection_sheet.csv, manual_subset.csv, plots and run_settings.json under
results/dsp/formant_validation/. The complete analysis manifest is measured;
only selected recordings are plotted. All results retain sample IDs.

Run from project root:
    python -m scripts.dsp.inspect_formants --dev-only
    python -m scripts.dsp.inspect_formants
    python -m scripts.dsp.inspect_formants --no-plots
    python -m scripts.dsp.inspect_formants --ceilings 5000 5250 5500
    python -m scripts.dsp.inspect_formants --run-name ceiling_review --overwrite

Options:
- --dev-only checks dev_manifest.csv before the full analysis sweep.
- --ceilings changes candidates; the current formants.json ceiling is included.
- --no-plots skips rendering, but still writes the inspection selections.
- --overwrite replaces run_name_01; otherwise the next available number is used.
"""

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.common.audio_io import SAMPLE_RATE, load_processed_audio
from src.common.run_management import file_sha256
from src.dsp.features.formants import (
    FEATURE_VERSION,
    InsufficientFormantFramesError,
    extract_formant_track,
    summarise_formants,
)
from src.dsp.features.pitch import extract_pitch_track


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/analysis_manifest.csv"
DEV_MANIFEST_PATH = PROJECT_ROOT / "data/manifests/dev_manifest.csv"
QC_MANIFEST_PATH = PROJECT_ROOT / "data/processed/audio_qc_study_manifest.csv"
PROCESSED_ROOT = PROJECT_ROOT / "data/processed"
FORMANT_CONFIG_PATH = PROJECT_ROOT / "configs/dsp/formants.json"
SPEECH_MASK_CONFIG_PATH = PROJECT_ROOT / "configs/dsp/speech_mask.json"
MANUAL_EXTRAS_PATH = PROJECT_ROOT / "configs/dsp/formant_inspection_extras.csv"
RESULTS_ROOT = PROJECT_ROOT / "results/dsp/formant_validation"

DEFAULT_CEILINGS_HZ = (4500.0, 5000.0, 5250.0, 5500.0, 6000.0)
POLICIES = ("all_voiced", "loud_half")
WINDOW_LENGTHS_S = (2.0, 5.0, 10.0)
WINDOW_POSITIONS = ("start", "middle", "end")
STABILITY_MIN_FULL_USABLE_S = 4.0
COVERAGE_MINIMA_S = (1.0, 2.0, 3.0, 5.0)

# Diagnostic-only thresholds. These flag tracks to inspect; they do not
# define acceptable formants or automatically determine the best ceiling.
LARGE_JUMP_FRACTION = 0.20  # >20% change between adjacent used frames
WIDE_BANDWIDTH_RATIO = 0.30  # bandwidth >30% of the formant frequency

N_STRESS_PER_DATASET = 2
N_SUSPICIOUS_PER_METRIC = 2
N_RANDOM_PER_DATASET = 4
RANDOM_SEED = 0

# Frame energy for the loud-half policy uses a 25 ms centred window.
# Frames are ranked before formant-validity filtering to avoid selecting
# a different loudness population solely because of the Burg ceiling.
ENERGY_WINDOW_S = 0.025
# Wideband spectrogram: a short analysis window shows broad vowel
# resonances rather than the closely spaced pitch harmonics. 6 ms is short
# enough to blur the harmonics of high voices too; at 10 ms a voice above
# about 200 Hz shows its harmonics as stripes that can be mistaken for
# formant bands.
SPECTROGRAM_N_FFT = 512
SPECTROGRAM_WINDOW = 96
# The picture is drawn in steps of this size. The bands stay as clear and
# the saved plots are about half the size.
SPECTROGRAM_STEP_DB = 4.0
SPECTROGRAM_HOP = 80
SPECTROGRAM_MAX_HZ = 6000
SPECTROGRAM_PREEMPHASIS = 0.97
HISTOGRAM_BIN_HZ = 50.0
FORMANT_COLOURS = ("#d95f02", "#1b9e77", "#7570b3")

SUMMARY_FEATURES = [
    f"f{number}_{stat}_hz"
    for number in (1, 2, 3)
    for stat in ("median", "p10", "p90")
]

# Short-stretch deviations from the full recording, one per summary feature.
STABILITY_SHIFT_COLUMNS = [
    f"f{number}{suffix}_abs_shift_st"
    for number in (1, 2, 3)
    for suffix in ("", "_p10", "_p90")
]


def load_json(path: Path) -> dict:
    """Load a configuration from JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def prepare_run_directory(run_name: str, overwrite: bool) -> Path:
    """Create a numbered results folder without overwriting earlier work."""

    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a non-empty folder name, not a path.")

    if overwrite:
        output = RESULTS_ROOT / f"{run_name}_01"
        if output.exists():
            shutil.rmtree(output)
    else:
        number = 1
        while (RESULTS_ROOT / f"{run_name}_{number:02d}").exists():
            number += 1
        output = RESULTS_ROOT / f"{run_name}_{number:02d}"

    output.mkdir(parents=True, exist_ok=False)
    return output


def load_recordings(dev_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load analysis or dev recordings, plus optional inspection extras.

    Extras (sample_id, reason) may add previously QC-passing recordings.
    Only sample metadata is used here; no ASR or WER files are read.
    """

    analysis = pd.read_csv(ANALYSIS_MANIFEST_PATH)
    dev = pd.read_csv(DEV_MANIFEST_PATH)
    required = {"sample_id", "dataset", "processed_filepath"}
    for name, table in (("analysis", analysis), ("dev", dev)):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} manifest is missing columns: {sorted(missing)}")
        if table["sample_id"].isna().any() or table["sample_id"].duplicated().any():
            raise ValueError(f"{name} manifest contains missing or repeated sample IDs.")

    extras = pd.DataFrame(columns=["sample_id", "reason"])
    if MANUAL_EXTRAS_PATH.exists() and not dev_only:
        extras = pd.read_csv(MANUAL_EXTRAS_PATH)
        missing = {"sample_id", "reason"} - set(extras.columns)
        if missing:
            raise ValueError(f"Formant extras are missing columns: {sorted(missing)}")
        if extras["sample_id"].isna().any() or extras["sample_id"].duplicated().any():
            raise ValueError("Formant extras contain missing or repeated sample IDs.")

    qc = pd.read_csv(QC_MANIFEST_PATH)
    if "sample_id" not in qc or qc["sample_id"].isna().any() or qc["sample_id"].duplicated().any():
        raise ValueError("QC data must have unique sample_id values.")

    # The manifests are authoritative; QC is joined only for stress selection.
    records = dev.copy() if dev_only else analysis.copy()
    records["in_analysis_manifest"] = records["sample_id"].isin(analysis["sample_id"])
    records["in_dev_manifest"] = records["sample_id"].isin(dev["sample_id"])

    if not dev_only:
        missing_dev = dev[~dev["sample_id"].isin(records["sample_id"])].copy()
        if not missing_dev.empty:
            missing_dev["in_analysis_manifest"] = False
            missing_dev["in_dev_manifest"] = True
            records = pd.concat([records, missing_dev], ignore_index=True)

    extra_ids = set(extras["sample_id"]) - set(records["sample_id"])
    if extra_ids:
        required_qc = {"sample_id", "dataset", "processed_filepath"}
        if required_qc - set(qc.columns):
            raise ValueError("QC table cannot supply the paths for manual extras.")
        extra_rows = qc[qc["sample_id"].isin(extra_ids)].copy()
        if set(extra_rows["sample_id"]) != extra_ids:
            raise ValueError("Some manually requested samples are absent from QC.")
        if "exclude" not in extra_rows or "error" not in extra_rows:
            raise ValueError("QC must contain exclude and error for manual extras.")
        excluded = extra_rows["exclude"].astype(str).str.lower().eq("true")
        failed = extra_rows["error"].notna() & extra_rows["error"].astype(str).str.strip().ne("")
        if excluded.any() or failed.any():
            raise ValueError("Manual extras must have passed audio QC.")
        extra_rows["in_analysis_manifest"] = False
        extra_rows["in_dev_manifest"] = False
        records = pd.concat([records, extra_rows], ignore_index=True)

    if records["processed_filepath"].isna().any():
        raise ValueError("A selected recording has no processed_filepath.")

    qc_cols = [col for col in ("wada_snr_db", "floor_gap_db", "processed_duration_s") if col in qc]
    # Drop matching manifest fields before joining QC to avoid _x/_y suffixes.
    records = records.drop(columns=qc_cols, errors="ignore")
    records = records.merge(
        qc[["sample_id", *qc_cols]],
        on="sample_id", how="left", validate="one_to_one",
    )
    for column in ("wada_snr_db", "floor_gap_db", "processed_duration_s"):
        if column not in records:
            records[column] = np.nan
    for column in ("accent_group", "sex"):
        if column not in records:
            records[column] = ""

    if records["sample_id"].duplicated().any():
        raise ValueError("Selected recordings contain repeated sample IDs.")

    records = records.sort_values(["dataset", "sample_id"]).reset_index(drop=True)
    return records, extras


def frame_energy(audio: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Return centred 25 ms RMS energy at each formant frame time.

    Cumulative squared samples give constant-time energy queries. The
    intervals are clipped at the recording boundaries. No normalisation
    or modification of the audio is performed.
    """

    width = max(1, round(ENERGY_WINDOW_S * SAMPLE_RATE))
    centre = np.rint(times_s * SAMPLE_RATE).astype(int)
    first = np.clip(centre - width // 2, 0, len(audio))
    last = np.clip(centre - width // 2 + width, 0, len(audio))

    squares = np.square(np.asarray(audio, dtype=np.float64))
    cumulative = np.concatenate(([0.0], np.cumsum(squares)))
    power = np.maximum(cumulative[last] - cumulative[first], 0.0)
    return np.sqrt(power / np.maximum(last - first, 1))


def policy_mask(track, policy: str, energy: np.ndarray) -> np.ndarray:
    """Return usable frames for all-voiced or loud-half selection.

    The loud half is ranked among speech-active voiced frames, before any
    formant-validity check. Stable sorting resolves equal-energy ties by
    time. Each policy uses identical Burg estimates for a given ceiling.
    """

    if policy == "all_voiced":
        return track.used.copy()
    if policy != "loud_half":
        raise ValueError(f"Unknown formant selection policy: {policy}")

    candidate_indices = np.flatnonzero(track.in_speech & track.voiced)
    selected = np.zeros(len(track.times_s), dtype=bool)
    if len(candidate_indices) == 0:
        return selected

    ranked = np.argsort(-energy[candidate_indices], kind="stable")
    keep_count = (len(candidate_indices) + 1) // 2
    selected[candidate_indices[ranked[:keep_count]]] = True
    return selected & track.valid_formants


def measure_track(track, selected: np.ndarray, formant_config: dict) -> dict:
    """Return production formant summaries plus inspection-only diagnostics.

    An insufficient-duration summary is also calculated with the minimum
    switched off, if any frames remain. It stays flagged as insufficient;
    the relaxed values are for inspection, not final feature extraction.
    """

    selected_track = replace(track, used=selected)
    speech_voiced = track.in_speech & track.voiced
    n_speech = int(track.in_speech.sum())
    n_voiced = int(speech_voiced.sum())
    n_used = int(selected.sum())

    n_valid_voiced = int((speech_voiced & track.valid_formants).sum())
    n_invalid_voiced = n_voiced - n_valid_voiced
    disordered = (
        speech_voiced
        & np.isfinite(track.formants_hz).all(axis=1)
        & ~track.valid_formants
    )
    result = {
        "n_speech_frames": n_speech,
        "n_voiced_frames": n_voiced,
        "n_valid_voiced_frames": n_valid_voiced,
        "n_usable_formant_frames": n_used,
        "usable_duration_s": n_used * track.time_step_s,
        "voiced_valid_fraction": n_valid_voiced / n_voiced if n_voiced else np.nan,
        "n_invalid_voiced_frames": n_invalid_voiced,
        "n_disordered_frames": int(disordered.sum()),
        "invalid_voiced_fraction": n_invalid_voiced / n_voiced if n_voiced else np.nan,
        "passes_minimum": False,
        "status": "insufficient",
        "error": "",
    }
    result.update({name: np.nan for name in SUMMARY_FEATURES})
    result["n_formant_frames"] = n_used
    result["formant_frame_fraction"] = float(n_used / n_speech) if n_speech else np.nan

    try:
        features = summarise_formants(selected_track, formant_config)
        result["passes_minimum"] = True
        result["status"] = "ok"
    except InsufficientFormantFramesError as exc:
        result["error"] = str(exc)
        if n_used == 0 or n_speech == 0:
            return result
        relaxed = dict(formant_config, min_usable_formant_s=0.0)
        features = summarise_formants(selected_track, relaxed)

    result.update(features)
    for index in range(1, 4):
        label = f"f{index}"
        with np.errstate(divide="ignore", invalid="ignore"):
            result[f"{label}_upper_spread_st"] = float(
                12 * np.log2(result[f"{label}_p90_hz"] / result[f"{label}_median_hz"])
            )
            result[f"{label}_lower_spread_st"] = float(
                12 * np.log2(result[f"{label}_median_hz"] / result[f"{label}_p10_hz"])
            )

    values = track.formants_hz[selected]
    bandwidths = track.bandwidths_hz[selected]

    # Adjacent accepted frames only: frames across a pause are not a jump.
    pairs = selected[:-1] & selected[1:]
    result["n_adjacent_frame_pairs"] = int(pairs.sum())
    for index in range(3):
        label = f"f{index + 1}"
        frequency = values[:, index]
        bandwidth = bandwidths[:, index]
        result[f"{label}_bandwidth_median_hz"] = (
            float(np.nanmedian(bandwidth)) if np.isfinite(bandwidth).any() else np.nan
        )
        finite_bandwidth = np.isfinite(bandwidth) & (bandwidth > 0)
        result[f"{label}_bandwidth_defined_fraction"] = float(np.mean(finite_bandwidth))
        result[f"{label}_wide_bandwidth_fraction"] = float(np.mean(
            bandwidth[finite_bandwidth] > WIDE_BANDWIDTH_RATIO * frequency[finite_bandwidth]
        )) if finite_bandwidth.any() else np.nan
        if np.any(pairs):
            previous = track.formants_hz[:-1, index][pairs]
            following = track.formants_hz[1:, index][pairs]
            result[f"{label}_jump_fraction"] = float(np.mean(
                np.abs(following - previous) / previous > LARGE_JUMP_FRACTION
            ))
        else:
            result[f"{label}_jump_fraction"] = np.nan

    return result


def measure_duration_windows(
    track, selected: np.ndarray, duration_s: float, result: dict,
) -> list[dict]:
    """Compare short contiguous time windows with the full-track medians.

    Windows describe sensitivity to the speech material available; they do
    not isolate Praat tracking error from differences between vowel sounds.
    """

    if result.get("usable_duration_s", 0) < STABILITY_MIN_FULL_USABLE_S:
        return []

    windows = []
    for length_s in WINDOW_LENGTHS_S:
        if length_s > duration_s / 2:
            continue
        starts = (0.0, (duration_s - length_s) / 2.0, duration_s - length_s)
        for position, start_s in zip(WINDOW_POSITIONS, starts):
            window_mask = (
                selected
                & (track.times_s >= start_s)
                & (track.times_s < start_s + length_s)
            )
            values = {"window_length_s": length_s, "position": position, "start_s": start_s,
                      "n_usable_formant_frames": int(window_mask.sum()),
                      "window_usable_s": float(window_mask.sum() * track.time_step_s)}
            # All nine production values are compared, not only the medians:
            # P10 and P90 are also output columns, and percentiles away from
            # the middle need more frames to settle.
            if window_mask.any():
                window_p10, window_median, window_p90 = np.percentile(
                    track.formants_hz[window_mask], [10, 50, 90], axis=0
                )
            else:
                window_p10 = window_median = window_p90 = np.full(3, np.nan)

            for index in range(3):
                label = f"f{index + 1}"
                for stat, window_values in (
                    ("median", window_median), ("p10", window_p10), ("p90", window_p90),
                ):
                    baseline = result.get(f"{label}_{stat}_hz", np.nan)
                    window_value = float(window_values[index])
                    prefix = label if stat == "median" else f"{label}_{stat}"
                    if stat == "median":
                        values[f"{label}_median_hz"] = window_value
                        values[f"{label}_abs_shift_hz"] = abs(window_value - baseline)
                    if np.isfinite(window_value) and np.isfinite(baseline) and baseline > 0:
                        values[f"{prefix}_abs_shift_st"] = abs(12 * np.log2(window_value / baseline))
                    else:
                        values[f"{prefix}_abs_shift_st"] = np.nan
            windows.append(values)
    return windows


def measure_recording(row, formant_config: dict, speech_mask_config: dict, ceilings: list[float]):
    """Measure one processed recording at each ceiling and both policies."""

    try:
        audio = load_processed_audio(PROCESSED_ROOT, row.processed_filepath)
        load_error = ""
    except Exception as exc:
        audio = None
        load_error = f"{type(exc).__name__}: {exc}"

    # The production extractor constructs this pitch track internally for
    # each ceiling. One extra call exposes its grid and the voice range for QC.
    pitch_track = None
    if audio is not None:
        try:
            pitch_track = extract_pitch_track(
                audio=audio,
                sample_rate=SAMPLE_RATE,
                pitch_config=formant_config["voicing_pitch"],
                speech_mask_config=speech_mask_config,
            )
        except Exception:
            pass  # Each ceiling still reports its own production extraction error.

    if pitch_track is not None and pitch_track.used.any():
        f0_median_hz = float(np.median(pitch_track.f0_hz[pitch_track.used]))
    else:
        f0_median_hz = np.nan

    results, windows = [], []
    for ceiling in ceilings:
        candidate = dict(formant_config, formant_ceiling_hz=ceiling)
        base = {"sample_id": row.sample_id, "dataset": row.dataset,
                "accent_group": row.accent_group, "in_analysis_manifest": row.in_analysis_manifest,
                "in_dev_manifest": row.in_dev_manifest, "formant_ceiling_hz": ceiling}

        if audio is None:
            for policy in POLICIES:
                results.append({**base, "policy": policy, "status": "error", "error": load_error})
            continue

        try:
            track = extract_formant_track(
                audio=audio, sample_rate=SAMPLE_RATE, formant_config=candidate,
                speech_mask_config=speech_mask_config,
            )
            energy = frame_energy(audio, track.times_s)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for policy in POLICIES:
                results.append({**base, "policy": policy, "status": "error", "error": error})
            continue

        for policy in POLICIES:
            try:
                selected = policy_mask(track, policy, energy)
                measured = measure_track(track, selected, candidate)
                if pitch_track is not None and len(pitch_track.times_s):
                    offset = (
                        (track.times_s[0] - pitch_track.times_s[0])
                        / pitch_track.time_step_s
                    ) % 1.0
                    grid_offset = float(round(min(offset, 1.0 - offset), 3))
                else:
                    grid_offset = np.nan

                results.append({
                    **base, "policy": policy, "f0_median_hz": f0_median_hz,
                    "voicing_grid_offset_steps": grid_offset, **measured,
                })

                for window in measure_duration_windows(
                    track, selected, len(audio) / SAMPLE_RATE, measured
                ):
                    windows.append({**base, "policy": policy, **window})
            except Exception as exc:
                results.append({**base, "policy": policy, "status": "error",
                                "error": f"{type(exc).__name__}: {exc}"})

    return results, windows


def add_comparisons(summary: pd.DataFrame, reference_ceiling: float) -> pd.DataFrame:
    """Add paired ceiling and policy differences without selecting a winner."""

    summary = summary.copy()
    reference = summary[
        (summary["formant_ceiling_hz"] == reference_ceiling)
        & summary["status"].isin(["ok", "insufficient"])
    ].set_index(["sample_id", "policy"])
    all_voiced = summary[
        (summary["policy"] == "all_voiced")
        & summary["status"].isin(["ok", "insufficient"])
    ].set_index(["sample_id", "formant_ceiling_hz"])

    reference_index = pd.MultiIndex.from_frame(summary[["sample_id", "policy"]])
    all_index = pd.MultiIndex.from_frame(summary[["sample_id", "formant_ceiling_hz"]])

    for index in range(1, 4):
        name = f"f{index}_median_hz"
        baseline = reference[name].reindex(reference_index).to_numpy()
        other_policy = all_voiced[name].reindex(all_index).to_numpy()
        summary[f"f{index}_ceiling_shift_hz"] = summary[name] - baseline
        summary[f"f{index}_policy_shift_hz"] = summary[name] - other_policy
        with np.errstate(divide="ignore", invalid="ignore"):
            summary[f"f{index}_ceiling_shift_st"] = 12 * np.log2(summary[name] / baseline)
            summary[f"f{index}_policy_shift_st"] = 12 * np.log2(summary[name] / other_policy)

    sensitivity = (summary[[f"f{i}_ceiling_shift_hz" for i in (1, 2, 3)]]
                   .abs().max(axis=1).groupby(summary["sample_id"]).max())
    policy_change = (summary[[f"f{i}_policy_shift_hz" for i in (1, 2, 3)]]
                     .abs().max(axis=1).groupby(summary["sample_id"]).max())
    summary["ceiling_sensitivity_hz"] = summary["sample_id"].map(sensitivity)
    summary["policy_sensitivity_hz"] = summary["sample_id"].map(policy_change)
    ceiling_st = (
        summary[[f"f{i}_ceiling_shift_st" for i in (1, 2, 3)]]
        .abs().max(axis=1).groupby(summary["sample_id"]).max()
    )
    policy_st = (
        summary[[f"f{i}_policy_shift_st" for i in (1, 2, 3)]]
        .abs().max(axis=1).groupby(summary["sample_id"]).max()
    )
    summary["ceiling_sensitivity_st"] = summary["sample_id"].map(ceiling_st)
    summary["policy_sensitivity_st"] = summary["sample_id"].map(policy_st)

    # Low, middle and high voices, as thirds of the measured recordings.
    # Used only to break the ceiling comparison down, never to set anything.
    voice_pitch = summary.groupby("sample_id")["f0_median_hz"].first().dropna()
    if voice_pitch.nunique() >= 3:
        try:
            thirds = pd.qcut(voice_pitch, 3, labels=["low", "middle", "high"])
        except ValueError:
            # Tied F0 values can make quantile boundaries coincide.
            # Leave the optional pitch groups unset rather than invent a split.
            summary["voice_third"] = np.nan
        else:
            summary["voice_third"] = summary["sample_id"].map(thirds.astype(object))
    else:
        summary["voice_third"] = np.nan
    return summary


def select_inspection_rows(recordings: pd.DataFrame, summary: pd.DataFrame,
                           extras: pd.DataFrame, reference_ceiling: float) -> pd.DataFrame:
    """Select dev, low-quality, suspicious and random-control recordings.

    Ranking metrics are inspection aids only. Selection is deterministic;
    selecting a recording does not mean its estimated formants are wrong.
    """

    reasons: dict[str, list[str]] = {}

    def add(sample_id: str, reason: str):
        if reason not in reasons.setdefault(sample_id, []):
            reasons[sample_id].append(reason)

    for sample_id in recordings.loc[recordings["in_dev_manifest"], "sample_id"]:
        add(sample_id, "dev_manifest")

    candidates = recordings[recordings["in_analysis_manifest"] & ~recordings["in_dev_manifest"]]
    for _, group in candidates.groupby("dataset", sort=True):
        for column, label in (
            ("floor_gap_db", "lowest_floor_gap"),
            ("wada_snr_db", "lowest_wada_snr"),
            ("processed_duration_s", "shortest_duration"),
        ):
            ranked = group[group[column].notna()].sort_values([column, "sample_id"])
            for sample_id in ranked["sample_id"].head(N_STRESS_PER_DATASET):
                add(sample_id, label)

    measured = summary[(summary["formant_ceiling_hz"] == reference_ceiling)
                       & (summary["policy"] == "all_voiced")
                       & summary["status"].isin(["ok", "insufficient"])]
    metrics = (
        ("f0_median_hz", "lowest_voice", True),
        ("f0_median_hz", "highest_voice", False),
        ("ceiling_sensitivity_st", "ceiling_sensitive", False),
        ("policy_sensitivity_st", "frame_policy_sensitive", False),
        ("f1_upper_spread_st", "large_f1_spread", False),
        ("f3_upper_spread_st", "large_f3_spread", False),
        ("f3_lower_spread_st", "large_f3_lower_spread", False),
        ("invalid_voiced_fraction", "highest_invalid_formant_fraction", False),
        ("f3_jump_fraction", "f3_jumps", False),
        ("f3_wide_bandwidth_fraction", "f3_wide_bandwidths", False),
        ("usable_duration_s", "shortest_usable_duration", True),
    )
    for column, reason, ascending in metrics:
        if column not in measured:
            continue
        ranked = measured[measured[column].notna()].copy()
        if not ascending:
            ranked = ranked[ranked[column] > 0]
        ranked = ranked.sort_values([column, "sample_id"], ascending=[ascending, True])
        for sample_id in ranked["sample_id"].head(N_SUSPICIOUS_PER_METRIC):
            add(sample_id, reason)

    # Judged on the production selection only. The louder half has half the
    # frames, so it falls below the minimum in recordings that are fine.
    failed = (summary["status"] != "ok") & (summary["policy"] == "all_voiced")
    for sample_id in summary.loc[failed, "sample_id"].unique():
        add(sample_id, "measurement_failure_or_insufficient")

    rng = np.random.default_rng(RANDOM_SEED)
    for _, group in recordings[recordings["in_analysis_manifest"]].groupby("dataset", sort=True):
        remaining = sorted(set(group["sample_id"]) - set(reasons))
        selected_controls = rng.choice(
            remaining, min(N_RANDOM_PER_DATASET, len(remaining)), replace=False
        )
        for sample_id in selected_controls:
            add(str(sample_id), "random_control")

    for row in extras.itertuples(index=False):
        add(str(row.sample_id), f"manual: {row.reason}")

    selected = recordings[recordings["sample_id"].isin(reasons)].copy()
    selected["selection_reason"] = selected["sample_id"].map(lambda sid: "; ".join(reasons[sid]))
    return selected.sort_values(["dataset", "sample_id"]).reset_index(drop=True)


def select_manual_subset(
    selected: pd.DataFrame, summary: pd.DataFrame, reference_ceiling: float,
) -> pd.DataFrame:
    """Choose six recordings for focused vowel checks: four SAA, two AfriSpeech.

    Include one suspicious example and random controls in each corpus, when
    available. These are suggestions, not manually confirmed vowel labels.
    """

    rng = np.random.default_rng(RANDOM_SEED + 1)
    # Failed recordings are still listed on the general inspection sheet,
    # but are unsuitable for a focused vowel-tracking check.
    available = set(summary.loc[
        (summary["formant_ceiling_hz"] == reference_ceiling)
        & (summary["policy"] == "all_voiced")
        & (summary["status"] == "ok"),
        "sample_id",
    ])
    selected = selected[selected["sample_id"].isin(available)]
    chosen = []
    for dataset, n in (("saa", 4), ("afrispeech", 2)):
        group = selected[selected["dataset"] == dataset]
        if group.empty:
            continue
        hard = group[~group["selection_reason"].str.fullmatch("dev_manifest|random_control")]
        if not hard.empty:
            chosen.append(str(hard.sort_values("sample_id")["sample_id"].iloc[0]))
        remaining = sorted(set(group["sample_id"]) - set(chosen))
        n_remaining = max(0, n - int(not hard.empty))
        chosen.extend(map(str, rng.choice(
            remaining, min(n_remaining, len(remaining)), replace=False
        )))

    subset = selected[selected["sample_id"].isin(chosen)].copy()
    subset["vowel_region_times_s"] = ""
    subset["vowel_tracking_notes"] = ""
    return subset.reset_index(drop=True)


def build_inspection_sheet(selected: pd.DataFrame, ceilings: list[float]) -> pd.DataFrame:
    """Return an editable sheet for manual tracking and frame-policy notes."""

    sheet = selected[["sample_id", "dataset", "accent_group", "selection_reason",
                      "processed_filepath"]].copy()
    sheet["plot"] = "plots/" + sheet["sample_id"] + ".png"
    for ceiling in ceilings:
        sheet[f"tracking_{ceiling:g}_hz"] = ""  # ok / suspicious / bad / unclear
    sheet["preferred_ceiling_hz"] = ""
    sheet["frame_policy_observation"] = ""
    sheet["preferred_policy"] = ""
    sheet["notes"] = ""
    return sheet


def save_inspection_plot(
    row,
    audio: np.ndarray,
    tracks: dict,
    reference_ceiling: float,
    output_path: Path,
) -> None:
    """Plot each ceiling's F1-F3 tracks and value distributions.

    Faint dots represent all usable voiced frames; stronger dots represent
    the louder half. The selection strips correspond to the reference ceiling.
    Diagnostic plots are not an automatic formant-accuracy criterion.
    """

    duration_s = len(audio) / SAMPLE_RATE
    ceilings = list(tracks)
    emphasized = librosa.effects.preemphasis(
        np.asarray(audio, dtype=np.float32), coef=SPECTROGRAM_PREEMPHASIS
    )
    spectrum = librosa.stft(
        emphasized,
        n_fft=SPECTROGRAM_N_FFT,
        win_length=SPECTROGRAM_WINDOW,
        hop_length=SPECTROGRAM_HOP,
    )
    spectrogram = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
    max_bin = min(
        len(spectrogram),
        int(SPECTROGRAM_MAX_HZ * SPECTROGRAM_N_FFT / SAMPLE_RATE) + 1,
    )
    spectrogram = SPECTROGRAM_STEP_DB * np.round(
        np.maximum(spectrogram[:max_bin], -60.0) / SPECTROGRAM_STEP_DB
    )
    frequency_step = SAMPLE_RATE / SPECTROGRAM_N_FFT

    width = min(28.0, max(14.0, 0.45 * duration_s))
    figure = plt.figure(
        figsize=(width, 2.7 * len(ceilings) + 1.4),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        len(ceilings) + 1, 2,
        height_ratios=[3] * len(ceilings) + [0.8],
        width_ratios=[1, 0.12],
    )
    first_axis = None
    histogram_bins = np.arange(
        0.0, SPECTROGRAM_MAX_HZ + HISTOGRAM_BIN_HZ, HISTOGRAM_BIN_HZ
    )

    for index, ceiling in enumerate(ceilings):
        track = tracks[ceiling]
        energy = frame_energy(audio, track.times_s)
        all_frames = policy_mask(track, "all_voiced", energy)
        loud_frames = policy_mask(track, "loud_half", energy)

        axis = figure.add_subplot(grid[index, 0], sharex=first_axis)
        if first_axis is None:
            first_axis = axis

        axis.imshow(
            spectrogram,
            extent=(0.0, duration_s, -frequency_step / 2,
                    (max_bin - 0.5) * frequency_step),
            aspect="auto", origin="lower", cmap="Greys", vmin=-60, vmax=0,
            interpolation="nearest",
        )

        for formant_index, colour in enumerate(FORMANT_COLOURS):
            axis.scatter(
                track.times_s[all_frames],
                track.formants_hz[all_frames, formant_index],
                s=3, alpha=0.25, color=colour, rasterized=True,
            )
            axis.scatter(
                track.times_s[loud_frames],
                track.formants_hz[loud_frames, formant_index],
                s=6, alpha=0.9, color=colour, rasterized=True,
            )

        axis.set_title(
            f"Burg ceiling {ceiling:g} Hz | "
            f"all voiced {int(all_frames.sum())}, louder half {int(loud_frames.sum())}",
            loc="left", fontsize=9,
        )
        axis.set_ylim(0, SPECTROGRAM_MAX_HZ)
        axis.set_ylabel("Hz")
        axis.tick_params(labelbottom=False)

        histogram_axis = figure.add_subplot(grid[index, 1], sharey=axis)
        for formant_index, colour in enumerate(FORMANT_COLOURS):
            histogram_axis.hist(
                track.formants_hz[all_frames, formant_index],
                bins=histogram_bins,
                orientation="horizontal", histtype="step",
                color=colour, linewidth=1,
            )
        histogram_axis.tick_params(labelleft=False, labelbottom=False)
        histogram_axis.set_xlim(left=0)

    reference_track = tracks.get(reference_ceiling, next(iter(tracks.values())))
    reference_energy = frame_energy(audio, reference_track.times_s)
    mask_axis = figure.add_subplot(grid[len(ceilings), 0], sharex=first_axis)

    masks = (
        reference_track.in_speech,
        reference_track.in_speech & reference_track.voiced,
        policy_mask(reference_track, "all_voiced", reference_energy),
        policy_mask(reference_track, "loud_half", reference_energy),
    )
    for index, active in enumerate(masks):
        mask_axis.fill_between(
            reference_track.times_s, index, index + 0.8,
            where=active, step="mid", color=f"C{index}", linewidth=0,
        )

    mask_axis.set_yticks(
        np.arange(4) + 0.4,
        labels=("speech", "voiced", "all voiced", "louder half"),
    )
    mask_axis.set_ylim(0, 4)
    mask_axis.set_xlim(0, duration_s)
    mask_axis.set_xlabel("Time (s)")

    legend = [
        Line2D([0], [0], marker="o", linestyle="", color=colour,
               label=f"F{index + 1}")
        for index, colour in enumerate(FORMANT_COLOURS)
    ]
    figure.legend(handles=legend, loc="upper right", ncol=3, fontsize=9)
    # Full selection reasons are in selection.csv; they can be too long for a plot title.
    figure.suptitle(
        f"{row.sample_id} | {row.dataset} | {row.accent_group}\n"
        "faint = all voiced, bright = louder half",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110)
    plt.close(figure)


def plot_recording(row, formant_config: dict, speech_mask_config: dict,
                   ceilings: list[float], reference_ceiling: float, output_path: Path) -> None:
    """Recompute a selected recording's tracks for plotting only."""

    audio = load_processed_audio(PROCESSED_ROOT, row.processed_filepath)
    tracks = {}
    for ceiling in ceilings:
        try:
            tracks[ceiling] = extract_formant_track(
                audio=audio, sample_rate=SAMPLE_RATE,
                formant_config=dict(formant_config, formant_ceiling_hz=ceiling),
                speech_mask_config=speech_mask_config,
            )
        except Exception as exc:
            print(
                f"  {row.sample_id}: ceiling {ceiling:g} not plotted: "
                f"{type(exc).__name__}: {exc}"
            )

    if not tracks:
        raise ValueError("No ceiling produced a plottable formant track.")
    save_inspection_plot(row, audio, tracks, reference_ceiling, output_path)


def build_overviews(summary: pd.DataFrame, windows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create four descriptive tables for comparing settings and coverage.

    Analysis recordings only; manual extras cannot change the reported
    distributions. The tables flag candidates for inspection, not winners.
    """

    analysis = summary[summary["in_analysis_manifest"]]
    by_ceiling = []

    def describe_ceiling(group, ceiling, policy, voice_third):
        row = {
            "formant_ceiling_hz": ceiling,
            "policy": policy,
            "voice_third": voice_third,
            "n_recordings": len(group),
            "n_ok": int((group["status"] == "ok").sum()),
            "n_insufficient": int((group["status"] == "insufficient").sum()),
            "n_error": int((group["status"] == "error").sum()),
            "median_usable_s": group["usable_duration_s"].median(),
            "median_invalid_voiced_fraction": group["invalid_voiced_fraction"].median(),
            "n_disordered_frames": int(group["n_disordered_frames"].sum()),
            "n_midpoint_grids": int((group["voicing_grid_offset_steps"] > 0.25).sum()),
        }
        # All three formants: a ceiling that is too low tends to show first
        # in F2, one that is too high in F3.
        for number in (1, 2, 3):
            row[f"f{number}_median_hz"] = group[f"f{number}_median_hz"].median()
            row[f"f{number}_jump_pct_median"] = (
                100 * group[f"f{number}_jump_fraction"].median()
            )
            row[f"f{number}_jump_pct_p90"] = (
                100 * group[f"f{number}_jump_fraction"].quantile(0.9)
            )
            row[f"f{number}_ceiling_shift_st_p90"] = (
                group[f"f{number}_ceiling_shift_st"].abs().quantile(0.9)
            )
        row["f3_wide_bandwidth_pct_p90"] = (
            100 * group["f3_wide_bandwidth_fraction"].quantile(0.9)
        )
        return row

    for (ceiling, policy), group in analysis.groupby(
        ["formant_ceiling_hz", "policy"], sort=True
    ):
        by_ceiling.append(describe_ceiling(group, ceiling, policy, "all"))

    # One common ceiling has to work for low and high voices alike, so the
    # same figures are repeated for each third of the voice-pitch range.
    for (ceiling, policy, voice_third), group in analysis.groupby(
        ["formant_ceiling_hz", "policy", "voice_third"], sort=True
    ):
        by_ceiling.append(describe_ceiling(group, ceiling, policy, voice_third))

    selection = []
    valid = analysis[analysis["status"].isin(["ok", "insufficient"])]

    def describe_selection(group, ceiling, dataset, accent_group):
        # All nine production values, not only the medians: P10 and P90 are
        # output columns too, and are the ones a change of frames moves most.
        by_policy = group.pivot_table(
            index="sample_id", columns="policy", values=SUMMARY_FEATURES
        )
        for column in SUMMARY_FEATURES:
            if (column, "all_voiced") not in by_policy or (column, "loud_half") not in by_policy:
                continue
            shifts_st = 12 * np.log2(
                by_policy[(column, "loud_half")] / by_policy[(column, "all_voiced")]
            )
            selection.append({
                "formant_ceiling_hz": ceiling,
                "dataset": dataset,
                "accent_group": accent_group,
                "feature": column,
                "n_paired": int(shifts_st.notna().sum()),
                "median_shift_st": shifts_st.median(),
                "abs_shift_st_p90": shifts_st.abs().quantile(0.9),
            })

    for (ceiling, dataset), group in valid.groupby(
        ["formant_ceiling_hz", "dataset"], sort=True
    ):
        describe_selection(group, ceiling, dataset, "all")
        for accent_group, subgroup in group.groupby("accent_group", sort=True):
            describe_selection(subgroup, ceiling, dataset, accent_group)

    coverage = []

    def describe_coverage(group, ceiling, policy, dataset, accent_group):
        row = {
            "formant_ceiling_hz": ceiling,
            "policy": policy,
            "dataset": dataset,
            "accent_group": accent_group,
            "n_recordings": len(group),
            "n_error": int((group["status"] == "error").sum()),
            "median_usable_s": group["usable_duration_s"].median(),
        }
        for minimum in COVERAGE_MINIMA_S:
            row[f"n_below_{minimum:g}_s"] = int(
                (group["usable_duration_s"] < minimum).sum()
            )
        return row

    for (ceiling, policy, dataset), group in analysis.groupby(
        ["formant_ceiling_hz", "policy", "dataset"], sort=True
    ):
        coverage.append(describe_coverage(group, ceiling, policy, dataset, "all"))
        for accent_group, subgroup in group.groupby("accent_group", sort=True):
            coverage.append(
                describe_coverage(subgroup, ceiling, policy, dataset, accent_group)
            )

    stability = []
    if not windows.empty and "window_usable_s" in windows:
        sample_ids = set(analysis["sample_id"])
        selected_windows = windows[windows["sample_id"].isin(sample_ids)].copy()
        bins = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
        selected_windows["usable_s_bin"] = pd.cut(
            selected_windows["window_usable_s"], bins=bins, right=False
        ).astype(str)
        for (ceiling, policy, duration), group in selected_windows.groupby(
            ["formant_ceiling_hz", "policy", "usable_s_bin"], sort=True
        ):
            stability.append({
                "formant_ceiling_hz": ceiling,
                "policy": policy,
                "usable_s_bin": duration,
                "n_windows": len(group),
                "n_recordings": group["sample_id"].nunique(),
                **{
                    f"{name}_p90": group[name].quantile(0.9)
                    for name in STABILITY_SHIFT_COLUMNS
                },
            })

    # Explicit headers keep empty overview CSVs readable by pandas.
    stability_columns = [
        "formant_ceiling_hz", "policy", "usable_s_bin", "n_windows", "n_recordings",
        *(f"{name}_p90" for name in STABILITY_SHIFT_COLUMNS),
    ]
    selection_columns = [
        "formant_ceiling_hz", "dataset", "accent_group", "feature", "n_paired",
        "median_shift_st", "abs_shift_st_p90",
    ]
    return {
        "overview_by_ceiling.csv": pd.DataFrame(by_ceiling),
        "overview_selection.csv": pd.DataFrame(selection, columns=selection_columns),
        "overview_coverage.csv": pd.DataFrame(coverage),
        "overview_stability.csv": pd.DataFrame(stability, columns=stability_columns),
    }


def print_overview(summary: pd.DataFrame) -> None:
    """Display descriptive diagnostics without ranking the candidate ceilings."""

    analysis = summary[summary["in_analysis_manifest"]]
    rows = []
    for (ceiling, policy), group in analysis.groupby(["formant_ceiling_hz", "policy"], sort=True):
        rows.append({
            "ceiling_hz": ceiling,
            "policy": policy,
            "ok": int((group["status"] == "ok").sum()),
            "insufficient": int((group["status"] == "insufficient").sum()),
            "errors": int((group["status"] == "error").sum()),
            "median_usable_s": group["usable_duration_s"].median(),
            "median_valid_fraction": group["voiced_valid_fraction"].median(),
            "f3_jump_pct_p90": 100 * group["f3_jump_fraction"].quantile(0.9),
            "f3_wide_band_pct_p90": 100 * group["f3_wide_bandwidth_fraction"].quantile(0.9),
            "f3_ceiling_shift_p90_hz": group["f3_ceiling_shift_hz"].abs().quantile(0.9),
        })

    print("\nAnalysis-manifest overview:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse the inspection run options."""

    parser = argparse.ArgumentParser(
        description="Compare Praat formant ceilings and voiced-frame selection."
    )
    parser.add_argument("--run-name", default="ceiling_comparison", help="Results folder prefix.")
    parser.add_argument("--ceilings", nargs="+", type=float, default=list(DEFAULT_CEILINGS_HZ))
    parser.add_argument("--overwrite", action="store_true", help="Replace run_name_01.")
    parser.add_argument("--no-plots", action="store_true", help="Write tables and sheets only.")
    parser.add_argument(
        "--dev-only", action="store_true",
        help="Measure only dev-manifest recordings.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete formant-inspection sweep and save the results."""

    args = parse_args()
    formant_config = load_json(FORMANT_CONFIG_PATH)
    speech_mask_config = load_json(SPEECH_MASK_CONFIG_PATH)
    reference_ceiling = float(formant_config["formant_ceiling_hz"])
    # Ascending, so the plot panels and tables read from lowest to highest.
    ceilings = sorted({reference_ceiling, *(float(value) for value in args.ceilings)})
    if any(not np.isfinite(value) or value <= 0 or value >= SAMPLE_RATE / 2 for value in ceilings):
        raise ValueError(
            f"Formant ceilings must be finite and between 0 and {SAMPLE_RATE / 2:g} Hz."
        )

    recordings, extras = load_recordings(dev_only=args.dev_only)
    output = prepare_run_directory(args.run_name, args.overwrite)
    paths = [ANALYSIS_MANIFEST_PATH, DEV_MANIFEST_PATH, QC_MANIFEST_PATH,
             FORMANT_CONFIG_PATH, SPEECH_MASK_CONFIG_PATH]
    if MANUAL_EXTRAS_PATH.exists():
        paths.append(MANUAL_EXTRAS_PATH)
    # Hash the implementation too; configuration alone cannot reproduce a
    # sweep if Burg extraction or frame selection code changes later.
    for relative in ("scripts/dsp/inspect_formants.py", "src/dsp/features/formants.py",
                     "src/dsp/features/pitch.py", "src/dsp/speech_mask.py"):
        path = PROJECT_ROOT / relative
        if path.is_file():
            paths.append(path)
    settings = {
        "formant_feature_version": FEATURE_VERSION,
        "formant_config": formant_config,
        "speech_mask_config": speech_mask_config,
        "input_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in paths},
        "reference_ceiling_hz": reference_ceiling,
        "ceilings_hz": ceilings,
        "policies": POLICIES,
        "loud_half_energy_window_s": ENERGY_WINDOW_S,
        "window_lengths_s": WINDOW_LENGTHS_S,
        "window_positions": WINDOW_POSITIONS,
        "large_jump_fraction": LARGE_JUMP_FRACTION,
        "wide_bandwidth_ratio": WIDE_BANDWIDTH_RATIO,
        "selection_seed": RANDOM_SEED,
        "n_stress_per_dataset": N_STRESS_PER_DATASET,
        "n_suspicious_per_metric": N_SUSPICIOUS_PER_METRIC,
        "n_random_per_dataset": N_RANDOM_PER_DATASET,
        "write_plots": not args.no_plots,
        "dev_only": args.dev_only,
        "stability_min_full_usable_s": STABILITY_MIN_FULL_USABLE_S,
        "coverage_minima_s": COVERAGE_MINIMA_S,
        "spectrogram_window_samples": SPECTROGRAM_WINDOW,
        "spectrogram_preemphasis": SPECTROGRAM_PREEMPHASIS,
        "voicing_grid_offset_units": "pitch frame steps (0 aligned, 0.5 midpoint)",
    }
    (output / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    print(
        f"Measuring {len(recordings)} recordings at {len(ceilings)} ceilings, "
        f"{len(POLICIES)} policies"
    )
    summary_rows, window_rows = [], []
    for number, row in enumerate(recordings.itertuples(index=False), start=1):
        if number == 1 or number % 20 == 0 or number == len(recordings):
            print(f"[{number}/{len(recordings)}] {row.sample_id}")
        results, windows = measure_recording(row, formant_config, speech_mask_config, ceilings)
        summary_rows.extend(results)
        window_rows.extend(windows)

    summary = pd.DataFrame(summary_rows)
    for column in SUMMARY_FEATURES:
        if column not in summary:
            summary[column] = np.nan
    # Retain the expected diagnostic columns even if every candidate fails.
    for column in (
        "invalid_voiced_fraction", "f3_jump_fraction", "f3_wide_bandwidth_fraction",
        "usable_duration_s", "voiced_valid_fraction", "voicing_grid_offset_steps",
        "f0_median_hz", "n_disordered_frames",
        *(f"f{i}_jump_fraction" for i in (1, 2)),
    ):
        if column not in summary:
            summary[column] = np.nan
    summary = add_comparisons(summary, reference_ceiling)
    summary.to_csv(output / "summary.csv", index=False)
    window_columns = [
        "sample_id", "dataset", "accent_group", "in_analysis_manifest",
        "in_dev_manifest", "formant_ceiling_hz", "policy", "window_length_s",
        "position", "start_s", "n_usable_formant_frames", "window_usable_s",
        *(f"f{i}_{suffix}" for i in (1, 2, 3)
          for suffix in ("median_hz", "abs_shift_hz", "abs_shift_st",
                         "p10_abs_shift_st", "p90_abs_shift_st")),
    ]
    windows = pd.DataFrame(window_rows, columns=window_columns)
    windows.to_csv(output / "duration_windows.csv", index=False)

    overviews = build_overviews(summary, windows)
    for filename, table in overviews.items():
        table.to_csv(output / filename, index=False)

    selected = select_inspection_rows(recordings, summary, extras, reference_ceiling)
    selected.to_csv(output / "selection.csv", index=False)
    build_inspection_sheet(selected, ceilings).to_csv(output / "inspection_sheet.csv", index=False)
    select_manual_subset(selected, summary, reference_ceiling).to_csv(
        output / "manual_subset.csv", index=False
    )

    n_plot_errors = 0
    if not args.no_plots:
        print(f"\nPlotting {len(selected)} recordings")
        for number, row in enumerate(selected.itertuples(index=False), start=1):
            print(f"[{number}/{len(selected)}] {row.sample_id}")
            try:
                plot_recording(row, formant_config, speech_mask_config, ceilings,
                               reference_ceiling, output / "plots" / f"{row.sample_id}.png")
            except Exception as exc:
                n_plot_errors += 1
                print(f"  Plot failed: {type(exc).__name__}: {exc}")

    print_overview(summary)
    print(f"\nRows with status != ok: {int((summary['status'] != 'ok').sum())}")
    print(f"Recordings selected for inspection: {len(selected)}")
    if not args.no_plots:
        print(f"Recordings not plotted: {n_plot_errors}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()


# run using:
# python -m scripts.dsp.inspect_formants --dev-only
# python -m scripts.dsp.inspect_formants
# python -m scripts.dsp.inspect_formants --no-plots
# python -m scripts.dsp.inspect_formants --ceilings 5000 5250 5500
# python -m scripts.dsp.inspect_formants --run-name ceiling_review --overwrite
