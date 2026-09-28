"""Inspect audio properties of a manifest before designing preprocessing.

Does NOT modify any audio. For each file it records:
  - header info (sample rate, channels, MP3 bitrate / WAV subtype)
  - peak level and fraction of clipped samples
  - how much leading/trailing silence a relative-threshold trim WOULD
    remove, at several thresholds, after conversion to 16 kHz mono

Results are written to results/inspection/ and summarised by
dataset + accent group.

Run from project root:
    python -m scripts.inspection.inspect_audio
    python -m scripts.inspection.inspect_audio --manifest data/manifests/dev_manifest.csv
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from mutagen.mp3 import MP3


# Script lives in scripts/inspection/, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "study_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "inspection" / "results"

TARGET_SR = 16_000
CLIP_LEVEL = 0.999

# Trim thresholds in dB below the loudest frame of each file.
# Several are recorded to see how sensitive the silence estimate is.
TRIM_THRESHOLDS_DB = [30, 40, 50]
TRIM_FRAME_LENGTH = 1024  # 64 ms at 16 kHz
TRIM_HOP_LENGTH = 256     # 16 ms at 16 kHz

# Energy above this frequency is used to detect band-limited recordings
# (e.g. originally captured at 8 kHz, which contain nothing above 4 kHz).
HF_CUTOFF_HZ = 4000

GROUP_COLS = ["dataset", "accent_group"]


def header_info(path: Path) -> dict:
    """Read format information from the file header without decoding."""
    if path.suffix.lower() == ".mp3":
        info = MP3(path).info
        return {
            "format": "mp3",
            "header_sr": info.sample_rate,
            "channels": info.channels,
            "bitrate_kbps": info.bitrate / 1000,
            "bitrate_mode": str(info.bitrate_mode),
        }

    info = sf.info(path)
    return {
        "format": info.subtype,
        "header_sr": info.samplerate,
        "channels": info.channels,
        "bitrate_kbps": np.nan,
        "bitrate_mode": "",
    }


def rms_db(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x ** 2))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def channel_info(y: np.ndarray) -> dict:
    """Compare the two channels of a stereo file before downmixing."""
    if y.ndim != 2 or y.shape[0] != 2:
        return {}

    left, right = y
    return {
        "channel_corr": float(np.corrcoef(left, right)[0, 1]),
        "channel_max_diff": float(np.max(np.abs(left - right))),
        "left_rms_db": rms_db(left),
        "right_rms_db": rms_db(right),
    }


def hf_energy_pct(y16: np.ndarray) -> float:
    """Percentage of spectral energy above HF_CUTOFF_HZ in the 16 kHz signal."""
    power = np.abs(librosa.stft(y16, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=1024)
    return float(100 * power[freqs >= HF_CUTOFF_HZ].sum() / power.sum())


def signal_info(path: Path) -> dict:
    """Decode the audio and measure level, clipping and edge silence."""
    y, sr = librosa.load(path, sr=None, mono=False)

    peak = float(np.max(np.abs(y)))
    clipped_fraction = float(np.mean(np.abs(y) >= CLIP_LEVEL))

    # Measure silence on the signal we would actually trim: 16 kHz mono.
    mono = librosa.to_mono(y)
    y16 = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    duration = len(y16) / TARGET_SR

    result = {
        "decoded_sr": sr,
        "duration_s": duration,
        "peak_dbfs": 20 * np.log10(peak) if peak > 0 else -np.inf,
        "clipped_fraction": clipped_fraction,
        "hf_energy_pct": hf_energy_pct(y16),
    }
    result.update(channel_info(y))

    for top_db in TRIM_THRESHOLDS_DB:
        _, (start, end) = librosa.effects.trim(
            y16,
            top_db=top_db,
            frame_length=TRIM_FRAME_LENGTH,
            hop_length=TRIM_HOP_LENGTH,
        )
        leading = start / TARGET_SR
        trailing = (len(y16) - end) / TARGET_SR

        result[f"lead_sil_{top_db}db"] = leading
        result[f"trail_sil_{top_db}db"] = trailing
        result[f"pct_trimmed_{top_db}db"] = 100 * (leading + trailing) / duration

    return result


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def print_summary(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)

    errors = df[df["error"] != ""]
    ok = df[df["error"] == ""].copy()

    section(f"Files: {len(df)} total, {len(errors)} failed")
    if not errors.empty:
        print(errors[["sample_id", "error"]].to_string(index=False))

    section("Header sample rate (count of files)")
    print(ok.groupby(GROUP_COLS)["header_sr"].value_counts().unstack(fill_value=0))

    mismatch = ok[ok["header_sr"] != ok["decoded_sr"]]
    print(f"\nHeader vs decoded sample rate mismatches: {len(mismatch)}")

    section("Channels (count of files)")
    print(ok.groupby(GROUP_COLS)["channels"].value_counts().unstack(fill_value=0))

    section("Format / WAV subtype (count of files)")
    print(ok.groupby(GROUP_COLS)["format"].value_counts().unstack(fill_value=0))

    mp3 = ok[ok["format"] == "mp3"]
    if not mp3.empty:
        section("MP3 bitrate (kbps)")
        print(mp3.groupby(GROUP_COLS).agg(
            files=("bitrate_kbps", "size"),
            min=("bitrate_kbps", "min"),
            median=("bitrate_kbps", "median"),
            max=("bitrate_kbps", "max"),
            below_64=("bitrate_kbps", lambda s: (s < 64).sum()),
        ).round(1))
        print("\nBitrate mode:")
        print(mp3["bitrate_mode"].value_counts())

    section("Level and clipping")
    print(ok.groupby(GROUP_COLS).agg(
        files=("sample_id", "size"),
        median_peak_dbfs=("peak_dbfs", "median"),
        files_with_clipping=("clipped_fraction", lambda s: (s > 0).sum()),
        max_clipped_pct=("clipped_fraction", lambda s: 100 * s.max()),
    ).round(3))

    stereo = ok[ok["channels"] == 2]
    if not stereo.empty:
        stereo = stereo.assign(
            lr_rms_diff_db=(stereo["left_rms_db"] - stereo["right_rms_db"]).abs()
        )
        section("Stereo files: are the channels identical?")
        print(stereo.groupby(GROUP_COLS).agg(
            files=("sample_id", "size"),
            min_corr=("channel_corr", "min"),
            median_corr=("channel_corr", "median"),
            max_abs_diff=("channel_max_diff", "max"),
            max_lr_rms_diff_db=("lr_rms_diff_db", "max"),
        ).round(4))

    section(f"Energy above {HF_CUTOFF_HZ} Hz (% of total) - near zero means band-limited")
    print(ok.groupby(GROUP_COLS)["hf_energy_pct"].agg(["min", "median", "max"]).round(3))
    print("\nLowest 10 files:")
    print(ok.nsmallest(10, "hf_energy_pct")[
        ["sample_id", "accent_group", "hf_energy_pct"]
    ].round(3).to_string(index=False))

    section("Duration (s)")
    print(ok.groupby(GROUP_COLS)["duration_s"].agg(["min", "median", "max"]).round(1))

    for top_db in TRIM_THRESHOLDS_DB:
        section(f"Edge silence at {top_db} dB below peak (seconds / % of file)")
        print(ok.groupby(GROUP_COLS).agg(
            lead_median=(f"lead_sil_{top_db}db", "median"),
            lead_max=(f"lead_sil_{top_db}db", "max"),
            trail_median=(f"trail_sil_{top_db}db", "median"),
            trail_max=(f"trail_sil_{top_db}db", "max"),
            pct_median=(f"pct_trimmed_{top_db}db", "median"),
            pct_max=(f"pct_trimmed_{top_db}db", "max"),
        ).round(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    n = len(manifest)
    print(f"Inspecting {n} files from {args.manifest}")

    rows = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        path = RAW_DIR / row.filepath
        record = {
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "accent_group": row.accent_group,
            "sex": row.sex,
            "error": "",
        }
        try:
            record.update(header_info(path))
            record.update(signal_info(path))
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(record)

        if i % 50 == 0 or i == n:
            print(f"  {i}/{n}")

    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"audio_inspection_{Path(args.manifest).stem}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nPer-file results written to: {output_path}")

    print_summary(df)


if __name__ == "__main__":
    main()

# run using
# python -m scripts.inspection.inspect_audio 