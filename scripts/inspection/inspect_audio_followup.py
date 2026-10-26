"""Follow-up checks on the audio inspection CSV. Scratch script, no tests.

Looks more closely at the three open issues:
  - band-limited files (almost no energy above 4 kHz)
  - very short / very long files
  - files where a large share of the duration is edge silence

Run from project root, passing the inspection CSV:
    python -m scripts.inspection.inspect_audio_followup scripts/inspection/results/audio_inspection_study_manifest.csv
"""

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "study_manifest.csv"

GROUP_COLS = ["dataset", "accent_group"]
SHOW_COLS = ["sample_id", "accent_group", "speaker_id", "duration_s", "hf_energy_pct", "filepath"]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    qc = pd.read_csv(sys.argv[1])
    manifest = pd.read_csv(STUDY_MANIFEST, keep_default_na=False)
    df = qc.merge(manifest[["sample_id", "speaker_id", "filepath"]], on="sample_id", how="left")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 120)

    groups = [df["dataset"], df["accent_group"]]

    # --- Band-limiting -----------------------------------------------------
    df["hf_bin"] = pd.cut(
        df["hf_energy_pct"],
        bins=[-1, 0.01, 0.05, 0.1, 0.5, 1000],
        labels=["<0.01", "0.01-0.05", "0.05-0.1", "0.1-0.5", ">=0.5"],
    )
    section("Files by % energy above 4 kHz")
    print(pd.crosstab(groups, df["hf_bin"]))

    low = df[df["hf_energy_pct"] < 0.05]
    section("Files with <0.05% energy above 4 kHz: files and distinct speakers")
    print(low.groupby(GROUP_COLS).agg(
        files=("sample_id", "size"),
        speakers=("speaker_id", "nunique"),
    ))

    section("Lowest 10 energy above 4 kHz")
    print(df.nsmallest(10, "hf_energy_pct")[SHOW_COLS].round(3).to_string(index=False))

    section("Highest 10 energy above 4 kHz")
    print(df.nlargest(10, "hf_energy_pct")[SHOW_COLS].round(3).to_string(index=False))

    # --- Duration -----------------------------------------------------------
    df["dur_bin"] = pd.cut(
        df["duration_s"],
        bins=[0, 2, 3, 5, 30, 60, 1e6],
        labels=["<2", "2-3", "3-5", "5-30", "30-60", ">60"],
    )
    section("Files by duration (s)")
    print(pd.crosstab(groups, df["dur_bin"]))

    section("Shortest 10")
    print(df.nsmallest(10, "duration_s")[SHOW_COLS].round(3).to_string(index=False))

    section("Longest 10")
    print(df.nlargest(10, "duration_s")[SHOW_COLS].round(3).to_string(index=False))

    # --- Silence-heavy files ------------------------------------------------
    heavy = df[df["pct_trimmed_30db"] > 20]
    section(f"Files where >20% is edge silence at 30 dB: {len(heavy)}")
    if not heavy.empty:
        print(heavy[SHOW_COLS + ["pct_trimmed_30db"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()