"""
Create a manifest containing only recordings at or below a
specified maximum duration.

Example for audios <= 30s:

python -m scripts.experiments.build_duration_subset \
    --input data/manifests/study_manifest.csv \
    --output data/manifests/study_under_30s.csv \
    --max-duration 30

Both short-form and long-form transcription are
run on the created manifest to compare how WER
changes.
"""

import argparse
from pathlib import Path

import pandas as pd


def filter_by_duration(manifest: pd.DataFrame, max_duration_s: float) -> pd.DataFrame:
    """
    Keep samples whose manifest duration is <= max_duration_s.
    """

    if "duration_s" not in manifest.columns:
        raise ValueError(
            "Manifest must contain a duration_s column."
        )

    if manifest["duration_s"].isna().any():
        raise ValueError(
            "Manifest contains missing duration_s values."
        )

    if (manifest["duration_s"] <= 0).any():
        raise ValueError(
            "Manifest contains non-positive duration values."
        )

    # Only keeps manifest rows with duration <= specified maximum
    filtered = manifest.loc[manifest["duration_s"] <= max_duration_s].copy()

    return filtered.reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Create a duration-filtered manifest.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source manifest.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Filtered manifest output path.",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help=(
            "Maximum recording duration in seconds. \n"
            "Default: 30."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {args.input}"
        )

    manifest = pd.read_csv(args.input)

    filtered = filter_by_duration(
        manifest,
        args.max_duration,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    filtered.to_csv(args.output, index=False,)

    print(f"Original samples: {len(manifest)}")

    print(f"Samples <= {args.max_duration:.1f}s: {len(filtered)}")
    print(f"Removed: {len(manifest) - len(filtered)}")

    if len(filtered) > 0:
        print(
            f"Duration range: {filtered['duration_s'].min():.2f}s "
            f"to {filtered['duration_s'].max():.2f}s"
        )

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()