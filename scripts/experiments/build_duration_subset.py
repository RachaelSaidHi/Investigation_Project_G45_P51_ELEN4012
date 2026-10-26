"""
Create a manifest containing only recordings at or below a
specified maximum duration, optionally restricted to one or
more datasets.
 
Example for SAA audios <= 30s:
 
python -m scripts.experiments.build_duration_subset \
    --input data/manifests/study_manifest.csv \
    --max-duration 30 \
    --dataset saa
 
This saves data/manifests/study_under_30s_saa.csv. Pass
--output to choose a different path.
 
Both short-form and long-form transcription are
run on the created manifest to compare how WER
changes.
"""

import argparse
from pathlib import Path

import pandas as pd

def filter_by_dataset(manifest: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """
    Keep samples belonging to any of the given datasets.
    """
 
    datasets = [d.lower() for d in datasets]
    dataset_column = manifest["dataset"].astype(str).str.lower()
 
    available = sorted(dataset_column.unique())
    unknown = sorted(set(datasets) - set(available))
 
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. "
            f"Available in manifest: {available}"
        )
 
    filtered = manifest.loc[dataset_column.isin(datasets)].copy()
 
    return filtered.reset_index(drop=True)


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


def default_output_path(
    input_path: Path,
    max_duration_s: float,
    datasets: list[str] | None,
) -> Path:
    """
    Build an output path such as study_under_30s_saa.csv.
    """
 
    stem = input_path.stem.removesuffix("_manifest")
    name = f"{stem}_under_{max_duration_s:g}s"
 
    if datasets:
        name += "_" + "_".join(d.lower() for d in datasets)
 
    return input_path.parent / f"{name}.csv"


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
        default=None,
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

    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help=(
            "Only keep samples from these datasets, e.g. --dataset saa. \n"
            "Default: keep all datasets."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {args.input}"
        )

    output = args.output or default_output_path(
        args.input,
        args.max_duration,
        args.dataset,
    )

    manifest = pd.read_csv(args.input)
    print(f"Original samples: {len(manifest)}")

    filtered = manifest

    if args.dataset:
        filtered = filter_by_dataset(filtered, args.dataset)
        print(f"Samples in {', '.join(args.dataset)}: {len(filtered)}")
 
    n_before_duration = len(filtered)
 
    filtered = filter_by_duration(
        filtered,
        args.max_duration,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    filtered.to_csv(output, index=False,)

    print(f"Samples <= {args.max_duration:.1f}s: {len(filtered)}")
    print(f"Removed by duration: {n_before_duration - len(filtered)}")

    if len(filtered) > 0:
        print(
            f"Duration range: {filtered['duration_s'].min():.2f}s "
            f"to {filtered['duration_s'].max():.2f}s"
        )

    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()