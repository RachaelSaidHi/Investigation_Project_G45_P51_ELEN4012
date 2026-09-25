"""
Run Whisper transcription over a manifest.

Example:

python -m scripts.asr.run_transcription \
    --manifest data/manifests/dev_manifest.csv \
    --run-name dev_tiny_long \
    --model openai/whisper-tiny \
    --decode-mode sequential_long_form
"""

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import pandas as pd
import torch
import transformers

from src.asr.transcribe import WhisperTranscriber
from src.asr.transcription_pipeline import (run_transcription_manifest)


DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_RESULTS_ROOT = Path("results/asr/runs")


def build_run_config(
    args,
    transcriber: WhisperTranscriber,
    manifest_hash: str,
) -> dict:
    """
    Build the configuration saved alongside the experiment.
    """

    return {
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "data_root": str(args.data_root.resolve()),
        "whisper": transcriber.describe(),

        "transcription": {
            "duration_tolerance_s": args.duration_tolerance,
        },

        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },

        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_run_directory(
    run_dir: Path,
    manifest_path: Path,
    run_config: dict,
) -> Path:
    """
    Create a new run directory or validate an existing one
    before resuming.

    Returns:
        Path to the manifest snapshot belonging to the run.
    """

    config_path = run_dir / "run_config.json"
    manifest_snapshot_path = (run_dir / "manifest_snapshot.csv")

    # Resume an existing run
    if run_dir.exists():

        if not config_path.exists():
            raise FileNotFoundError(
                "Existing run directory has no "
                f"run_config.json: {run_dir}"
            )

        if not manifest_snapshot_path.exists():
            raise FileNotFoundError(
                "Existing run directory has no "
                f"manifest_snapshot.csv: {run_dir}"
            )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            previous_config = (
                json.load(file)
            )

        validate_resume_config(previous_config, run_config)

        print(f"Resuming run: {run_dir.name}")

        return manifest_snapshot_path

    run_dir.mkdir(parents=True, exist_ok=False)

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_config, file, indent=2)

    shutil.copy2(manifest_path, manifest_snapshot_path)

    print(f"Created run: {run_dir.name}")

    return manifest_snapshot_path

    
def parse_args():
    parser = argparse.ArgumentParser(
        description=("Run Whisper transcription over a manifest.")
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest CSV to transcribe.",
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "Unique name for this experiment, e.g. 'pilot_small_long'."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face Whisper checkpoint, e.g. 'openai/whisper-small'."
        ),
    )

    parser.add_argument(
        "--decode-mode",
        choices=[
            "short_form",
            "sequential_long_form",
        ],
        default="sequential_long_form",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device such as 'cuda', 'cuda:0' or 'cpu'. Detected automatically if omitted."
        ),
    )

    parser.add_argument(
        "--fp32",
        action="store_true",
        help=(
            "Use FP32 instead of FP16 on CUDA. CPU always uses FP32."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop the run after the first sample error."
        ),
    )

    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=0.5,
        help=(
            "Allowed difference in seconds between manifest and processed durations."
        ),
    )

    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help=(
            "Retry samples previously recorded with status='error'. "
            "Successful samples are never repeated."
        ),
    )

    return parser.parse_args()

def file_sha256(path: Path) -> str:
    """
    Return the SHA-256 hash of a file to ensure manfifest hasn't changed.
    """

    hasher = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def validate_resume_config(previous: dict, current: dict) -> None:
    """
    Ensure an existing run is only resumed using the same
    experimental configuration.
    """

    # Maps which settings need to be identical
    comparisons = {
        "manifest": (
            previous["manifest_sha256"],
            current["manifest_sha256"],
        ),

        "data_root": (
            previous["data_root"],
            current["data_root"],
        ),

        "model_name": (
            previous["whisper"]["model_name"],
            current["whisper"]["model_name"],
        ),

        "decode_mode": (
            previous["whisper"]["decode_mode"],
            current["whisper"]["decode_mode"],
        ),

        "precision": (
            previous["whisper"]["precision"],
            current["whisper"]["precision"],
        ),

        "device": (
            previous["whisper"]["device"],
            current["whisper"]["device"],
        ),

        "hardware": (
            previous["whisper"]["hardware"],
            current["whisper"]["hardware"],
        ),

        "torch": (
            previous["software"]["torch"],
            current["software"]["torch"],
        ),

        "transformers": (
            previous["software"]["transformers"],
            current["software"]["transformers"],
        ),

        "duration_tolerance_s": (
            previous["transcription"]["duration_tolerance_s"],
            current["transcription"]["duration_tolerance_s"],
        ),
    }

    mismatches = []

    for name, (previous_value, current_value) in comparisons.items():

        if ( previous_value != current_value):
            mismatches.append(
                f"{name}: "
                f"previous={previous_value!r}, "
                f"current={current_value!r}"
            )

    if mismatches:
        differences = "\n".join(mismatches)

        raise ValueError(
            "Cannot resume this run because its configuration has changed:\n"
            f"{differences}\n\n"
            "Use a different --run-name for the new experiment."
        )


def main():
    args = parse_args()

    # Load manifest
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {args.manifest}"
        )

    if not args.manifest.is_file():
        raise ValueError(
            f"Manifest path is not a file: {args.manifest}"
        )

    run_dir = (args.results_root / args.run_name)

    # Load Whisper once
    transcriber = WhisperTranscriber(
        model_name=args.model,
        device=args.device,
        use_fp16=not args.fp32,
        decode_mode=args.decode_mode,
    )

    # Build current experiment configuration (run metadata)
    manifest_hash = file_sha256(args.manifest)

    run_config = build_run_config(
        args=args,
        transcriber=transcriber,
        manifest_hash=manifest_hash,
    )

    # Create or resume run directory
    manifest_snapshot_path = (
        prepare_run_directory(
            run_dir=run_dir,
            manifest_path=args.manifest,
            run_config=run_config,
        )
    )

    # Always use the saved snapshot belonging to this experiment.
    manifest = pd.read_csv(manifest_snapshot_path)

    # Run or resume transcription
    results_path = (run_dir / "transcriptions.csv")

    run_transcription_manifest(
        manifest=manifest,
        transcriber=transcriber,
        data_root=args.data_root,
        results_path=results_path,
        retry_errors=args.retry_errors,
        fail_fast=args.fail_fast,
        duration_tolerance_s=args.duration_tolerance,
    )

    print()
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()