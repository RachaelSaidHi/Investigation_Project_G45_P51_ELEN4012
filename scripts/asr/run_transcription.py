"""
Run Whisper transcription over a manifest.

Example:

python -m scripts.run_transcription \
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

import pandas as pd
import torch
import transformers

from src.asr.transcribe import WhisperTranscriber
from src.asr.transcription_pipeline import (
    run_transcription_manifest,
)


DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_RESULTS_ROOT = Path("results/asr/runs")


def build_run_config(
    args,
    transcriber: WhisperTranscriber,
) -> dict:
    """
    Build the configuration saved alongside the experiment.
    """

    return {
        "manifest_path": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "whisper": transcriber.describe(),

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
) -> None:
    """
    Create the directory for a new transcription experiment.
    """

    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: {run_dir}\n Choose a different --run-name."
        )

    run_dir.mkdir(parents=True, exist_ok=True,)

    config_path = run_dir / "run_config.json"
    manifest_snapshot_path = (run_dir / "manifest_snapshot.csv")

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_config, file, indent=2)

    shutil.copy2(manifest_path, manifest_snapshot_path)

    print(f"Created run: {run_dir.name}")

    
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

    return parser.parse_args()


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

    manifest = pd.read_csv(args.manifest)

    run_dir = (args.results_root / args.run_name)

    if run_dir.exists():
        raise FileExistsError(
            f"Run already exists: {run_dir}\n Choose a different --run-name."
        )


    # Load Whisper once
    transcriber = WhisperTranscriber(
        model_name=args.model,
        device=args.device,
        use_fp16=not args.fp32,
        decode_mode=args.decode_mode,
    )


    # Create run metadata
    run_config = build_run_config(
        args=args,
        transcriber=transcriber,
    )

    # Run directory which throws an error if run already exists
    prepare_run_directory(
        run_dir=run_dir,
        manifest_path=args.manifest,
        run_config=run_config,
    )

    # Run transcription
    results_path = (run_dir / "transcriptions.csv")

    results = run_transcription_manifest(
        manifest=manifest,
        transcriber=transcriber,
        data_root=args.data_root,
        fail_fast=args.fail_fast,
        duration_tolerance_s=args.duration_tolerance,
    )

    # Save transcription results
    results_path = (run_dir / "transcriptions.csv")

    results.to_csv(results_path, index=False,)

    print()
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()