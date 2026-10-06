"""
Normalize and score a completed or partially completed ASR run.

Example:

python -m scripts.asr.run_scoring \
    --run-dir results/asr/runs/dev_small_long
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

import jiwer
import pandas as pd

from src.asr.scoring_pipeline import (run_scoring)
from src.asr.normalize import (describe_normalizer, AFRISPEECH_PUNCTUATION_TERMS)


DEFAULT_OUTPUT_NAME = "scored.csv"
DEFAULT_CONFIG_NAME = "scoring_config.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Normalize and calculate WER for a saved ASR transcription run.")
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "ASR experiment directory containing "
            "manifest_snapshot.csv and transcriptions.csv."
        ),
    )

    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=(
            "Filename for scored results. Default: scored.csv"
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop immediately if normalization or WER calculation fails."
        ),
    )

    parser.add_argument(
        "--drop-punctuation-words",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove spoken punctuation words (AfriSpeech only). "
            "Use --no-drop-punctuation-words to keep them. Default: drop."
        ),
    )

    return parser.parse_args()


def build_scoring_config(
    manifest_path: Path,
    transcriptions_path: Path,
    drop_punctuation_words: bool,
) -> dict:
    """
    Build metadata describing the scoring run.
    """

    return {
        "manifest_path": str(manifest_path.resolve()),
        "transcriptions_path": str(transcriptions_path.resolve()),

        "normalization": {
            **describe_normalizer(),
            "drop_punctuation_words": drop_punctuation_words,
            "afrispeech_punctuation_terms": list(AFRISPEECH_PUNCTUATION_TERMS),
        },

        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "jiwer": version("jiwer"),
            "transformers": version("transformers"),
        },

        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    args = parse_args()

    run_dir = args.run_dir

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    if not run_dir.is_dir():
        raise ValueError(
            f"Run path is not a directory: {run_dir}"
        )

    manifest_path = (run_dir/ "manifest_snapshot.csv")

    transcriptions_path = (run_dir/ "transcriptions.csv")
    output_path = (run_dir/ args.output_name)
    config_path = run_dir / f"{Path(args.output_name).stem}_config.json"
    # config_path = (run_dir/ DEFAULT_CONFIG_NAME)

   
    # Verify required run files
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest snapshot does not exist: {manifest_path}"
        )

    if not transcriptions_path.exists():
        raise FileNotFoundError(
            f"Transcription results do not exist: {transcriptions_path}"
        )


    # Load input data
    manifest = pd.read_csv(manifest_path)
    transcriptions = pd.read_csv(transcriptions_path)


    # Score
    results = run_scoring(
        manifest=manifest,
        transcriptions=transcriptions,
        fail_fast=args.fail_fast,
        drop_punctuation_words=args.drop_punctuation_words,
    )


    # Save derived scoring results
    results.to_csv(output_path, index=False)

    # Save scoring metadata
    scoring_config = build_scoring_config(
        manifest_path,
        transcriptions_path,
        args.drop_punctuation_words,
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            scoring_config,
            file,
            indent=2,
        )

    print()
    print(f"Scored results saved to: {output_path}")
    print(f"Scoring metadata saved to: {config_path}")


if __name__ == "__main__":
    main()