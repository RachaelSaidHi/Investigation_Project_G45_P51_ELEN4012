"""
Run DSP feature extraction over a manifest.
 
Each feature family (e.g. mfcc, pitch, formants) is a module in
src/dsp/features/ with a matching config file in configs/dsp/. This script
runs one or more of them, one feature at a time over the whole manifest.

Re-running the same command resumes each feature where it stopped. A
feature's run is only resumed if its configuration (manifest, feature
version, config files, software versions) is unchanged; otherwise use a new
--run-name.

Examples (from the project root):
 
    # One feature
    python -m scripts.dsp.run_feature_extraction \
        --manifest data/manifests/dev_manifest.csv \
        --run-name dev_v1 \
        --feature mfcc
 
    # Several features, one after the other
    python -m scripts.dsp.run_feature_extraction \
        --manifest data/manifests/dev_manifest.csv \
        --run-name dev_v1 \
        --feature mfcc pitch
 
    # Every feature in FEATURE_MODULES
    python -m scripts.dsp.run_feature_extraction \
        --manifest data/manifests/dev_manifest.csv \
        --run-name dev_v1 \
        --feature all

When several features are run, a feature that crashes (e.g. a bug in its
code or a missing config file) is reported and the remaining features still
run. The script exits with an error at the end if any feature failed.
"""

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import argparse
import importlib
 
import numpy as np
import pandas as pd
import soundfile as sf
 
from src.common.run_management import file_sha256, prepare_run_directory
from src.dsp.extraction_pipeline import run_feature_manifest, validate_feature_module
from src.preprocessing.audio import TARGET_SAMPLE_RATE


FEATURES_PACKAGE = "src.dsp.features"

# Features run by --feature all, in this order. Add a feature here once its
# module (src/dsp/features/<name>.py) and config (configs/dsp/<name>.json) exist.
FEATURE_MODULES = [
    "rms_test",
    "mfcc",
    "pitch",
    "formants",
]


DEFAULT_PROCESSED_ROOT = Path("data/processed")
DEFAULT_RESULTS_ROOT = Path("results/dsp/runs")
DEFAULT_CONFIG_DIR = Path("configs/dsp")
 
SPEECH_MASK_CONFIG_NAME = "speech_mask.json"
RESULTS_FILE_NAME = "features.csv"

# Keys in run_config that may legitimately differ when a run is resumed
# so they are excluded from the resume check.
IGNORED_CONFIG_KEYS = (
    "created_at_utc",
    "paths",
)

# Package versions recorded in each run's config.
RECORDED_PACKAGES = [
    "numpy",
    "pandas",
    "librosa",
    "soundfile",
    "praat-parselmouth",
]


# Audio loading
def temporary_loader(processed_root: Path, processed_filepath: str) -> np.ndarray:
    """
    Load one processed WAV as a mono 16 kHz float32 array.
 
    Stand-in until audio_io.load_processed_audio is available. It refuses
    files that are not 16 kHz mono rather than resampling them, so every
    feature analyses exactly the audio produced by preprocessing.
 
    TODO: replace with load_processed_audio from src/common/audio_io.py.
    """
 
    audio, sample_rate = sf.read(
        processed_root / processed_filepath,
        dtype="float32",
    )
 
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
        )
 
    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio, got {audio.shape[1]} channels."
        )
 
    return audio



# Features and configs
def resolve_feature_names(requested: list[str]) -> list[str]:
    """
    Turn the --feature arguments into the list of features to run.

    "all" expands to FEATURE_MODULES. Duplicates are removed while keeping
    the order given.
    """

    if "all" in requested:
        if len(requested) > 1:
            raise ValueError("Use either --feature all or a list of features, not both.")

        return list(FEATURE_MODULES)

    return list(dict.fromkeys(requested))


def load_feature_module(feature_name: str):
    """
    Import src/dsp/features/<feature_name>.py and check it has what the pipeline needs.
 
    A missing feature file gives a clear "unknown feature" error. If the file
    exists but one of its own imports fails (e.g. parselmouth not installed),
    the original error is raised instead, since that is the real problem.
    """
 
    module_path = f"{FEATURES_PACKAGE}.{feature_name}"
 
    try:
        feature_module = importlib.import_module(module_path)
 
    except ModuleNotFoundError as error:
        if error.name == module_path:
            raise ValueError(
                f"Unknown feature '{feature_name}': no file src/dsp/features/{feature_name}.py. "
                f"Features in FEATURE_MODULES: {FEATURE_MODULES}"
            ) from None
        raise
 
    validate_feature_module(feature_module)
 
    # The run folder is named after the module, so the module's declared
    # name must match it to avoid two names for the same feature.
    if feature_module.FEATURE_NAME != feature_name:
        raise ValueError(
            f"Feature module {feature_name}.py declares "
            f"FEATURE_NAME = '{feature_module.FEATURE_NAME}'. They must match."
        )
 
    return feature_module


def read_json(path: Path) -> dict:
    """
    Read a JSON config file, with a clear error if it does not exist.
    """
 
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
 
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_config(config_dir: Path, feature_name: str) -> dict:
    """
    Build the config passed to a feature's feature_columns() and extract().
 
    Combines the feature's own settings (configs/dsp/<feature>.json) with
    the speech-mask settings shared by all features.
    """
 
    return {
        "feature": read_json(config_dir / f"{feature_name}.json"),
        "speech_mask": read_json(config_dir / SPEECH_MASK_CONFIG_NAME),
    }



# Run configuration
def package_version(package: str) -> str:
    """
    Return an installed package's version, or "not installed".
    """
 
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def build_run_config(
    manifest_path: Path,
    processed_root: Path,
    feature_module,
    config: dict,
) -> dict:
    """
    Describe everything that defines one feature's run.
 
    Saved as run_config.json. On resume, every key except those in
    IGNORED_CONFIG_KEYS must be identical, so a run is never continued with
    a different manifest, feature version, config or software version.
    """
 
    return {
        "manifest_sha256": file_sha256(manifest_path),
 
        "feature": {
            "name": feature_module.FEATURE_NAME,
            "version": feature_module.FEATURE_VERSION,
        },
 
        "config": config,
 
        "software": {
            "python": platform.python_version(),
            **{package: package_version(package) for package in RECORDED_PACKAGES},
        },
 
        # Absolute paths differ between machines, so they are recorded for
        # reference but ignored when resuming.
        "paths": {
            "manifest": str(manifest_path.resolve()),
            "processed_root": str(processed_root.resolve()),
        },
 
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }



# Running
def run_one_feature(feature_name: str, args) -> pd.DataFrame:
    """
    Set up or resume one feature's run folder and extract it over the manifest.
    """
 
    feature_module = load_feature_module(feature_name)
    config = load_config(args.config_dir, feature_name)
 
    run_config = build_run_config(
        manifest_path=args.manifest,
        processed_root=args.processed_root,
        feature_module=feature_module,
        config=config,
    )
 
    run_dir = args.results_root / args.run_name / feature_name
 
    manifest_snapshot_path = prepare_run_directory(
        run_dir=run_dir,
        manifest_path=args.manifest,
        run_config=run_config,
        ignored_keys=IGNORED_CONFIG_KEYS,
    )
 
    # Always use the run's own snapshot so a resumed run processes exactly
    # the same samples. Blank cells are read as "" to match the manifest
    # builders.
    manifest = pd.read_csv(manifest_snapshot_path, keep_default_na=False)

    results_path = run_dir / RESULTS_FILE_NAME
    
    return run_feature_manifest(
        manifest=manifest,
        feature_module=feature_module,
        config=config,
        processed_root=args.processed_root,
        results_path=results_path,
        load_audio=temporary_loader,
        retry_errors=args.retry_errors,
        fail_fast=args.fail_fast,
    )


def print_section(title: str) -> None:
    """
    Print a heading between features.
    """
 
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSP feature extraction over a manifest.",
    )
 
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest CSV to process, e.g. data/manifests/dev_manifest.csv.",
    )
 
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name for this experiment, e.g. 'dev_v1'. Reuse it to resume.",
    )
 
    parser.add_argument(
        "--feature",
        nargs="+",
        required=True,
        help=(
            "Feature(s) to run, e.g. 'mfcc' or 'mfcc pitch', or 'all' for "
            "every module in src/dsp/features/."
        ),
    )
 
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Folder that processed_filepath values are relative to.",
    )
 
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Folder in which run folders are created.",
    )
 
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Folder containing <feature>.json and speech_mask.json.",
    )
 
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess samples previously recorded with status='error'.",
    )
 
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop at the first sample error. When running several features, "
            "also stop at the first feature that fails."
        ),
    )
 
    return parser.parse_args()


def main():
    args = parse_args()
 
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
 
    feature_names = resolve_feature_names(args.feature)
 
    # A single feature behaves like any other script where errors are 
    # raised normally with a full traceback.
    if len(feature_names) == 1:
        run_one_feature(feature_names[0], args)
        return
 
    # Several features run one after the other. A failing feature is
    # reported and the rest still run (unless --fail-fast is set).
    print(f"Features to run: {feature_names}")
 
    summary = {}
 
    for feature_name in feature_names:
        print_section(f"Feature: {feature_name}")
 
        try:
            results = run_one_feature(feature_name, args)
 
        except Exception as error:
            summary[feature_name] = f"FAILED ({type(error).__name__}: {error})"
            print(f"\nFeature '{feature_name}' failed: {type(error).__name__}: {error}")
 
            if args.fail_fast:
                break
 
            continue
 
        n_ok = int((results["status"] == "ok").sum())
        n_errors = int((results["status"] == "error").sum())
        summary[feature_name] = f"{n_ok} ok, {n_errors} sample errors"
 
    print_section("Summary")
 
    for feature_name in feature_names:
        print(f"{feature_name}: {summary.get(feature_name, 'not run')}")
 
    failed = [
        name for name, outcome in summary.items()
        if outcome.startswith("FAILED")
    ]
 
    if failed:
        sys.exit(f"\n{len(failed)} feature(s) failed: {failed}")
 
 
if __name__ == "__main__":
    main()