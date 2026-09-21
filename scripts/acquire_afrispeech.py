import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "intronhealth/afrispeech-200"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "afrispeech_acquisition.json"
)

AFRISPEECH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "afrispeech"
)


def load_config():
    """
    Read in the AfriSpeech acquisition configuration (afrispeech_acquisition.json)
    """ 

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Acquisition config not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "selections" not in config:
        raise ValueError(
            "Acquisition config must contain 'selections'."
        )

    return config


def count_wavs(audio_dir):
    """
    Count all WAV files below an audio directory
    Used to confirm the number of WAV files equals the number of transcript CSVs
    """

    if not audio_dir.exists():
        return 0

    return sum(
        1
        for path in audio_dir.rglob("*.wav")
        if path.is_file()
    )


def acquisition_is_complete(metadata_path, audio_dir):
    """
    Check whether an existing local acquisition appears complete

    A selection is considered complete when:
    - its metadata CSV exists
    - its audio directory exists
    - the CSV contains at least one row
    - the number of WAV files equals the number of metadata rows
    """

    if not metadata_path.exists():
        return False

    if not audio_dir.exists():
        return False

    try:
        metadata_df = pd.read_csv(metadata_path)
    except Exception:
        return False

    metadata_count = len(metadata_df)
    wav_count = count_wavs(audio_dir)

    return (
        metadata_count > 0
        and metadata_count == wav_count
    )


def remove_existing_selection(metadata_path, audio_dir):
    """
    Remove an incomplete or deliberately refreshed selection
    """

    if metadata_path.exists():
        metadata_path.unlink()

    if audio_dir.exists():
        shutil.rmtree(audio_dir)


def extract_archive(archive_path, output_dir):
    """
    Extract one AfriSpeech tar.gz audio archive
    """

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(
            path=output_dir,
            filter="data",
        )


def acquire_selection(repo_files, revision, accent, split, force=False,):
    """
    Acquire one AfriSpeech accent/split combination
    """

    print()
    print(f"Acquiring: {accent} / {split}")

    # determine local location for file
    metadata_repo_path = (
        f"transcripts/{accent}/{split}.csv"
    )

    metadata_path = (
        AFRISPEECH_ROOT
        / "transcripts"
        / accent
        / f"{split}.csv"
    )

    audio_dir = (
        AFRISPEECH_ROOT
        / "audio"
        / accent
        / split
    )

    # If it was already acquired successfully, don't download it again unless --force was requested
    if (not force and acquisition_is_complete(metadata_path, audio_dir,)):
        wav_count = count_wavs(audio_dir)

        print(
            f"Already acquired: "
            f"{wav_count} recordings"
        )
        return
    
    # If not acquired previously:

    # Check that the requested metadata actually exists remotely
    if metadata_repo_path not in repo_files:
        raise FileNotFoundError(
            f"No AfriSpeech metadata found for {accent} / {split}"
        )

    # Find all archive shards belonging to this accent/split (Some selections have more than one archive)
    archive_prefix = (
        f"audio/{accent}/{split}/"
    )

    archive_repo_paths = sorted(
        file
        for file in repo_files
        if file.startswith(archive_prefix)
        and file.endswith(".tar.gz")
    )

    if not archive_repo_paths:
        raise FileNotFoundError(
            "No AfriSpeech audio archives found for "
            f"{accent} / {split}"
        )

    print(
        f"Audio archives: "
        f"{len(archive_repo_paths)}"
    )


    # If there is an incomplete old acquisition, remove it before rebuilding this particular accent/split.
    if metadata_path.exists() or audio_dir.exists():
        if force:
            print("Removing existing data (--force)")
        else:
            print("Existing data is incomplete; reacquiring")

        remove_existing_selection(
            metadata_path,
            audio_dir,
        )

    
    # Download the metadata CSV.
    downloaded_metadata_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=metadata_repo_path,
            repo_type="dataset",
            revision=revision,
            local_dir=AFRISPEECH_ROOT,
        )
    )

    metadata_df = pd.read_csv(
        downloaded_metadata_path
    )

    expected_count = len(metadata_df)

    if expected_count == 0:
        raise ValueError(
            f"Metadata contains no rows for "
            f"{accent} / {split}"
        )

    print(
        f"Metadata rows: {expected_count}"
    )

    # Create the local directory into which WAV files will be extracted
    audio_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Download and extract every archive shard
    for archive_repo_path in archive_repo_paths:
        archive_name = Path(
            archive_repo_path
        ).name

        print(
            f"Downloading: {archive_name}"
        )

        archive_path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                filename=archive_repo_path,
                repo_type="dataset",
                revision=revision,
                local_dir=AFRISPEECH_ROOT,
            )
        )

        print(
            f"Extracting: {archive_name}"
        )

        extract_archive(
            archive_path,
            audio_dir,
        )

        archive_path.unlink()

    # verification
    wav_count = count_wavs(audio_dir)

    if wav_count != expected_count:
        raise ValueError(
            f"Acquisition verification failed for "
            f"{accent} / {split}: "
            f"metadata has {expected_count} rows, "
            f"but {wav_count} WAV files were extracted."
        )

    print(
        f"Verified: {wav_count} recordings"
    )


def main(force=False):
    config = load_config()

    requested_revision = config.get(
        "revision",
        "main",
    )

    api = HfApi()

    # Resolve the requested revision once so every file downloaded during this run comes from the same dataset snapshot.
    dataset_info = api.dataset_info(
        repo_id=DATASET_ID,
        revision=requested_revision,
    )

    revision = dataset_info.sha

    print(f"Dataset: {DATASET_ID}")
    print(
        f"Requested revision: "
        f"{requested_revision}"
    )
    print(
        f"Resolved commit: {revision}"
    )

    # Get the repository file list once and reuse it for every requested accent/split.
    repo_files = set(
        api.list_repo_files(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
        )
    )

    for selection in config["selections"]:
        accent = selection["accent"]
        splits = selection["splits"]

        for split in splits:
            acquire_selection(
                repo_files=repo_files,
                revision=revision,
                accent=accent,
                split=split,
                force=force,
            )

    print()
    print("AfriSpeech acquisition complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Acquire selected AfriSpeech "
            "accent/split combinations."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-download selections even if "
            "they are already complete."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        force=args.force,
    )