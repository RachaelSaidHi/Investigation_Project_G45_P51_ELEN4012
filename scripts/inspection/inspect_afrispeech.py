"""Exploratory inspection of the AfriSpeech repository on Hugging Face.

Summarises the repository structure (accent-specific and combined transcript
CSVs and audio archives), checks that each accent/split has both metadata
and audio, and checks three sample metadata files for a consistent schema,
missing values and valid audio paths. Also reports the size of the small
Afemai test split used to trial the acquisition. Used to design the
acquisition script and loader; not part of the pipeline.

Run from project root:
    python -m scripts.inspection.inspect_afrispeech
"""

from collections import Counter

import pandas as pd
from huggingface_hub import HfApi, get_paths_info, hf_hub_download


DATASET_ID = "intronhealth/afrispeech-200"

SAMPLE_FILES = [
    "transcripts/afrikaans/train.csv",
    "transcripts/akan-fante/dev.csv",
    "transcripts/afemai/test.csv",
]

KEY_COLUMNS = [
    "user_ids",
    "accent",
    "country",
    "transcript",
    "audio_ids",
    "audio_paths",
    "duration",
    "domain",
    "split",
]

TEST_ACCENT = "afemai"
TEST_SPLIT = "test"


def format_size(size_bytes):
    """Convert bytes to MB or GB."""

    size_mb = size_bytes / (1024 ** 2)

    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"

    return f"{size_mb:.2f} MB"


def main():
    api = HfApi()

    print("Getting AfriSpeech repository file list...")

    files = api.list_repo_files(
        repo_id=DATASET_ID,
        repo_type="dataset",
    )

    # Separate accent-specific metadata from combined metadata.
    #
    # Accent-specific:
    # transcripts/afrikaans/train.csv
    #
    # Combined:
    # transcripts/train.csv

    accent_transcript_files = []
    combined_transcript_files = []

    for file in files:
        if not (
            file.startswith("transcripts/")
            and file.endswith(".csv")
        ):
            continue

        parts = file.split("/")

        if len(parts) == 3:
            accent_transcript_files.append(file)

        elif len(parts) == 2:
            combined_transcript_files.append(file)

    
    # Do the same for audio.
    #
    # Accent-specific:
    # audio/afrikaans/train/train_afrikaans_0.tar.gz
    #
    # Combined:
    # audio/train/train_0.tar.gz

    accent_audio_archives = []
    combined_audio_archives = []

    for file in files:
        if not (
            file.startswith("audio/")
            and file.endswith(".tar.gz")
        ):
            continue

        parts = file.split("/")

        if len(parts) == 4:
            accent_audio_archives.append(file)

        elif len(parts) == 3:
            combined_audio_archives.append(file)

    print("\nREPOSITORY STRUCTURE")
    print("--------------------")

    print(
        f"Accent-specific transcript CSVs: "
        f"{len(accent_transcript_files)}"
    )

    print(
        f"Combined transcript CSVs: "
        f"{len(combined_transcript_files)}"
    )

    print(
        f"Accent-specific audio archives: "
        f"{len(accent_audio_archives)}"
    )

    print(
        f"Combined audio archives: "
        f"{len(combined_audio_archives)}"
    )

    
    # Check accent/split coverage.
    transcript_pairs = set()

    for file in accent_transcript_files:
        parts = file.split("/")

        accent = parts[1]
        split = parts[2].removesuffix(".csv")

        transcript_pairs.add((accent, split))

    audio_pairs = set()

    for file in accent_audio_archives:
        parts = file.split("/")

        accent = parts[1]
        split = parts[2]

        audio_pairs.add((accent, split))

    accents = sorted(
        {
            accent
            for accent, split in transcript_pairs
        }
    )

    split_counts = Counter(
        split
        for accent, split in transcript_pairs
    )

    print(f"Accent folders: {len(accents)}")
    print(
        f"Transcript files by split: "
        f"{dict(split_counts)}"
    )

    missing_audio = sorted(
        transcript_pairs - audio_pairs
    )

    extra_audio = sorted(
        audio_pairs - transcript_pairs
    )

    print(
        "Transcript accent/splits with no audio archive: "
        f"{len(missing_audio)}"
    )

    print(
        "Audio accent/splits with no transcript CSV: "
        f"{len(extra_audio)}"
    )

    if missing_audio:
        print("\nFirst unmatched transcript groups:")
        print(missing_audio[:10])

    if extra_audio:
        print("\nFirst unmatched audio groups:")
        print(extra_audio[:10])



    # Check three representative metadata files.
    print("\nSAMPLED METADATA CHECK")
    print("----------------------")

    reference_columns = None
    sampled_audio_ids = []

    for file in SAMPLE_FILES:
        local_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=file,
            repo_type="dataset",
        )

        df = pd.read_csv(local_path)

        print(f"\n{file}")
        print(f"Rows: {len(df)}")

        if reference_columns is None:
            reference_columns = df.columns.tolist()
            print("Schema: reference")

        elif df.columns.tolist() == reference_columns:
            print("Schema: matches reference")

        else:
            print("Schema: DOES NOT match reference")

        missing_columns = [
            column
            for column in KEY_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            print(
                f"Missing source columns: "
                f"{missing_columns}"
            )
            continue

        print("Required source columns: present")

        for column in KEY_COLUMNS:
            missing_count = int(
                df[column].isna().sum()
            )

            blank_count = int(
                df[column]
                .map(
                    lambda value:
                    isinstance(value, str)
                    and value.strip() == ""
                )
                .sum()
            )

            if missing_count > 0 or blank_count > 0:
                print(
                    f"  {column}: "
                    f"{missing_count} missing, "
                    f"{blank_count} blank"
                )

        print(
            "audio_ids unique within file: "
            f"{df['audio_ids'].is_unique}"
        )

        sampled_audio_ids.extend(
            df["audio_ids"].dropna().tolist()
        )

        wav_paths_valid = (
            df["audio_paths"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.endswith(".wav")
            .all()
        )

        print(
            f"audio_paths end in .wav: "
            f"{wav_paths_valid}"
        )

        print(
            f"Accent values: "
            f"{df['accent'].dropna().unique().tolist()}"
        )

        print(
            f"Country values: "
            f"{df['country'].dropna().unique().tolist()}"
        )

        print(
            f"Split values: "
            f"{df['split'].dropna().unique().tolist()}"
        )

        print(
            f"Domain values: "
            f"{df['domain'].dropna().unique().tolist()}"
        )

    sampled_audio_ids = pd.Series(
        sampled_audio_ids
    )

    duplicate_ids = sampled_audio_ids[
        sampled_audio_ids.duplicated(keep=False)
    ]

    print(
        "\nDuplicate audio_ids across sampled files: "
        f"{duplicate_ids.nunique()}"
    )

    # Check the size of the tiny Afemai test subset.
    print("\nTEST ACQUISITION SUBSET")
    print("-----------------------")

    test_metadata = (
        f"transcripts/{TEST_ACCENT}/"
        f"{TEST_SPLIT}.csv"
    )

    test_archives = [
        file
        for file in accent_audio_archives
        if file.startswith(
            f"audio/{TEST_ACCENT}/{TEST_SPLIT}/"
        )
    ]

    print(f"Metadata: {test_metadata}")
    print(f"Audio archives: {len(test_archives)}")

    for archive in test_archives:
        print(f"  {archive}")

    if test_archives:
        archive_info = get_paths_info(
            repo_id=DATASET_ID,
            paths=test_archives,
            repo_type="dataset",
        )

        total_size = sum(
            item.size
            for item in archive_info
        )

        print(
            "Total compressed audio size: "
            f"{format_size(total_size)}"
        )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()

###############################
##### Test 3

# from huggingface_hub import hf_hub_download
# import pandas as pd


# DATASET_ID = "intronhealth/afrispeech-200"

# METADATA_FILES = [
#     "transcripts/afrikaans/train.csv",
#     "transcripts/akan-fante/dev.csv",
#     "transcripts/afemai/test.csv",
# ]


# def main():
#     reference_columns = None

#     for file in METADATA_FILES:
#         print(f"\nChecking: {file}")

#         # Download/cache only this CSV file.
#         local_path = hf_hub_download(
#             repo_id=DATASET_ID,
#             filename=file,
#             repo_type="dataset",
#         )

#         # Read the metadata.
#         metadata_df = pd.read_csv(local_path)

#         print(f"Rows: {len(metadata_df)}")
#         print(f"Columns: {metadata_df.columns.tolist()}")

#         # Use the first file as the schema we compare the others against.
#         if reference_columns is None:
#             reference_columns = metadata_df.columns.tolist()
#             print("Schema: reference")
#         else:
#             if metadata_df.columns.tolist() == reference_columns:
#                 print("Schema: matches reference")
#             else:
#                 print("Schema: DOES NOT match reference")

#         # Show one row so we can also see that the values look sensible.
#         print("\nExample row:")
#         print(metadata_df.head(1).to_string(index=False))


# if __name__ == "__main__":
#     main()


#######################################
#### TEST 2 ########



# from huggingface_hub import hf_hub_download
# import pandas as pd


# DATASET_ID = "intronhealth/afrispeech-200"
# METADATA_FILE = "transcripts/afrikaans/train.csv"


# def main():
#     # Download one AfriSpeech metadata CSV into the Hugging Face cache
#     metadata_path = hf_hub_download(
#         repo_id=DATASET_ID,
#         filename=METADATA_FILE,
#         repo_type="dataset",
#     )

#     # Read the CSV into a pandas DataFrame.
#     metadata_df = pd.read_csv(metadata_path)

#     print(f"Metadata file: {METADATA_FILE}")
#     print(f"Rows: {len(metadata_df)}")

#     print("\nColumns:")
#     for column in metadata_df.columns:
#         print(column)

#     print("\nFirst 3 rows:")
#     print(metadata_df.head(3).to_string())


# if __name__ == "__main__":
#     main()




####################################
##### TEST 1 #####


# from huggingface_hub import HfApi


# DATASET_ID = "intronhealth/afrispeech-200"


# def main():
#     # Create an object that lets us ask Hugging Face
#     # for information about repositories.
#     api = HfApi()

#     # Ask Hugging Face for a list of every file
#     # inside the AfriSpeech dataset repository.
#     files = api.list_repo_files(
#         repo_id=DATASET_ID,
#         repo_type="dataset",
#     )

#     # We only care about CSV files in the transcripts folder.
#     transcript_files = [
#         file
#         for file in files
#         if file.startswith("transcripts/")
#         and file.endswith(".csv")
#     ]

#     print(f"Found {len(transcript_files)} transcript CSV files.")

#     print("\nFirst 20 transcript files:")

#     for file in transcript_files[:20]:
#         print(file)


# if __name__ == "__main__":
#     main()