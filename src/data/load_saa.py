"""Load the Speech Accent Archive into the common manifest schema.

Reads speakers_all.csv and the reading passage from data/raw/saa/ and
creates one manifest row per recording marked as available whose MP3 exists
on disk. Two recordings listed as available but absent from the Kaggle
release (KNOWN_MISSING_FILES) are skipped; any other missing file raises an
error.

Every row shares the same reference transcript (the fixed reading passage)
and the domain "read_fixed". Duration and sample rate are read from each MP3
header. No study selection is applied, so accent_group and
processed_filepath are left blank.
"""

from pathlib import Path
from mutagen.mp3 import MP3

import pandas as pd

from .schema import MANIFEST_COLUMNS

# two files that say they have an audio file actually don't have an audio file
# so isn't an error if these two files are missing
KNOWN_MISSING_FILES = {"nicaragua", "sinhalese1"}

# convert metadata value to text for dataframe
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_saa(data_root: Path) -> pd.DataFrame:
    # define SAA paths
    saa_dir = data_root / "saa"
    metadata_path = saa_dir / "speakers_all.csv"
    transcript_path = saa_dir / "reading-passage.txt"
    recordings_dir = saa_dir / "recordings" / "recordings"
    
    # check all the files exist
    if not metadata_path.exists():
        raise FileNotFoundError(f"SAA metadata file not found: {metadata_path}")

    if not transcript_path.exists():
        raise FileNotFoundError(f"SAA reading passage not found: {transcript_path}")

    if not recordings_dir.exists():
        raise FileNotFoundError(f"SAA recordings directory not found: {recordings_dir}")

    # read in info
    metadata_df = pd.read_csv(metadata_path)
    reference_transcript = transcript_path.read_text(encoding="utf-8").strip()

    manifest_rows = []

    for index, row in metadata_df.iterrows():
        # check the row actually has the audio file in the dataset
        if row["file_missing?"]:
            continue

        if pd.isna(row["filename"]):
            raise ValueError("SAA metadata row has no filename")

        # check the audio file is in the recordings directory 
        source_sample_id = str(row["filename"])
        audio_path = recordings_dir / f"{source_sample_id}.mp3"

        if not audio_path.exists():
            if source_sample_id not in KNOWN_MISSING_FILES:
                raise FileNotFoundError(f"Expected SAA recording not found: {audio_path}")
            continue

        audio_info = MP3(audio_path)

        manifest_row = {
            "sample_id": f"saa_{source_sample_id}",
            "dataset": "saa",
            "source_split": "",
            "source_sample_id": source_sample_id,
            "filepath": audio_path.relative_to(data_root).as_posix(),
            "processed_filepath": "",
            "speaker_id": f"saa_spk_{int(row['speakerid'])}",
            "accent_raw": clean_text(row["native_language"]),
            "accent_group": "",
            "country": clean_text(row["country"]),
            "native_language": clean_text(row["native_language"]),
            "birthplace": clean_text(row["birthplace"]),
            "sex": clean_text(row["sex"]),
            "age": row["age"],
            "age_group": "",
            "age_onset": row["age_onset"],
            "duration_s": audio_info.info.length,
            "original_sample_rate": audio_info.info.sample_rate,
            "domain": "read_fixed",
            "reference_transcript": reference_transcript,
        }

        manifest_rows.append(manifest_row)


    return pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS,)

    # testing
    # return metadata_df
    # return pd.DataFrame(manifest_rows)

    