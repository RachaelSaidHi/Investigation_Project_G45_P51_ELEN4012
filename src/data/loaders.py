from pathlib import Path

import pandas as pd

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
        raise FileNotFoundError("SAA metadata file not found: {metadata_path}")

    if not transcript_path.exists():
        raise FileNotFoundError(f"SAA reading passage not found: {transcript_path}")

    if not recordings_dir.exists():
        raise FileNotFoundError(f"SAA recordings directory not found: {recordings_dir}")

    # read in info
    metadata_df = pd.read_csv(metadata_path)
    transcript = transcript_path.read_text(encoding="utf-8").strip()

    valid_rows = []

    for index, row in metadata_df.iterrows():
        # check the row actually has the audio file in the dataset
        if row["file_missing?"] == True:
            continue

        # check the audio file is in the recordings directory 
        audio_path = recordings_dir / f"{row['filename']}.mp3"

        if not audio_path.exists():
            if row["filename"] not in KNOWN_MISSING_FILES:
                raise FileNotFoundError(f"Expected SAA recording not found: {audio_path}")
            continue

        valid_rows.append(row)


    # testing
    # return metadata_df
    return pd.DataFrame(valid_rows)

    