"""Load locally acquired AfriSpeech data into the common manifest schema.

Discovers all transcript CSVs under data_root/afrispeech/transcripts/
and matches each metadata row to its corresponding extracted WAV file.
Only processes whatever the acquisition script has downloaded.
"""

import wave
from pathlib import Path, PurePosixPath

import pandas as pd

from .schema import MANIFEST_COLUMNS


# -----------------------------------------------------------------------
# AfriSpeech uses ISO 3166 alpha-2 country codes. The manifest uses
# lower-case country names for consistency with SAA.
#
# Only codes that have been verified against actual AfriSpeech metadata
# are included. Unknown codes raise an error rather than passing through
# silently, so new codes are caught immediately.
# -----------------------------------------------------------------------

AFRISPEECH_COUNTRY_MAP = {
    "NG": "nigeria",
    "KE": "kenya",
    "ZA": "south africa",
    "GH": "ghana",
    "BW": "botswana",
    "UG": "uganda",
    "RW": "rwanda",
    "US": "usa",
    "TR": "turkey",
    "ZW": "zimbabwe",
    "MW": "malawi",
    "TZ": "tanzania",
    "LS": "lesotho",
}


# Columns the loader requires from each AfriSpeech transcript CSV.
REQUIRED_SOURCE_COLUMNS = {
    "user_ids",
    "accent",
    "age_group",
    "country",
    "transcript",
    "audio_ids",
    "audio_paths",
    "domain",
    "split",
    "gender",
}


def _clean_text(value) -> str:
    """Convert a metadata value to a clean string."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def _normalise_country(code: str) -> str:
    """Convert an AfriSpeech country code to a lower-case country name."""

    code = _clean_text(code).upper()

    if code not in AFRISPEECH_COUNTRY_MAP:
        raise ValueError(
            f"Unknown AfriSpeech country code: '{code}'. "
            f"Add it to AFRISPEECH_COUNTRY_MAP."
        )

    return AFRISPEECH_COUNTRY_MAP[code]


def _read_wav_info(wav_path: Path) -> tuple[float, int]:
    """Read duration and sample rate from a WAV file header."""

    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()

    duration_s = frame_count / sample_rate

    return duration_s, sample_rate


def _build_wav_lookup(audio_dir: Path) -> dict[str, Path]:
    """
    Index extracted WAV files by parent UUID and filename.

    AfriSpeech can contain the same WAV filename in different source
    directories, so filename alone is not guaranteed to be unique.
    """

    lookup = {}

    for wav_path in audio_dir.rglob("*.wav"):

        key = (
            wav_path.parent.name,
            wav_path.name,
        )

        if key in lookup:
            raise ValueError(
                f"Duplicate AfriSpeech audio key {key} in "
                f"{audio_dir}:\n"
                f"  {lookup[key]}\n"
                f"  {wav_path}"
            )

        lookup[key] = wav_path

    return lookup


def _load_one_selection(data_root: Path, metadata_path: Path, audio_dir: Path,) -> list[dict]:
    """Load one accent/split CSV and return a list of manifest row dicts."""

    metadata_df = pd.read_csv(metadata_path)

    if metadata_df.empty:
        raise ValueError(
            f"AfriSpeech metadata file contains no rows: {metadata_path}"
        )

    # Check that the CSV contains every column the loader needs.
    missing_columns = REQUIRED_SOURCE_COLUMNS - set(metadata_df.columns)

    if missing_columns:
        raise ValueError(
            f"{metadata_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Build a WAV lookup scoped to this accent/split only.
    wav_lookup = _build_wav_lookup(audio_dir)

    if not wav_lookup:
        raise FileNotFoundError(
            f"No WAV files found under: {audio_dir}"
        )

    rows = []
    missing_audio = []

    for _, row in metadata_df.iterrows():

        # Extract the WAV filename from the source audio path.
        audio_paths_value = _clean_text(row["audio_paths"])

        if audio_paths_value == "":
            raise ValueError(f"AfriSpeech row in {metadata_path} has no audio_paths value")

        source_audio_path = PurePosixPath(audio_paths_value)

        audio_key = (source_audio_path.parent.name, source_audio_path.name,)

        if audio_key not in wav_lookup:
            missing_audio.append(str(source_audio_path))
            continue

        wav_path = wav_lookup[audio_key]
        relative_path = wav_path.relative_to(data_root)

        duration_s, sample_rate = _read_wav_info(wav_path)

        source_sample_id = _clean_text(row["audio_ids"])
        source_speaker_id = _clean_text(row["user_ids"])

        if source_sample_id == "":
            raise ValueError(f"AfriSpeech row in {metadata_path} has no audio_id")

        if source_speaker_id == "":
            raise ValueError(f"AfriSpeech row in {metadata_path} has no user_id")

        rows.append({
            "sample_id": f"afrispeech_{source_sample_id}",
            "dataset": "afrispeech",
            "source_split": _clean_text(row["split"]),
            "source_sample_id": source_sample_id,
            "filepath": relative_path.as_posix(),
            "speaker_id": f"afrispeech_spk_{source_speaker_id}",
            "accent_raw": _clean_text(row["accent"]),
            "accent_group": "",
            "country": _normalise_country(row["country"]),
            "native_language": "",
            "birthplace": "",
            "sex": _clean_text(row["gender"]).lower(),
            "age": "",
            "age_group": _clean_text(row["age_group"]),
            "age_onset": "",
            "duration_s": duration_s,
            "original_sample_rate": sample_rate,
            "snr_est": "",
            "domain": _clean_text(row["domain"]),
            "reference_transcript": _clean_text(row["transcript"]),
        })

    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} audio file(s) referenced in "
            f"{metadata_path.name} but not found under {audio_dir}. "
            f"First few: {missing_audio[:5]}"
        )

    return rows


def load_afrispeech(data_root: Path) -> pd.DataFrame:
    """Load all locally acquired AfriSpeech data into the common manifest.

    Discovers transcript CSVs automatically — the loader does not read
    the acquisition config. It processes whatever has been downloaded.
    """

    afrispeech_root = data_root / "afrispeech"
    transcripts_root = afrispeech_root / "transcripts"
    audio_root = afrispeech_root / "audio"

    if not transcripts_root.exists():
        raise FileNotFoundError(
            f"AfriSpeech transcripts not found at: {transcripts_root}\n"
            "Run 'python -m scripts.acquire_afrispeech' first."
        )

    if not audio_root.exists():
        raise FileNotFoundError(
            f"AfriSpeech audio not found at: {audio_root}\n"
            "Run 'python -m scripts.acquire_afrispeech' first."
        )

    # Discover all acquired transcript CSVs.
    csv_paths = sorted(transcripts_root.glob("*/*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"No transcript CSVs found under: {transcripts_root}"
        )

    # Process each accent/split independently.
    all_rows = []

    for metadata_path in csv_paths:

        accent = metadata_path.parent.name
        split = metadata_path.stem

        audio_dir = audio_root / accent / split

        if not audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory not found for {accent}/{split}: "
                f"{audio_dir}"
            )

        rows = _load_one_selection(
            data_root=data_root,
            metadata_path=metadata_path,
            audio_dir=audio_dir,
        )

        print(
            f"  {accent}/{split}: "
            f"{len(rows)} recordings loaded"
        )

        all_rows.extend(rows)

    return pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)