import pandas as pd
import pytest

from src.data.schema import MANIFEST_COLUMNS
from src.data.validation import validate_manifest


def make_valid_manifest(tmp_path):
    """Create a minimal valid manifest with a matching fake audio file."""

    audio_dir = tmp_path / "saa"
    audio_dir.mkdir()

    audio_path = audio_dir / "example.mp3"
    audio_path.touch()

    row = {
        "sample_id": "saa_example",
        "dataset": "saa",
        "source_split": "",
        "source_sample_id": "example",
        "filepath": "saa/example.mp3",
        "speaker_id": "saa_spk_1",
        "accent_raw": "english",
        "accent_group": "",
        "country": "usa",
        "native_language": "english",
        "birthplace": "",
        "sex": "female",
        "age": 25,
        "age_group": "",
        "age_onset": 0,
        "duration_s": 10.0,
        "original_sample_rate": 44100,
        "snr_est": "",
        "domain": "read_fixed",
        "reference_transcript": "example transcript",
    }

    return pd.DataFrame(
        [row],
        columns=MANIFEST_COLUMNS,
    )


def test_valid_manifest_passes(tmp_path):
    manifest = make_valid_manifest(tmp_path)

    validate_manifest(manifest, tmp_path)


def test_duplicate_sample_ids_fail(tmp_path):
    manifest = make_valid_manifest(tmp_path)

    manifest = pd.concat(
        [manifest, manifest],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        validate_manifest(manifest, tmp_path)


def test_missing_required_value_fails(tmp_path):
    manifest = make_valid_manifest(tmp_path)

    manifest.loc[0, "speaker_id"] = ""

    with pytest.raises(
        ValueError,
        match="Required column contains empty values",
    ):
        validate_manifest(manifest, tmp_path)


def test_missing_filepath_fails(tmp_path):
    manifest = make_valid_manifest(tmp_path)

    manifest.loc[0, "filepath"] = "saa/does_not_exist.mp3"

    with pytest.raises(FileNotFoundError):
        validate_manifest(manifest, tmp_path)