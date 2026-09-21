"""Tests for the AfriSpeech loader."""

import wave
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_afrispeech import (
    AFRISPEECH_COUNTRY_MAP,
    load_afrispeech,
    _clean_text,
    _normalise_country,
)
from src.data.schema import MANIFEST_COLUMNS, REQUIRED_COLUMNS


# ======================================================================
# Synthetic dataset helpers
# ======================================================================

def make_source_row(**overrides):
    """Return a valid synthetic AfriSpeech metadata row."""

    defaults = {
        "idx": 1,
        "user_ids": "speaker_1",
        "accent": "hausa",
        "age_group": "19-25",
        "country": "NG",
        "transcript": "the patient is stable",
        "nchars": 21,
        "audio_ids": "utterance_1",
        "audio_paths": "/AfriSpeech-100/test/uuid_a/shared.wav",
        "duration": 999.0,
        "origin": "nigerian",
        "domain": "clinical",
        "split": "test",
        "gender": "Female",
    }

    defaults.update(overrides)
    return defaults


def write_wav(wav_path, sample_rate=16000, duration_s=0.25):
    """Create a small valid mono WAV file for testing."""

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(sample_rate * duration_s)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frame_count)


def write_selection(data_root, accent, split, rows, wav_files):
    """Write one synthetic AfriSpeech accent/split selection.

    wav_files is a list of (uuid, filename, sample_rate, duration_s) tuples.
    """

    metadata_dir = data_root / "afrispeech" / "transcripts" / accent
    audio_dir = data_root / "afrispeech" / "audio" / accent / split

    metadata_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(
        metadata_dir / f"{split}.csv",
        index=False,
    )

    for uuid, filename, sample_rate, duration_s in wav_files:
        wav_path = audio_dir / "data" / "data" / "intron" / uuid / filename
        write_wav(wav_path, sample_rate=sample_rate, duration_s=duration_s)


@pytest.fixture
def afrispeech_data(tmp_path):
    """Create a small synthetic AfriSpeech dataset."""

    data_root = tmp_path / "raw"

    rows = [
        make_source_row(
            audio_ids="utterance_1",
            user_ids="speaker_1",
            audio_paths="/AfriSpeech-100/test/uuid_a/shared.wav",
            transcript="the patient is stable",
            duration=999.0,
        ),
        make_source_row(
            idx=2,
            audio_ids="utterance_2",
            user_ids="speaker_1",
            audio_paths="/AfriSpeech-100/test/uuid_b/shared.wav",
            transcript="the patient is alert",
            age_group=None,
            duration=999.0,
        ),
    ]

    wav_files = [
        ("uuid_a", "shared.wav", 16000, 0.25),
        ("uuid_b", "shared.wav", 8000, 0.50),
    ]

    write_selection(
        data_root=data_root,
        accent="hausa",
        split="test",
        rows=rows,
        wav_files=wav_files,
    )

    return data_root


# ======================================================================
# Unit tests for helper functions
# ======================================================================


class TestCleanText:

    def test_strips_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_nan_becomes_empty_string(self):
        assert _clean_text(float("nan")) == ""

    def test_none_becomes_empty_string(self):
        assert _clean_text(None) == ""

    def test_pd_na_becomes_empty_string(self):
        assert _clean_text(pd.NA) == ""

    def test_numeric_value_becomes_string(self):
        assert _clean_text(44100) == "44100"


class TestNormaliseCountry:

    def test_known_code_is_normalised(self):
        assert _normalise_country("NG") == "nigeria"

    def test_lowercase_code_is_normalised(self):
        assert _normalise_country("ng") == "nigeria"

    def test_whitespace_is_stripped(self):
        assert _normalise_country("  ZA  ") == "south africa"

    def test_all_mapped_codes_produce_lowercase_names(self):
        for code, name in AFRISPEECH_COUNTRY_MAP.items():
            result = _normalise_country(code)
            assert result == name
            assert result == result.lower()

    def test_unknown_code_raises_error(self):
        with pytest.raises(ValueError, match="Unknown AfriSpeech country code"):
            _normalise_country("XX")

    def test_empty_code_raises_error(self):
        with pytest.raises(ValueError, match="Unknown AfriSpeech country code"):
            _normalise_country("")



# ======================================================================
# Schema and structure
# ======================================================================


class TestSchema:

    def test_columns_match_manifest_schema(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert manifest.columns.tolist() == MANIFEST_COLUMNS

    def test_no_duplicate_sample_ids(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert manifest["sample_id"].is_unique

    def test_repeated_speaker_ids_are_allowed(self, afrispeech_data):
        """AfriSpeech can contain multiple utterances from the same speaker."""

        manifest = load_afrispeech(afrispeech_data)

        assert len(manifest) == 2
        assert manifest["speaker_id"].nunique() == 1

    def test_required_columns_not_empty(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        for col in REQUIRED_COLUMNS:
            assert not manifest[col].isna().any(), (
                f"Required column '{col}' contains NaN values"
            )

            empty = manifest[col].astype(str).str.strip().eq("")

            assert not empty.any(), (
                f"Required column '{col}' contains empty values"
            )


# ======================================================================
# Column conventions
# ======================================================================


class TestColumnConventions:

    def test_sample_ids_prefixed(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert manifest["sample_id"].str.startswith("afrispeech_").all()

    def test_speaker_ids_prefixed(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert manifest["speaker_id"].str.startswith("afrispeech_spk_").all()

    def test_dataset_column_is_constant(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert (manifest["dataset"] == "afrispeech").all()

    def test_source_split_is_preserved(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert (manifest["source_split"] == "test").all()

    def test_accent_group_is_blank(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert (manifest["accent_group"] == "").all()

    def test_native_language_is_blank(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert (manifest["native_language"] == "").all()

    def test_sex_is_normalised_to_lowercase(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        assert (manifest["sex"]== "female").all()


# ======================================================================
# Column mapping
# ======================================================================


class TestColumnMapping:

    @staticmethod
    def _get_row(manifest, source_sample_id):
        rows = manifest.loc[
            manifest["source_sample_id"] == source_sample_id
        ]

        assert len(rows) == 1
        return rows.iloc[0]

    def test_source_metadata_is_mapped_correctly(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)
        row = self._get_row(manifest, "utterance_1")

        assert row["sample_id"] == "afrispeech_utterance_1"
        assert row["dataset"] == "afrispeech"
        assert row["source_split"] == "test"
        assert row["speaker_id"] == "afrispeech_spk_speaker_1"
        assert row["accent_raw"] == "hausa"
        assert row["country"] == "nigeria"
        assert row["sex"] == "female"
        assert row["age_group"] == "19-25"
        assert row["domain"] == "clinical"
        assert row["reference_transcript"] == "the patient is stable"

    def test_unavailable_fields_are_blank(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)
        row = manifest.iloc[0]

        assert row["accent_group"] == ""
        assert row["native_language"] == ""
        assert row["birthplace"] == ""
        assert row["age"] == ""
        assert row["age_onset"] == ""
        assert row["snr_est"] == ""

    def test_missing_age_group_becomes_blank(self, afrispeech_data):
        """The second synthetic row has age_group=None."""

        manifest = load_afrispeech(afrispeech_data)
        row = self._get_row(manifest, "utterance_2")

        assert row["age_group"] == ""


# ======================================================================
# Audio resolution
# ======================================================================


class TestAudioResolution:

    def test_same_filename_in_different_uuids_is_supported(self, afrispeech_data):
        """Reproduces the duplicate-filename bug found in real Yoruba data."""

        manifest = load_afrispeech(afrispeech_data)

        assert len(manifest) == 2

        row_1 = manifest.loc[
            manifest["source_sample_id"] == "utterance_1"
        ].iloc[0]

        row_2 = manifest.loc[
            manifest["source_sample_id"] == "utterance_2"
        ].iloc[0]

        assert "uuid_a" in row_1["filepath"]
        assert "uuid_b" in row_2["filepath"]

    def test_filepaths_are_relative_and_exist(self, afrispeech_data):
        manifest = load_afrispeech(afrispeech_data)

        for filepath in manifest["filepath"]:
            path = Path(filepath)

            assert not path.is_absolute()
            assert (afrispeech_data / filepath).exists()

    def test_duration_and_sample_rate_come_from_wav(self, afrispeech_data):
        """CSV duration is deliberately set to 999. The manifest should
        contain the actual WAV values instead."""

        manifest = load_afrispeech(afrispeech_data)

        row_1 = manifest.loc[
            manifest["source_sample_id"] == "utterance_1"
        ].iloc[0]

        row_2 = manifest.loc[
            manifest["source_sample_id"] == "utterance_2"
        ].iloc[0]

        assert row_1["duration_s"] == pytest.approx(0.25)
        assert row_1["original_sample_rate"] == 16000

        assert row_2["duration_s"] == pytest.approx(0.50)
        assert row_2["original_sample_rate"] == 8000


# ======================================================================
# Discovery across multiple accents
# ======================================================================


class TestDiscovery:

    def test_multiple_accent_selections_are_discovered(self, afrispeech_data):
        """The loader should find all acquired accent/split CSVs automatically."""

        yoruba_rows = [
            make_source_row(
                idx=3,
                user_ids="speaker_2",
                accent="yoruba",
                audio_ids="utterance_3",
                audio_paths="/AfriSpeech-100/test/uuid_y/yoruba.wav",
                transcript="another recording",
            ),
        ]

        yoruba_wavs = [
            ("uuid_y", "yoruba.wav", 16000, 0.30),
        ]

        write_selection(
            data_root=afrispeech_data,
            accent="yoruba",
            split="test",
            rows=yoruba_rows,
            wav_files=yoruba_wavs,
        )

        manifest = load_afrispeech(afrispeech_data)

        assert set(manifest["accent_raw"]) == {"hausa", "yoruba"}
        assert len(manifest) == 3


# ======================================================================
# Invalid input
# ======================================================================


class TestInvalidInput:

    def test_missing_required_column_fails(self, afrispeech_data):
        metadata_path = (
            afrispeech_data / "afrispeech" / "transcripts" / "hausa" / "test.csv"
        )

        metadata = pd.read_csv(metadata_path)
        metadata = metadata.drop(columns=["transcript"])
        metadata.to_csv(metadata_path, index=False)

        with pytest.raises(ValueError, match="missing columns"):
            load_afrispeech(afrispeech_data)

    def test_missing_audio_file_fails(self, afrispeech_data):
        wav_path = (
            afrispeech_data
            / "afrispeech" / "audio" / "hausa" / "test"
            / "data" / "data" / "intron" / "uuid_a" / "shared.wav"
        )

        wav_path.unlink()

        with pytest.raises(FileNotFoundError):
            load_afrispeech(afrispeech_data)

    def test_unknown_country_code_fails(self, afrispeech_data):
        metadata_path = (
            afrispeech_data / "afrispeech" / "transcripts" / "hausa" / "test.csv"
        )

        metadata = pd.read_csv(metadata_path)
        metadata.loc[0, "country"] = "XX"
        metadata.to_csv(metadata_path, index=False)

        with pytest.raises(ValueError, match="Unknown AfriSpeech country code"):
            load_afrispeech(afrispeech_data)

    def test_missing_audio_id_fails(self, afrispeech_data):
        metadata_path = (
            afrispeech_data / "afrispeech" / "transcripts" / "hausa" / "test.csv"
        )

        metadata = pd.read_csv(metadata_path)
        metadata.loc[0, "audio_ids"] = None
        metadata.to_csv(metadata_path, index=False)

        with pytest.raises(ValueError, match="no audio_id"):
            load_afrispeech(afrispeech_data)

    def test_missing_user_id_fails(self, afrispeech_data):
        metadata_path = (
            afrispeech_data / "afrispeech" / "transcripts" / "hausa" / "test.csv"
        )

        metadata = pd.read_csv(metadata_path)
        metadata.loc[0, "user_ids"] = None
        metadata.to_csv(metadata_path, index=False)

        with pytest.raises(ValueError, match="no user_id"):
            load_afrispeech(afrispeech_data)


# run using:
# python -m pytest tests/test_afrispeech_loader.py -v