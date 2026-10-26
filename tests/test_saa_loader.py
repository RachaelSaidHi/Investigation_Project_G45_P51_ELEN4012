"""Tests for the Speech Accent Archive loader."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.load_saa import load_saa
from src.data.schema import MANIFEST_COLUMNS, REQUIRED_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
SAA_ROOT = DATA_ROOT / "saa"


@pytest.fixture(scope="module")
def saa_manifest() -> pd.DataFrame:
    """Load SAA once for the whole test module."""

    if not SAA_ROOT.exists():
        pytest.skip("SAA raw dataset is not installed")

    return load_saa(DATA_ROOT)


class TestSchema:

    def test_columns_match_manifest_schema(self, saa_manifest):
        assert saa_manifest.columns.tolist() == MANIFEST_COLUMNS

    def test_row_count(self, saa_manifest):
        # 2140 rows marked usable minus 2 known missing audio files.
        assert len(saa_manifest) == 2138

    def test_no_duplicate_sample_ids(self, saa_manifest):
        assert saa_manifest["sample_id"].is_unique

    def test_no_duplicate_speaker_ids(self, saa_manifest):
        # SAA has one recording per speaker.
        assert saa_manifest["speaker_id"].is_unique


class TestColumnConventions:

    def test_sample_ids_prefixed(self, saa_manifest):
        assert saa_manifest["sample_id"].str.startswith("saa_").all()

    def test_speaker_ids_prefixed(self, saa_manifest):
        assert saa_manifest["speaker_id"].str.startswith("saa_spk_").all()

    def test_dataset_column_is_constant(self, saa_manifest):
        assert (saa_manifest["dataset"] == "saa").all()

    def test_domain_is_read_fixed(self, saa_manifest):
        assert (saa_manifest["domain"] == "read_fixed").all()

    def test_source_split_is_blank(self, saa_manifest):
        assert (saa_manifest["source_split"] == "").all()

    def test_accent_group_is_blank(self, saa_manifest):
        assert (saa_manifest["accent_group"] == "").all()

    def test_processed_filepath_is_blank(self, saa_manifest):
        assert (saa_manifest["processed_filepath"] == "").all()

    def test_age_group_is_blank(self, saa_manifest):
        assert (saa_manifest["age_group"] == "").all()


class TestRequiredColumns:

    def test_required_columns_not_null(self, saa_manifest):
        for column in REQUIRED_COLUMNS:
            assert not saa_manifest[column].isna().any(), (
                f"Required column '{column}' contains NaN values"
            )

    def test_required_columns_not_empty_string(self, saa_manifest):
        for column in REQUIRED_COLUMNS:
            if saa_manifest[column].dtype == object:
                empty = (
                    saa_manifest[column]
                    .astype(str)
                    .str.strip()
                    .eq("")
                )

                assert not empty.any(), (
                    f"Required column '{column}' contains empty strings"
                )


class TestAudioMetadata:

    def test_all_durations_positive(self, saa_manifest):
        assert (saa_manifest["duration_s"] > 0).all()

    def test_all_sample_rates_positive(self, saa_manifest):
        assert (saa_manifest["original_sample_rate"] > 0).all()

    def test_all_filepaths_exist(self, saa_manifest):
        missing = []

        for filepath in saa_manifest["filepath"]:
            if not (DATA_ROOT / filepath).exists():
                missing.append(filepath)

        assert missing == [], (
            f"Missing files: {missing[:5]}"
        )


class TestTranscript:

    def test_all_rows_share_one_transcript(self, saa_manifest):
        assert saa_manifest["reference_transcript"].nunique() == 1

    def test_transcript_starts_with_expected_text(self, saa_manifest):
        transcript = (
            saa_manifest["reference_transcript"]
            .iloc[0]
            .lower()
        )

        assert transcript.startswith("please call stella")


class TestKnownSpeakers:

    @staticmethod
    def _get_row(saa_manifest, source_id):
        rows = saa_manifest.loc[
            saa_manifest["source_sample_id"] == source_id
        ]

        assert len(rows) == 1, (
            f"Expected exactly 1 row for '{source_id}', "
            f"got {len(rows)}"
        )

        return rows.iloc[0]

    def test_yoruba_speaker(self, saa_manifest):
        row = self._get_row(saa_manifest, "yoruba2")

        assert row["country"] == "nigeria"
        assert row["native_language"] == "yoruba"
        assert row["accent_raw"] == "yoruba"
        assert row["sex"] == "female"
        assert row["age"] == 40
        assert row["domain"] == "read_fixed"
        assert row["duration_s"] > 0

    def test_american_english_speaker(self, saa_manifest):
        row = self._get_row(saa_manifest, "english1")

        assert row["country"] == "usa"
        assert row["native_language"] == "english"
        assert row["accent_raw"] == "english"
        assert row["sex"] == "male"
        assert row["age"] == 42
        assert row["duration_s"] > 0

    def test_kenyan_speaker(self, saa_manifest):
        row = self._get_row(saa_manifest, "gusii1")

        assert row["country"] == "kenya"
        assert row["native_language"] == "gusii"
        assert row["accent_raw"] == "gusii"
        assert row["sex"] == "male"
        assert row["age"] == 44
        assert row["duration_s"] > 0

    def test_south_african_speaker(self, saa_manifest):
        row = self._get_row(saa_manifest, "afrikaans1")

        assert row["country"] == "south africa"
        assert row["native_language"] == "afrikaans"
        assert row["accent_raw"] == "afrikaans"
        assert row["sex"] == "female"
        assert row["age"] == 27
        assert row["duration_s"] > 0