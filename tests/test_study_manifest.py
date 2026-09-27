"""Tests for the study manifest build script."""

import pandas as pd
import pytest

import scripts.data.build_study_manifest as study_builder
from src.data.schema import MANIFEST_COLUMNS


def make_manifest_row():
    """Return a minimal synthetic manifest row."""

    return {
        "sample_id": "saa_test",
        "dataset": "saa",
        "source_split": "",
        "source_sample_id": "test",
        "filepath": "saa/recordings/test.mp3",
        "processed_filepath": "",
        "speaker_id": "saa_spk_test",
        "accent_raw": "english",
        "accent_group": "",
        "country": "usa",
        "native_language": "english",
        "birthplace": "",
        "sex": "female",
        "age": 30,
        "age_group": "",
        "age_onset": 0,
        "duration_s": 20.0,
        "original_sample_rate": 44100,
        "domain": "read_fixed",
        "reference_transcript": "please call stella",
    }


class TestLoadFullManifest:

    def test_missing_full_manifest_raises_error(self, monkeypatch, tmp_path):
        missing_path = (tmp_path / "full_manifest.csv")

        monkeypatch.setattr(
            study_builder,
            "FULL_MANIFEST",
            missing_path,
        )

        with pytest.raises(
            FileNotFoundError,
            match="Full manifest not found",
        ):
            study_builder.load_full_manifest()

    def test_full_manifest_is_loaded(self, monkeypatch, tmp_path):
        path = (tmp_path / "full_manifest.csv")

        expected = pd.DataFrame(
            [make_manifest_row()],
            columns=MANIFEST_COLUMNS,
        )

        expected.to_csv(
            path,
            index=False,
        )

        monkeypatch.setattr(
            study_builder,
            "FULL_MANIFEST",
            path,
        )

        result = (study_builder.load_full_manifest())

        assert (result.columns.tolist() == MANIFEST_COLUMNS)
        assert (len(result) == 1)
        assert (result.iloc[0]["sample_id"]== "saa_test")


# run using:
# python -m pytest tests/test_study_manifest.py -v