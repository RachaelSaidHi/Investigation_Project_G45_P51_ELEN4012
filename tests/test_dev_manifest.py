"""Tests for the development manifest construction."""

import pandas as pd
import pytest

import scripts.build_dev_manifest as dev_builder
from scripts.build_dev_manifest import build_dev_manifest
from src.data.schema import MANIFEST_COLUMNS


def make_manifest_row(**overrides):
    """Return a valid synthetic study-manifest row with optional overrides."""

    defaults = {
        "sample_id": "saa_test",
        "dataset": "saa",
        "source_split": "",
        "source_sample_id": "test",
        "filepath": "saa/recordings/test.mp3",
        "speaker_id": "saa_spk_test",
        "accent_raw": "english",
        "accent_group": "american_english",
        "country": "usa",
        "native_language": "english",
        "birthplace": "",
        "sex": "female",
        "age": 30,
        "age_group": "",
        "age_onset": 0,
        "duration_s": 20.0,
        "original_sample_rate": 44100,
        "snr_est": "",
        "domain": "read_fixed",
        "reference_transcript": "please call stella",
    }

    defaults.update(overrides)
    return defaults


def make_manifest(rows):
    """Build a manifest DataFrame from synthetic rows."""

    return pd.DataFrame(
        rows,
        columns=MANIFEST_COLUMNS,
    )


@pytest.fixture
def study_manifest():
    """Create two accent groups with multiple female and male speakers."""

    rows = [
        make_manifest_row(
            sample_id="saa_amharic_f2",
            speaker_id="saa_spk_1",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_amharic_f1",
            speaker_id="saa_spk_2",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_amharic_m2",
            speaker_id="saa_spk_3",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="male",
        ),
        make_manifest_row(
            sample_id="saa_amharic_m1",
            speaker_id="saa_spk_4",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="male",
        ),
        make_manifest_row(
            sample_id="saa_yoruba_f2",
            speaker_id="saa_spk_5",
            accent_group="yoruba",
            native_language="yoruba",
            country="nigeria",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_yoruba_f1",
            speaker_id="saa_spk_6",
            accent_group="yoruba",
            native_language="yoruba",
            country="nigeria",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_yoruba_m2",
            speaker_id="saa_spk_7",
            accent_group="yoruba",
            native_language="yoruba",
            country="nigeria",
            sex="male",
        ),
        make_manifest_row(
            sample_id="saa_yoruba_m1",
            speaker_id="saa_spk_8",
            accent_group="yoruba",
            native_language="yoruba",
            country="nigeria",
            sex="male",
        ),
    ]

    return make_manifest(rows)


class TestSelection:

    def test_selects_requested_number_per_sex_and_group(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(study_manifest)

        counts = (
            result
            .groupby(["accent_group", "sex"])
            .size()
        )

        assert (counts == dev_builder.SPEAKERS_PER_SEX).all()

    def test_selection_is_deterministic(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(study_manifest)

        assert result["sample_id"].tolist() == [
            "saa_amharic_f1",
            "saa_amharic_m1",
            "saa_yoruba_f1",
            "saa_yoruba_m1",
        ]

    def test_speakers_per_sex_is_configurable(
        self,
        study_manifest,
        monkeypatch,
    ):
        monkeypatch.setattr(
            dev_builder,
            "SPEAKERS_PER_SEX",
            2,
        )

        result = build_dev_manifest(study_manifest)

        counts = (
            result
            .groupby(["accent_group", "sex"])
            .size()
        )

        assert (counts == 2).all()
        assert len(result) == 8


class TestLimitedAvailability:

    def test_fewer_available_speakers_warns_and_keeps_available(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            dev_builder,
            "SPEAKERS_PER_SEX",
            2,
        )

        manifest = make_manifest([
            make_manifest_row(
                sample_id="saa_yoruba_f1",
                speaker_id="saa_spk_1",
                accent_group="yoruba",
                native_language="yoruba",
                country="nigeria",
                sex="female",
            ),
            make_manifest_row(
                sample_id="saa_yoruba_m1",
                speaker_id="saa_spk_2",
                accent_group="yoruba",
                native_language="yoruba",
                country="nigeria",
                sex="male",
            ),
            make_manifest_row(
                sample_id="saa_yoruba_m2",
                speaker_id="saa_spk_3",
                accent_group="yoruba",
                native_language="yoruba",
                country="nigeria",
                sex="male",
            ),
        ])

        with pytest.warns(
            UserWarning,
            match="has only 1 female speaker",
        ):
            result = build_dev_manifest(manifest)

        female = result[
            result["sex"] == "female"
        ]

        male = result[
            result["sex"] == "male"
        ]

        assert len(female) == 1
        assert len(male) == 2


class TestContract:

    def test_selected_rows_come_from_study_manifest(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(study_manifest)

        assert set(result["sample_id"]).issubset(
            set(study_manifest["sample_id"])
        )

    def test_all_manifest_columns_are_preserved(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(study_manifest)

        assert result.columns.tolist() == MANIFEST_COLUMNS


# run using:
# python -m pytest tests/test_dev_manifest.py -v