"""Tests for the development manifest construction."""

import pandas as pd
import pytest

import scripts.data.build_dev_manifest as dev_builder
from scripts.data.build_dev_manifest import build_dev_manifest
from src.data.schema import MANIFEST_COLUMNS


def make_manifest_row(**overrides):
    """Return a valid synthetic study-manifest row with optional overrides."""

    defaults = {
        "sample_id": "saa_test",
        "dataset": "saa",
        "source_split": "",
        "source_sample_id": "test",
        "filepath": "saa/recordings/test.mp3",
        "processed_filepath": "",
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
    """Create SAA and AfriSpeech groups with multiple speakers."""

    rows = [
        # SAA Amharic.
        make_manifest_row(
            sample_id="saa_amharic_f2",
            speaker_id="saa_spk_2",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_amharic_f1",
            speaker_id="saa_spk_1",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_amharic_m2",
            speaker_id="saa_spk_4",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="male",
        ),
        make_manifest_row(
            sample_id="saa_amharic_m1",
            speaker_id="saa_spk_3",
            accent_group="amharic",
            native_language="amharic",
            country="ethiopia",
            sex="male",
        ),

        # SAA Yoruba.
        make_manifest_row(
            sample_id="saa_yoruba_f1",
            speaker_id="saa_spk_5",
            accent_group="yoruba",
            native_language="yoruba",
            country="nigeria",
            sex="female",
        ),
        make_manifest_row(
            sample_id="saa_yoruba_m1",
            speaker_id="saa_spk_6",
            accent_group="yoruba",
            native_language="yoruba",
            country="nigeria",
            sex="male",
        ),

        # AfriSpeech Yoruba.
        # Speaker 1 has two recordings to test the one-recording-per-speaker rule.
        make_manifest_row(
            sample_id="afrispeech_yoruba_f1_b",
            dataset="afrispeech",
            source_split="test",
            source_sample_id="afri_f1_b",
            filepath="afrispeech/audio/f1_b.wav",
            speaker_id="afrispeech_spk_1",
            accent_raw="yoruba",
            accent_group="yoruba",
            country="nigeria",
            native_language="",
            sex="female",
            age="",
            age_group="19-25",
            age_onset="",
            domain="general",
            reference_transcript="test transcript",
        ),
        make_manifest_row(
            sample_id="afrispeech_yoruba_f1_a",
            dataset="afrispeech",
            source_split="test",
            source_sample_id="afri_f1_a",
            filepath="afrispeech/audio/f1_a.wav",
            speaker_id="afrispeech_spk_1",
            accent_raw="yoruba",
            accent_group="yoruba",
            country="nigeria",
            native_language="",
            sex="female",
            age="",
            age_group="19-25",
            age_onset="",
            domain="general",
            reference_transcript="test transcript",
        ),
        make_manifest_row(
            sample_id="afrispeech_yoruba_f2",
            dataset="afrispeech",
            source_split="test",
            source_sample_id="afri_f2",
            filepath="afrispeech/audio/f2.wav",
            speaker_id="afrispeech_spk_2",
            accent_raw="yoruba",
            accent_group="yoruba",
            country="nigeria",
            native_language="",
            sex="female",
            age="",
            age_group="26-40",
            age_onset="",
            domain="general",
            reference_transcript="test transcript",
        ),
        make_manifest_row(
            sample_id="afrispeech_yoruba_m1",
            dataset="afrispeech",
            source_split="test",
            source_sample_id="afri_m1",
            filepath="afrispeech/audio/m1.wav",
            speaker_id="afrispeech_spk_3",
            accent_raw="yoruba",
            accent_group="yoruba",
            country="nigeria",
            native_language="",
            sex="male",
            age="",
            age_group="26-40",
            age_onset="",
            domain="general",
            reference_transcript="test transcript",
        ),
    ]

    return make_manifest(rows)


class TestSelection:

    def test_selects_requested_number_per_sex_dataset_and_group(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(
            study_manifest
        )

        counts = (
            result
            .groupby(
                [
                    "dataset",
                    "accent_group",
                    "sex",
                ]
            )
            .size()
        )

        assert (
            counts
            == dev_builder.SPEAKERS_PER_SEX
        ).all()

    def test_same_accent_from_different_datasets_is_kept_separate(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(
            study_manifest
        )

        yoruba = result[
            result["accent_group"] == "yoruba"
        ]

        assert set(
            yoruba["dataset"]
        ) == {
            "saa",
            "afrispeech",
        }

    def test_only_one_recording_per_selected_speaker(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(
            study_manifest
        )

        afrispeech = result[
            result["dataset"] == "afrispeech"
        ]

        assert (
            afrispeech["speaker_id"]
            .is_unique
        )

    def test_recording_selection_is_deterministic(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(
            study_manifest
        )

        # afrispeech_spk_1 has two recordings; the alphabetically
        # first sample_id should always be retained.
        female_yoruba = result[
            (result["dataset"] == "afrispeech")
            & (
                result["accent_group"]
                == "yoruba"
            )
            & (
                result["sex"]
                == "female"
            )
        ]

        assert (
            female_yoruba.iloc[0][
                "sample_id"
            ]
            == "afrispeech_yoruba_f1_a"
        )

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

        # Restrict this test to a group with two available
        # female and two available male speakers.
        manifest = study_manifest[
            (study_manifest["dataset"] == "saa")
            & (
                study_manifest["accent_group"]
                == "amharic"
            )
        ].copy()

        result = build_dev_manifest(manifest)

        counts = (
            result
            .groupby("sex")
            .size()
        )

        assert (counts == 2).all()
        assert len(result) == 4


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
            result = build_dev_manifest(
                manifest
            )

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
        result = build_dev_manifest(
            study_manifest
        )

        assert set(
            result["sample_id"]
        ).issubset(
            set(
                study_manifest["sample_id"]
            )
        )

    def test_all_manifest_columns_are_preserved(
        self,
        study_manifest,
    ):
        result = build_dev_manifest(
            study_manifest
        )

        assert (
            result.columns.tolist()
            == MANIFEST_COLUMNS
        )


# run using:
# python -m pytest tests/test_dev_manifest.py -v