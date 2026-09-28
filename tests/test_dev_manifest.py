"""Tests for development manifest construction."""

import pandas as pd
import pytest

import scripts.data.build_dev_manifest as dev_builder

from scripts.data.build_dev_manifest import build_dev_manifest
from src.data.schema import MANIFEST_COLUMNS


def make_manifest_row(
    sample_id,
    dataset,
    speaker_id,
    accent_group,
    sex,
):
    """Return a synthetic analysis-manifest row."""

    row = {
        column: ""
        for column in MANIFEST_COLUMNS
    }

    row.update({
        "sample_id": sample_id,
        "dataset": dataset,
        "speaker_id": speaker_id,
        "accent_group": accent_group,
        "sex": sex,
        "processed_filepath": f"audio/{sample_id}.wav",
    })

    return row


@pytest.fixture
def analysis_manifest():
    """Create QC-passing SAA and AfriSpeech analysis recordings."""

    rows = [
        # SAA Amharic: two recordings per sex.
        make_manifest_row("saa_af1", "saa", "saa_spk_1", "amharic", "female"),
        make_manifest_row("saa_af2", "saa", "saa_spk_2", "amharic", "female"),
        make_manifest_row("saa_am1", "saa", "saa_spk_3", "amharic", "male"),
        make_manifest_row("saa_am2", "saa", "saa_spk_4", "amharic", "male"),

        # SAA Yoruba.
        make_manifest_row("saa_yf1", "saa", "saa_spk_5", "yoruba", "female"),
        make_manifest_row("saa_ym1", "saa", "saa_spk_6", "yoruba", "male"),

        # AfriSpeech Yoruba: each source ID already has one recording.
        make_manifest_row("afri_yf1", "afrispeech", "afri_spk_1", "yoruba", "female"),
        make_manifest_row("afri_yf2", "afrispeech", "afri_spk_2", "yoruba", "female"),
        make_manifest_row("afri_ym1", "afrispeech", "afri_spk_3", "yoruba", "male"),

        # AfriSpeech Kiswahili.
        make_manifest_row("afri_kf1", "afrispeech", "afri_spk_4", "kiswahili", "female"),
        make_manifest_row("afri_km1", "afrispeech", "afri_spk_5", "kiswahili", "male"),
    ]

    return pd.DataFrame(
        rows,
        columns=MANIFEST_COLUMNS,
    )


class TestSelection:

    def test_selects_one_recording_per_sex_and_group(
        self,
        analysis_manifest,
    ):
        result = build_dev_manifest(analysis_manifest)

        counts = result.groupby(
            ["dataset", "accent_group", "sex"]
        ).size()

        assert (counts == 1).all()
        assert len(result) == 8

    def test_same_accent_from_different_datasets_is_kept_separate(
        self,
        analysis_manifest,
    ):
        result = build_dev_manifest(analysis_manifest)

        yoruba = result[
            result["accent_group"] == "yoruba"
        ]

        assert set(yoruba["dataset"]) == {
            "saa",
            "afrispeech",
        }

    def test_selection_is_reproducible(
        self,
        analysis_manifest,
    ):
        shuffled = analysis_manifest.sample(
            frac=1,
            random_state=10,
        )

        first = build_dev_manifest(analysis_manifest)
        second = build_dev_manifest(shuffled)

        pd.testing.assert_frame_equal(first, second)

    def test_speakers_per_sex_is_configurable(
        self,
        analysis_manifest,
        monkeypatch,
    ):
        monkeypatch.setattr(
            dev_builder,
            "SPEAKERS_PER_SEX",
            2,
        )

        amharic = analysis_manifest[
            analysis_manifest["accent_group"] == "amharic"
        ]

        result = build_dev_manifest(amharic)

        assert result.groupby("sex").size().to_dict() == {
            "female": 2,
            "male": 2,
        }


class TestLimitedAvailability:

    def test_fewer_available_speakers_warns_and_keeps_available(
        self,
        analysis_manifest,
        monkeypatch,
    ):
        monkeypatch.setattr(
            dev_builder,
            "SPEAKERS_PER_SEX",
            2,
        )

        amharic = analysis_manifest[
            (analysis_manifest["accent_group"] == "amharic")
            & (analysis_manifest["sample_id"] != "saa_af2")
        ]

        with pytest.warns(
            UserWarning,
            match="has only 1 female speaker",
        ):
            result = build_dev_manifest(amharic)

        assert result.groupby("sex").size().to_dict() == {
            "female": 1,
            "male": 2,
        }


class TestContract:

    def test_selected_rows_come_from_analysis_manifest(
        self,
        analysis_manifest,
    ):
        result = build_dev_manifest(analysis_manifest)

        assert set(result["sample_id"]).issubset(
            set(analysis_manifest["sample_id"])
        )

    def test_processed_filepaths_are_preserved(
        self,
        analysis_manifest,
    ):
        result = build_dev_manifest(analysis_manifest)

        assert (
            result["processed_filepath"]
            == "audio/" + result["sample_id"] + ".wav"
        ).all()

    def test_all_manifest_columns_are_preserved(
        self,
        analysis_manifest,
    ):
        result = build_dev_manifest(analysis_manifest)

        assert result.columns.tolist() == MANIFEST_COLUMNS

    def test_input_manifest_is_not_modified(
        self,
        analysis_manifest,
    ):
        original = analysis_manifest.copy(deep=True)

        build_dev_manifest(analysis_manifest)

        pd.testing.assert_frame_equal(
            analysis_manifest,
            original,
        )

    def test_empty_manifest_raises_error(self):
        manifest = pd.DataFrame(
            columns=MANIFEST_COLUMNS
        )

        with pytest.raises(
            ValueError,
            match="Analysis manifest is empty",
        ):
            build_dev_manifest(manifest)


class TestMain:

    def test_builds_dev_manifest_from_analysis_manifest(
        self,
        analysis_manifest,
        monkeypatch,
        tmp_path,
    ):
        analysis_path = tmp_path / "analysis_manifest.csv"
        output_path = tmp_path / "dev_manifest.csv"

        analysis_manifest.to_csv(
            analysis_path,
            index=False,
        )

        monkeypatch.setattr(
            dev_builder,
            "ANALYSIS_MANIFEST",
            analysis_path,
        )

        monkeypatch.setattr(
            dev_builder,
            "OUTPUT_PATH",
            output_path,
        )

        dev_builder.main()

        result = pd.read_csv(
            output_path,
            keep_default_na=False,
        )

        assert len(result) == 8
        assert set(result["sample_id"]).issubset(
            set(analysis_manifest["sample_id"])
        )

    def test_missing_analysis_manifest_raises_error(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setattr(
            dev_builder,
            "ANALYSIS_MANIFEST",
            tmp_path / "missing.csv",
        )

        with pytest.raises(
            FileNotFoundError,
            match="Analysis manifest not found",
        ):
            dev_builder.main()

    def test_missing_processed_filepath_raises_error(
        self,
        analysis_manifest,
        monkeypatch,
        tmp_path,
    ):
        analysis_manifest.loc[0, "processed_filepath"] = ""

        analysis_path = tmp_path / "analysis_manifest.csv"

        analysis_manifest.to_csv(
            analysis_path,
            index=False,
        )

        monkeypatch.setattr(
            dev_builder,
            "ANALYSIS_MANIFEST",
            analysis_path,
        )

        with pytest.raises(
            ValueError,
            match="missing processed filepaths",
        ):
            dev_builder.main()


# run using:
# python -m pytest tests/test_dev_manifest.py -v