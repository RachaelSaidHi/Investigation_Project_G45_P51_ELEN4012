"""Tests for final analysis manifest construction."""

import pandas as pd
import pytest

import scripts.data.build_analysis_manifest as analysis_builder

from scripts.data.build_analysis_manifest import (
    build_analysis_manifest,
    check_processed_files_exist,
)
from src.data.schema import MANIFEST_COLUMNS


def make_manifest_row(
    sample_id,
    dataset,
    speaker_id,
    accent_group,
    sex="female",
):
    """Return a synthetic study-manifest row."""

    row = {
        column: ""
        for column in MANIFEST_COLUMNS
    }

    is_afrispeech = dataset == "afrispeech"

    row.update({
        "sample_id": sample_id,
        "dataset": dataset,
        "source_split": "test" if is_afrispeech else "",
        "source_sample_id": sample_id,
        "filepath": (
            f"afrispeech/audio/{sample_id}.wav"
            if is_afrispeech
            else f"saa/recordings/{sample_id}.mp3"
        ),
        "speaker_id": speaker_id,
        "accent_raw": accent_group,
        "accent_group": accent_group,
        "sex": sex,
        "age": "" if is_afrispeech else 28,
        "age_group": "19-25" if is_afrispeech else "",
        "duration_s": 12.0,
        "original_sample_rate": 44100,
        "domain": "general" if is_afrispeech else "read_fixed",
        "reference_transcript": "test transcript",
    })

    return row


@pytest.fixture
def study_manifest():
    """Create SAA and AfriSpeech recordings before QC selection."""

    rows = [
        # SAA: two passing recordings and one excluded recording.
        make_manifest_row(
            "saa_f1", "saa", "saa_spk_1", "amharic", "female"
        ),
        make_manifest_row(
            "saa_m1", "saa", "saa_spk_2", "amharic", "male"
        ),
        make_manifest_row(
            "saa_bad", "saa", "saa_spk_3", "amharic", "female"
        ),

        # AfriSpeech: the first ID has one failing and one passing file.
        make_manifest_row(
            "afri_h1_bad", "afrispeech", "afri_spk_1", "hausa"
        ),
        make_manifest_row(
            "afri_h1_good", "afrispeech", "afri_spk_1", "hausa"
        ),

        # The second ID has two passing files.
        make_manifest_row(
            "afri_h2_a", "afrispeech", "afri_spk_2", "hausa", "male"
        ),
        make_manifest_row(
            "afri_h2_b", "afrispeech", "afri_spk_2", "hausa", "male"
        ),

        # One ID has no passing files, while another has one.
        make_manifest_row(
            "afri_y1_bad", "afrispeech", "afri_spk_3", "yoruba"
        ),
        make_manifest_row(
            "afri_y2_good", "afrispeech", "afri_spk_4", "yoruba"
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=MANIFEST_COLUMNS,
    )


@pytest.fixture
def quality_table(study_manifest):
    """Create QC results containing passing and excluded recordings."""

    excluded_ids = {
        "saa_bad",
        "afri_h1_bad",
        "afri_y1_bad",
    }

    rows = []

    for row in study_manifest.itertuples(index=False):

        excluded = row.sample_id in excluded_ids

        rows.append({
            "sample_id": row.sample_id,
            "dataset": row.dataset,
            "accent_group": row.accent_group,
            "speaker_id": row.speaker_id,
            "sex": row.sex,
            "domain": row.domain,
            "processed_filepath": f"audio/{row.sample_id}.wav",
            "error": "",
            "processed_duration_s": 2.0 if excluded else 12.0,
            "hf_energy_pct": 1.0,
            "wada_snr_db": 25.0,
            "wada_snr_saturated": False,
            "exclude": "True" if excluded else "False",
            "exclude_reason": (
                "duration_too_short" if excluded else ""
            ),
        })

    return pd.DataFrame(rows)


class TestSelection:

    def test_all_passing_saa_recordings_are_retained(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        saa = result[result["dataset"] == "saa"]

        assert set(saa["sample_id"]) == {
            "saa_f1",
            "saa_m1",
        }

    def test_selects_one_recording_per_afrispeech_id(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        afrispeech = result[
            result["dataset"] == "afrispeech"
        ]

        assert len(afrispeech) == 3
        assert afrispeech["speaker_id"].is_unique

    def test_qc_is_applied_before_afrispeech_selection(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        # The first ID must survive through its passing recording.
        assert "afri_h1_good" in result["sample_id"].values
        assert "afri_h1_bad" not in result["sample_id"].values

    def test_ids_with_no_passing_recordings_are_removed(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        assert "afri_spk_3" not in result["speaker_id"].values

    def test_no_excluded_recordings_are_selected(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        excluded_ids = set(
            quality_table.loc[
                quality_table["exclude"] == "True",
                "sample_id",
            ]
        )

        assert excluded_ids.isdisjoint(
            set(result["sample_id"])
        )

    def test_selection_is_reproducible(
        self,
        study_manifest,
        quality_table,
    ):
        shuffled_study = study_manifest.sample(
            frac=1,
            random_state=10,
        )

        shuffled_quality = quality_table.sample(
            frac=1,
            random_state=20,
        )

        first = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        second = build_analysis_manifest(
            shuffled_study,
            shuffled_quality,
        )

        pd.testing.assert_frame_equal(first, second)


class TestManifestContract:

    def test_manifest_columns_are_preserved(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        assert result.columns.tolist() == MANIFEST_COLUMNS

    def test_selected_rows_come_from_study_manifest(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        assert set(result["sample_id"]).issubset(
            set(study_manifest["sample_id"])
        )

    def test_processed_filepaths_are_populated(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        assert (
            result["processed_filepath"]
            == "audio/" + result["sample_id"] + ".wav"
        ).all()

    def test_original_metadata_is_preserved(
        self,
        study_manifest,
        quality_table,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        original = study_manifest.set_index("sample_id")
        selected = result.set_index("sample_id")

        for column in MANIFEST_COLUMNS:

            if column in {"sample_id", "processed_filepath"}:
                continue

            assert (
                selected[column]
                == original.loc[selected.index, column]
            ).all()

    def test_inputs_are_not_modified(
        self,
        study_manifest,
        quality_table,
    ):
        original_study = study_manifest.copy(deep=True)
        original_quality = quality_table.copy(deep=True)

        build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        pd.testing.assert_frame_equal(
            study_manifest,
            original_study,
        )

        pd.testing.assert_frame_equal(
            quality_table,
            original_quality,
        )


class TestInvalidInputs:

    def test_missing_manifest_column_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        study_manifest = study_manifest.drop(
            columns="reference_transcript"
        )

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_missing_qc_column_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        quality_table = quality_table.drop(
            columns="exclude"
        )

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_duplicate_qc_sample_ids_raise_error(
        self,
        study_manifest,
        quality_table,
    ):
        duplicate = quality_table.iloc[[0]]

        quality_table = pd.concat(
            [quality_table, duplicate],
            ignore_index=True,
        )

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_missing_qc_record_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        quality_table = quality_table[
            quality_table["sample_id"] != "saa_f1"
        ]

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_extra_qc_record_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        extra = quality_table.iloc[[0]].copy()
        extra["sample_id"] = "unknown_recording"

        quality_table = pd.concat(
            [quality_table, extra],
            ignore_index=True,
        )

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_preprocessing_error_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        quality_table.loc[
            quality_table["sample_id"] == "saa_f1",
            "error",
        ] = "Audio decoding failed"

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_invalid_exclusion_value_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        quality_table.loc[
            quality_table["sample_id"] == "saa_f1",
            "exclude",
        ] = "unknown"

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )

    def test_missing_processed_filepath_raises_error(
        self,
        study_manifest,
        quality_table,
    ):
        quality_table.loc[
            quality_table["sample_id"] == "saa_f1",
            "processed_filepath",
        ] = ""

        with pytest.raises(ValueError):
            build_analysis_manifest(
                study_manifest,
                quality_table,
            )


class TestProcessedFiles:

    def test_missing_processed_file_raises_error(
        self,
        study_manifest,
        quality_table,
        monkeypatch,
        tmp_path,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        monkeypatch.setattr(
            analysis_builder,
            "PROCESSED_ROOT",
            tmp_path,
        )

        with pytest.raises(FileNotFoundError):
            check_processed_files_exist(result)

    def test_existing_processed_files_pass(
        self,
        study_manifest,
        quality_table,
        monkeypatch,
        tmp_path,
    ):
        result = build_analysis_manifest(
            study_manifest,
            quality_table,
        )

        monkeypatch.setattr(
            analysis_builder,
            "PROCESSED_ROOT",
            tmp_path,
        )

        for filepath in result["processed_filepath"]:

            path = tmp_path / filepath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        check_processed_files_exist(result)


class TestMain:

    def test_builds_analysis_manifest_and_reapplies_qc(
        self,
        study_manifest,
        quality_table,
        monkeypatch,
        tmp_path,
    ):
        raw_root = tmp_path / "raw"
        processed_root = tmp_path / "processed"

        study_path = tmp_path / "study_manifest.csv"
        qc_path = tmp_path / "audio_qc_study_manifest.csv"
        output_path = tmp_path / "analysis_manifest.csv"

        # Create dummy files so path validation can run.
        for filepath in study_manifest["filepath"]:

            path = raw_root / filepath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        for filepath in quality_table["processed_filepath"]:

            path = processed_root / filepath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        # Deliberately make a stored exclusion flag incorrect.
        # main() should recompute exclusions from the measurements.
        quality_table.loc[
            quality_table["sample_id"] == "saa_bad",
            "exclude",
        ] = "False"

        study_manifest.to_csv(
            study_path,
            index=False,
        )

        quality_table.to_csv(
            qc_path,
            index=False,
        )

        monkeypatch.setattr(
            analysis_builder,
            "RAW_ROOT",
            raw_root,
        )

        monkeypatch.setattr(
            analysis_builder,
            "PROCESSED_ROOT",
            processed_root,
        )

        monkeypatch.setattr(
            analysis_builder,
            "STUDY_MANIFEST",
            study_path,
        )

        monkeypatch.setattr(
            analysis_builder,
            "QC_TABLE",
            qc_path,
        )

        monkeypatch.setattr(
            analysis_builder,
            "OUTPUT_PATH",
            output_path,
        )

        analysis_builder.main()

        result = pd.read_csv(
            output_path,
            keep_default_na=False,
        )

        assert len(result) == 5
        assert "saa_bad" not in result["sample_id"].values
        assert "afri_h1_good" in result["sample_id"].values
        assert result.columns.tolist() == MANIFEST_COLUMNS


# run using:
# python -m pytest tests/test_analysis_manifest.py -v