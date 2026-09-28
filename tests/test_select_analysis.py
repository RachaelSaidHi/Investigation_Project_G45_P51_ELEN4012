"""Tests for final analysis sample selection."""

import pandas as pd
import pytest

from src.data.select_analysis import select_analysis_sample


def make_row(
    sample_id,
    dataset,
    speaker_id,
    **overrides,
):
    """Return a synthetic manifest row."""

    row = {
        "sample_id": sample_id,
        "dataset": dataset,
        "speaker_id": speaker_id,
        "accent_group": "hausa",
        "sex": "female",
        "source_split": "test",
        "processed_filepath": f"audio/{sample_id}.wav",
    }

    row.update(overrides)

    return row


@pytest.fixture
def manifest():
    """Create SAA recordings and repeated AfriSpeech source IDs."""

    return pd.DataFrame([
        make_row("saa_1", "saa", "saa_spk_1"),
        make_row("saa_2", "saa", "saa_spk_2"),
        make_row("afri_1a", "afrispeech", "afri_user_1"),
        make_row("afri_1b", "afrispeech", "afri_user_1"),
        make_row("afri_1c", "afrispeech", "afri_user_1"),
        make_row("afri_2a", "afrispeech", "afri_user_2"),
        make_row("afri_2b", "afrispeech", "afri_user_2"),
    ])


class TestSelectAnalysisSample:

    def test_all_saa_recordings_are_retained(self, manifest):
        result = select_analysis_sample(manifest)

        saa_ids = set(
            result.loc[
                result["dataset"] == "saa",
                "sample_id",
            ]
        )

        assert saa_ids == {"saa_1", "saa_2"}

    def test_one_recording_per_afrispeech_id(self, manifest):
        result = select_analysis_sample(manifest)

        afrispeech = result[
            result["dataset"] == "afrispeech"
        ]

        assert len(afrispeech) == 2
        assert afrispeech["speaker_id"].is_unique

        assert set(afrispeech["speaker_id"]) == {
            "afri_user_1",
            "afri_user_2",
        }

    def test_selected_recordings_come_from_input(self, manifest):
        result = select_analysis_sample(manifest)

        assert set(result["sample_id"]).issubset(
            set(manifest["sample_id"])
        )

    def test_selection_is_reproducible(self, manifest):
        first = select_analysis_sample(
            manifest,
            random_seed=42,
        )

        second = select_analysis_sample(
            manifest,
            random_seed=42,
        )

        pd.testing.assert_frame_equal(first, second)

    def test_selection_does_not_depend_on_row_order(self, manifest):
        shuffled = manifest.sample(
            frac=1,
            random_state=10,
        )

        first = select_analysis_sample(manifest)
        second = select_analysis_sample(shuffled)

        pd.testing.assert_frame_equal(first, second)

    def test_selection_is_independent_of_acoustic_values(self, manifest):
        first = manifest.copy()
        second = manifest.copy()

        # These values must not influence which recording is selected.
        first["median_f0"] = range(len(first))
        first["wer"] = range(len(first))

        second["median_f0"] = first["median_f0"][::-1].to_numpy()
        second["wer"] = first["wer"][::-1].to_numpy()

        first_result = select_analysis_sample(first)
        second_result = select_analysis_sample(second)

        assert first_result["sample_id"].tolist() == (
            second_result["sample_id"].tolist()
        )

    def test_one_recording_per_id_across_splits(self):
        manifest = pd.DataFrame([
            make_row(
                "afri_test",
                "afrispeech",
                "afri_user_1",
                source_split="test",
            ),
            make_row(
                "afri_train",
                "afrispeech",
                "afri_user_1",
                source_split="train",
            ),
        ])

        result = select_analysis_sample(manifest)

        assert len(result) == 1
        assert result.iloc[0]["speaker_id"] == "afri_user_1"

    def test_original_columns_are_preserved(self, manifest):
        result = select_analysis_sample(manifest)

        assert result.columns.tolist() == manifest.columns.tolist()

        # Processed filepaths must remain associated with their
        # original recordings after selection.
        assert (
            result["processed_filepath"]
            == "audio/" + result["sample_id"] + ".wav"
        ).all()

    def test_input_manifest_is_not_modified(self, manifest):
        original = manifest.copy(deep=True)

        select_analysis_sample(manifest)

        pd.testing.assert_frame_equal(manifest, original)

    def test_saa_only_manifest(self, manifest):
        saa = manifest[
            manifest["dataset"] == "saa"
        ]

        result = select_analysis_sample(saa)

        assert set(result["sample_id"]) == set(saa["sample_id"])

    def test_afrispeech_only_manifest(self, manifest):
        afrispeech = manifest[
            manifest["dataset"] == "afrispeech"
        ]

        result = select_analysis_sample(afrispeech)

        assert len(result) == 2
        assert result["speaker_id"].is_unique

    def test_unknown_dataset_raises_error(self, manifest):
        manifest.loc[0, "dataset"] = "unknown"

        with pytest.raises(
            ValueError,
            match="Unexpected datasets",
        ):
            select_analysis_sample(manifest)

    @pytest.mark.parametrize(
        "column",
        ["sample_id", "dataset", "speaker_id"],
    )
    def test_missing_required_column_raises_error(
        self,
        manifest,
        column,
    ):
        manifest = manifest.drop(columns=column)

        with pytest.raises(
            ValueError,
            match="Missing required columns",
        ):
            select_analysis_sample(manifest)

    @pytest.mark.parametrize(
        "speaker_id",
        [None, "", "   "],
    )
    def test_missing_afrispeech_source_id_raises_error(
        self,
        manifest,
        speaker_id,
    ):
        manifest.loc[
            manifest["dataset"] == "afrispeech",
            "speaker_id",
        ] = speaker_id

        with pytest.raises(
            ValueError,
            match="missing source user IDs",
        ):
            select_analysis_sample(manifest)


# run using:
# python -m pytest tests/test_select_analysis.py -v