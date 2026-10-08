"""
Tests for the shared manifest and results-table checks.
"""

import numpy as np
import pandas as pd
import pytest

from src.common.manifest_checks import (
    _is_blank,
    require_columns,
    validate_sample_ids,
)


def make_table(**overrides):
    """
    Return a small valid manifest-like table with optional column overrides.
    """

    data = {
        "sample_id": ["sample_1", "sample_2", "sample_3"],
        "dataset": ["saa", "afrispeech", "saa"],
        "processed_filepath": [
            "audio/sample_1.wav",
            "audio/sample_2.wav",
            "audio/sample_3.wav",
        ],
    }

    data.update(overrides)

    return pd.DataFrame(data)


class TestIsBlank:

    @pytest.mark.parametrize(
        "value",
        [
            None,
            float("nan"),
            np.nan,
            pd.NA,
            "",
            "   ",
            "\t\n",
        ],
    )
    def test_missing_and_whitespace_values_are_blank(self, value):
        assert _is_blank(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "saa",
            " sample ",
            0,
            0.0,
            False,
            "0",
        ],
    )
    def test_real_values_are_not_blank(self, value):
        assert _is_blank(value) is False


class TestRequireColumns:

    def test_valid_table_passes(self):
        require_columns(
            make_table(),
            ["sample_id", "dataset"],
        )

    def test_empty_column_list_passes(self):
        require_columns(
            make_table(),
            [],
        )

    def test_missing_column_raises_error(self):
        with pytest.raises(
            ValueError,
            match=r"missing required columns: \['accent_group'\]",
        ):
            require_columns(
                make_table(),
                ["sample_id", "accent_group"],
            )

    def test_missing_columns_are_listed_in_sorted_order(self):
        with pytest.raises(
            ValueError,
            match=r"\['accent_group', 'sex'\]",
        ):
            require_columns(
                make_table(),
                ["sex", "accent_group", "dataset"],
            )

    def test_custom_label_is_used_in_error(self):
        with pytest.raises(
            ValueError,
            match="Results table is missing required columns",
        ):
            require_columns(
                make_table(),
                ["status"],
                label="Results table",
            )

    @pytest.mark.parametrize(
        "blank_value",
        [
            "",
            "   ",
            None,
            np.nan,
        ],
    )
    def test_blank_value_in_required_column_raises_error(self, blank_value):
        table = make_table(
            dataset=["saa", blank_value, "saa"],
        )

        with pytest.raises(
            ValueError,
            match="1 missing or blank values in required column: dataset",
        ):
            require_columns(
                table,
                ["sample_id", "dataset"],
            )

    def test_number_of_blank_values_is_reported(self):
        table = make_table(
            dataset=["", None, "saa"],
        )

        with pytest.raises(
            ValueError,
            match="has 2 missing or blank values",
        ):
            require_columns(
                table,
                ["dataset"],
            )

    def test_blank_values_in_optional_columns_are_ignored(self):
        table = make_table(
            processed_filepath=["", None, "audio/sample_3.wav"],
        )

        require_columns(
            table,
            ["sample_id", "dataset"],
        )

    def test_duplicate_required_column_names_are_checked_once(self):
        require_columns(
            make_table(),
            ["dataset", "dataset"],
        )

    def test_zero_values_are_not_treated_as_blank(self):
        table = make_table(
            age_onset=[0, 0, 0],
        )

        require_columns(
            table,
            ["age_onset"],
        )


class TestValidateSampleIds:

    def test_unique_ids_pass(self):
        validate_sample_ids(make_table())

    def test_empty_table_with_sample_id_column_passes(self):
        validate_sample_ids(
            pd.DataFrame(columns=["sample_id"])
        )

    def test_missing_sample_id_column_raises_error(self):
        table = make_table().drop(columns="sample_id")

        with pytest.raises(
            ValueError,
            match="Manifest has no sample_id column",
        ):
            validate_sample_ids(table)

    @pytest.mark.parametrize(
        "blank_id",
        [
            "",
            "  ",
            None,
            np.nan,
        ],
    )
    def test_blank_sample_id_raises_error(self, blank_id):
        table = make_table(
            sample_id=["sample_1", blank_id, "sample_3"],
        )

        with pytest.raises(
            ValueError,
            match="1 missing or empty sample_id values",
        ):
            validate_sample_ids(table)

    def test_duplicate_sample_ids_raise_error(self):
        table = make_table(
            sample_id=["sample_1", "sample_1", "sample_3"],
        )

        with pytest.raises(
            ValueError,
            match=r"duplicate sample_id values: \['sample_1'\]",
        ):
            validate_sample_ids(table)

    def test_each_duplicate_id_is_listed_once(self):
        table = pd.DataFrame(
            {
                "sample_id": ["a", "a", "a", "b", "b", "c"],
            }
        )

        with pytest.raises(ValueError) as error:
            validate_sample_ids(table)

        assert "['a', 'b']" in str(error.value)

    def test_numeric_and_string_ids_that_match_as_text_are_duplicates(self):
        # IDs are compared as strings, because that is how they are joined.
        table = pd.DataFrame(
            {
                "sample_id": [7, "7"],
            }
        )

        with pytest.raises(
            ValueError,
            match="duplicate sample_id values",
        ):
            validate_sample_ids(table)

    def test_custom_label_is_used_in_error(self):
        table = make_table(
            sample_id=["x", "x", "y"],
        )

        with pytest.raises(
            ValueError,
            match="Existing results contains duplicate",
        ):
            validate_sample_ids(
                table,
                label="Existing results",
            )

    def test_table_is_not_modified(self):
        table = make_table()
        original = table.copy()

        validate_sample_ids(table)
        require_columns(table, ["sample_id"])

        pd.testing.assert_frame_equal(
            table,
            original,
        )


# run using:
# python -m pytest tests/test_manifest_checks.py -v
