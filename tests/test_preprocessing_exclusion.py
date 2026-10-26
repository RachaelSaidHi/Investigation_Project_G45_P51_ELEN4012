"""Tests for the preprocessing quality-control exclusion rules."""

import pandas as pd
import pytest

from src.preprocessing.exclusion import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    MIN_FLOOR_GAP_DB,
    MIN_HF_ENERGY_PCT,
    REASON_BANDWIDTH_TOO_LOW,
    REASON_DURATION_TOO_LONG,
    REASON_DURATION_TOO_SHORT,
    REASON_PERSISTENT_NOISE_FLOOR_TOO_HIGH,
    apply_exclusion_rules,
    get_exclusion_reasons,
    should_exclude,
)


def make_quality_table(rows):
    """Build a small quality table for exclusion-rule tests."""

    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "processed_duration_s",
            "hf_energy_pct",
            "wada_snr_db",
            "floor_gap_db",
        ],
    )


class TestDurationExclusion:

    def test_recording_below_minimum_duration_is_excluded(self):
        reasons = get_exclusion_reasons(
            duration_s=MIN_DURATION_S - 0.01,
            hf_energy_pct=1.0,
            floor_gap_db=40.0,
        )

        assert reasons == [
            REASON_DURATION_TOO_SHORT
        ]

    def test_recording_at_minimum_duration_is_kept(self):
        reasons = get_exclusion_reasons(
            duration_s=MIN_DURATION_S,
            hf_energy_pct=1.0,
            floor_gap_db=40.0,
        )

        assert reasons == []

    def test_recording_at_maximum_duration_is_kept(self):
        reasons = get_exclusion_reasons(
            duration_s=MAX_DURATION_S,
            hf_energy_pct=1.0,
            floor_gap_db=40.0,
        )

        assert reasons == []

    def test_recording_above_maximum_duration_is_excluded(self):
        reasons = get_exclusion_reasons(
            duration_s=MAX_DURATION_S + 0.01,
            hf_energy_pct=1.0,
            floor_gap_db=40.0,
        )

        assert reasons == [
            REASON_DURATION_TOO_LONG
        ]


class TestBandwidthExclusion:

    def test_recording_below_hf_threshold_is_excluded(self):
        reasons = get_exclusion_reasons(
            duration_s=10.0,
            hf_energy_pct=MIN_HF_ENERGY_PCT - 0.001,
            floor_gap_db=40.0,
        )

        assert reasons == [
            REASON_BANDWIDTH_TOO_LOW
        ]

    def test_recording_at_hf_threshold_is_kept(self):
        reasons = get_exclusion_reasons(
            duration_s=10.0,
            hf_energy_pct=MIN_HF_ENERGY_PCT,
            floor_gap_db=40.0,
        )

        assert reasons == []

class TestPersistentNoiseExclusion:

    def test_recording_below_floor_gap_threshold_is_excluded(self):
        reasons = get_exclusion_reasons(
            duration_s=10.0,
            hf_energy_pct=1.0,
            floor_gap_db=MIN_FLOOR_GAP_DB - 0.01,
        )

        assert reasons == [
            REASON_PERSISTENT_NOISE_FLOOR_TOO_HIGH
        ]

    def test_recording_at_floor_gap_threshold_is_kept(self):
        reasons = get_exclusion_reasons(
            duration_s=10.0,
            hf_energy_pct=1.0,
            floor_gap_db=MIN_FLOOR_GAP_DB,
        )

        assert reasons == []

    def test_recording_above_floor_gap_threshold_is_kept(self):
        reasons = get_exclusion_reasons(
            duration_s=10.0,
            hf_energy_pct=1.0,
            floor_gap_db=MIN_FLOOR_GAP_DB + 0.01,
        )

        assert reasons == []

class TestCombinedExclusions:

    def test_all_applicable_reasons_are_returned(self):
        reasons = get_exclusion_reasons(
            duration_s=4.0,
            hf_energy_pct=0.01,
            floor_gap_db=20.0,
        )

        assert reasons == [
            REASON_DURATION_TOO_SHORT,
            REASON_BANDWIDTH_TOO_LOW,
            REASON_PERSISTENT_NOISE_FLOOR_TOO_HIGH,
        ]

    def test_should_exclude_matches_exclusion_reasons(self):
        assert not should_exclude(
            duration_s=10.0,
            hf_energy_pct=1.0,
            floor_gap_db=40.0,
        )

        assert should_exclude(
            duration_s=4.0,
            hf_energy_pct=1.0,
            floor_gap_db=40.0,
        )

        assert should_exclude(
            duration_s=10.0,
            hf_energy_pct=1.0,
            floor_gap_db=20.0,
        )


class TestApplyExclusionRules:

    def test_exclusion_columns_are_added(self):
        quality = make_quality_table([
            ("kept", 10.0, 1.0, 20.0, 40.0),
            ("short", 3.0, 1.0, 20.0, 40.0),
            ("both", 3.0, 0.01, 20.0, 40.0),
            ("noisy", 10.0, 1.0, 20.0, 20.0),
        ])

        result = apply_exclusion_rules(
            quality
        )

        assert result["exclude"].tolist() == [
            False,
            True,
            True,
            True,
        ]

        assert result["exclude_reason"].tolist() == [
            "",
            REASON_DURATION_TOO_SHORT,
            (
                REASON_DURATION_TOO_SHORT
                + ";"
                + REASON_BANDWIDTH_TOO_LOW
            ),
            REASON_PERSISTENT_NOISE_FLOOR_TOO_HIGH,
        ]

    def test_input_table_is_not_modified(self):
        quality = make_quality_table([
            ("kept", 10.0, 1.0, 20.0, 40.0),
        ])

        apply_exclusion_rules(
            quality
        )

        assert "exclude" not in quality.columns
        assert "exclude_reason" not in quality.columns

    def test_existing_columns_are_preserved(self):
        quality = make_quality_table([
            ("kept", 10.0, 1.0, 20.0, 40.0),
        ])

        result = apply_exclusion_rules(
            quality
        )

        assert result.columns.tolist()[
            :len(quality.columns)
        ] == quality.columns.tolist()

    def test_low_wada_snr_does_not_exclude(self):
        """WADA-SNR is measured but is not currently an exclusion rule."""

        quality = make_quality_table([
            ("noisy", 10.0, 1.0, -20.0, 40.0),
        ])

        result = apply_exclusion_rules(
            quality
        )

        assert not result[
            "exclude"
        ].iloc[0]

    def test_missing_required_column_fails(self):
        quality = pd.DataFrame({
            "sample_id": ["a"],
            "processed_duration_s": [10.0],
        })

        with pytest.raises(
            ValueError,
            match="missing required columns",
        ):
            apply_exclusion_rules(
                quality
            )

    def test_missing_measurement_fails(self):
        quality = make_quality_table([
            ("a", 10.0, None, 20.0, 40.0),
        ])

        with pytest.raises(
            ValueError,
            match="missing values",
        ):
            apply_exclusion_rules(
                quality
            )

    def test_missing_floor_gap_measurement_fails(self):
        quality = make_quality_table([
            ("a", 10.0, 1.0, 20.0, None),
        ])

        with pytest.raises(
            ValueError,
            match="missing values",
        ):
            apply_exclusion_rules(
                quality
            )


# run using 
# python -m pytest tests/test_preprocessing_exclusion.py -v