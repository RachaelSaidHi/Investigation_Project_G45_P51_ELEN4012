"""
Tests for the resumable DSP feature-extraction pipeline.

A fake feature module and a fake audio loader are used throughout, so these
tests cover the pipeline's checkpointing, validation and error handling
without depending on any real feature family or audio file.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.common.audio_io import SAMPLE_RATE
from src.dsp.extraction_pipeline import (
    ENVELOPE_COLUMNS,
    REQUIRED_MANIFEST_COLUMNS,
    TRAILING_COLUMNS,
    FeatureSchemaError,
    build_error_record,
    build_success_record,
    check_feature_columns,
    get_result_columns,
    is_finite_number,
    process_sample,
    run_feature_manifest,
    validate_feature_module,
)


FEATURE_COLUMNS = ["value_a", "value_b"]

PROCESSED_ROOT = Path("processed_root")


def make_feature_module(
    extract=None,
    columns=FEATURE_COLUMNS,
    name="fake_feature",
    version="1",
):
    """
    Return a fake feature module with the required interface.
    """

    def default_extract(audio, sample_rate, row, config):
        return {
            "value_a": float(len(audio)),
            "value_b": 2.0,
        }

    return SimpleNamespace(
        FEATURE_NAME=name,
        FEATURE_VERSION=version,
        feature_columns=lambda config: list(columns),
        extract=extract or default_extract,
    )


def fake_loader(processed_root, processed_filepath):
    """
    Return one second of audio for any processed path.
    """

    return np.ones(SAMPLE_RATE, dtype=np.float32)


def make_manifest_row(sample_id="sample_1", **overrides):
    """
    Return one manifest row with the columns the pipeline requires.
    """

    row = {
        "sample_id": sample_id,
        "dataset": "saa",
        "processed_filepath": f"audio/{sample_id}.wav",
        "sex": "female",
        "accent_group": "south_african_english",
    }

    row.update(overrides)

    return row


def make_manifest(sample_ids=("sample_1", "sample_2", "sample_3")):
    """
    Return a manifest containing the given sample IDs.
    """

    return pd.DataFrame(
        [make_manifest_row(sample_id) for sample_id in sample_ids]
    )


def run_pipeline(manifest, results_path, **overrides):
    """
    Run the pipeline with default fake inputs and optional overrides.
    """

    arguments = {
        "manifest": manifest,
        "feature_module": make_feature_module(),
        "config": {},
        "processed_root": PROCESSED_ROOT,
        "results_path": results_path,
        "load_audio": fake_loader,
    }

    arguments.update(overrides)

    return run_feature_manifest(**arguments)


def results_columns():
    """
    Return the full results column list for the fake feature.
    """

    return ENVELOPE_COLUMNS + FEATURE_COLUMNS + TRAILING_COLUMNS


class TestValidateFeatureModule:

    def test_complete_module_passes(self):
        validate_feature_module(make_feature_module())

    @pytest.mark.parametrize(
        "missing",
        [
            "FEATURE_NAME",
            "FEATURE_VERSION",
            "feature_columns",
            "extract",
        ],
    )
    def test_missing_attribute_raises_error(self, missing):
        module = make_feature_module()
        delattr(module, missing)

        with pytest.raises(
            TypeError,
            match=f"missing required attributes: \\['{missing}'\\]",
        ):
            validate_feature_module(module)

    def test_every_missing_attribute_is_listed(self):
        with pytest.raises(TypeError) as error:
            validate_feature_module(SimpleNamespace())

        for attribute in ["FEATURE_NAME", "FEATURE_VERSION", "feature_columns", "extract"]:
            assert attribute in str(error.value)


class TestGetResultColumns:

    def test_envelope_feature_and_trailing_columns_in_order(self):
        columns = get_result_columns(make_feature_module(), {})

        assert columns == results_columns()

    def test_config_is_passed_to_feature_columns(self):
        module = make_feature_module()
        module.feature_columns = lambda config: [
            f"coef_{i}" for i in range(config["n"])
        ]

        columns = get_result_columns(module, {"n": 3})

        assert columns[len(ENVELOPE_COLUMNS):-len(TRAILING_COLUMNS)] == [
            "coef_0",
            "coef_1",
            "coef_2",
        ]

    def test_duplicate_feature_columns_raise_error(self):
        module = make_feature_module(columns=["value_a", "value_a"])

        with pytest.raises(FeatureSchemaError, match="contains duplicates"):
            get_result_columns(module, {})

    @pytest.mark.parametrize(
        "reserved",
        [
            "sample_id",
            "status",
            "error_message",
            "elapsed_s",
            "completed_at_utc",
        ],
    )
    def test_reserved_column_names_raise_error(self, reserved):
        module = make_feature_module(columns=["value_a", reserved])

        with pytest.raises(
            FeatureSchemaError,
            match=f"clash with reserved columns: \\['{reserved}'\\]",
        ):
            get_result_columns(module, {})


class TestCheckFeatureColumns:

    def test_matching_columns_pass(self):
        check_feature_columns(
            {"value_a": 1.0, "value_b": 2.0},
            FEATURE_COLUMNS,
            "fake_feature",
        )

    def test_column_order_does_not_matter(self):
        check_feature_columns(
            {"value_b": 2.0, "value_a": 1.0},
            FEATURE_COLUMNS,
            "fake_feature",
        )

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0],
            None,
            1.0,
        ],
    )
    def test_non_dict_raises_error(self, values):
        with pytest.raises(FeatureSchemaError, match="must return a dict"):
            check_feature_columns(values, FEATURE_COLUMNS, "fake_feature")

    def test_missing_column_raises_error(self):
        with pytest.raises(
            FeatureSchemaError,
            match=r"Missing: \['value_b'\]",
        ):
            check_feature_columns(
                {"value_a": 1.0},
                FEATURE_COLUMNS,
                "fake_feature",
            )

    def test_unexpected_column_raises_error(self):
        with pytest.raises(
            FeatureSchemaError,
            match=r"Unexpected: \['value_c'\]",
        ):
            check_feature_columns(
                {"value_a": 1.0, "value_b": 2.0, "value_c": 3.0},
                FEATURE_COLUMNS,
                "fake_feature",
            )


class TestIsFiniteNumber:

    @pytest.mark.parametrize(
        "value",
        [
            0,
            1.5,
            -3,
            np.float32(2.0),
            np.int64(4),
            "2.5",
            True,
        ],
    )
    def test_finite_values(self, value):
        assert is_finite_number(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            np.nan,
            np.inf,
            -np.inf,
            None,
            "abc",
            "",
            [1.0],
        ],
    )
    def test_non_finite_or_non_numeric_values(self, value):
        assert is_finite_number(value) is False


class TestBuildRecords:

    def test_success_record_contains_envelope_features_and_timing(self):
        record = build_success_record(
            row=make_manifest_row(),
            manifest_index=4,
            feature_module=make_feature_module(),
            feature_values={"value_a": 1.0, "value_b": 2.0},
            processed_duration_s=1.0,
            elapsed_s=0.25,
        )

        assert set(record) == set(results_columns())
        assert record["sample_id"] == "sample_1"
        assert record["dataset"] == "saa"
        assert record["manifest_index"] == 4
        assert record["status"] == "ok"
        assert record["error_type"] is None
        assert record["error_message"] is None
        assert record["feature_name"] == "fake_feature"
        assert record["feature_version"] == "1"
        assert record["value_a"] == 1.0
        assert record["value_b"] == 2.0
        assert record["processed_duration_s"] == 1.0
        assert record["elapsed_s"] == 0.25
        assert pd.Timestamp(record["completed_at_utc"]).tzinfo is not None

    def test_error_record_has_error_details_and_no_features(self):
        record = build_error_record(
            row=make_manifest_row(),
            manifest_index=2,
            feature_module=make_feature_module(),
            error=FileNotFoundError("missing.wav"),
            processed_duration_s=None,
            elapsed_s=0.1,
        )

        assert record["sample_id"] == "sample_1"
        assert record["dataset"] == "saa"
        assert record["manifest_index"] == 2
        assert record["status"] == "error"
        assert record["error_type"] == "FileNotFoundError"
        assert record["error_message"] == "missing.wav"
        assert record["error_message"] is None
        assert record["feature_name"] == "fake_feature"
        assert record["processed_duration_s"] is None
        assert record["elapsed_s"] == 0.1
        assert pd.Timestamp(record["completed_at_utc"]).tzinfo is not None
        assert "value_a" not in record
        assert "value_b" not in record

    def test_sample_id_is_stored_as_string(self):
        record = build_success_record(
            row=make_manifest_row(sample_id=7),
            manifest_index=0,
            feature_module=make_feature_module(),
            feature_values={},
            processed_duration_s=1.0,
            elapsed_s=0.1,
        )

        assert record["sample_id"] == "7"


class TestProcessSample:

    def process(self, feature_module=None, load_audio=fake_loader, row=None, config=None):
        """
        Run process_sample() with default fake inputs.
        """

        return process_sample(
            row=pd.Series(row or make_manifest_row()),
            manifest_index=0,
            feature_module=feature_module or make_feature_module(),
            config=config if config is not None else {},
            feature_columns=FEATURE_COLUMNS,
            processed_root=PROCESSED_ROOT,
            load_audio=load_audio,
        )

    def test_successful_sample(self):
        record = self.process()

        assert record["status"] == "ok"
        assert record["value_a"] == float(SAMPLE_RATE)
        assert record["value_b"] == 2.0
        assert record["processed_duration_s"] == pytest.approx(1.0)
        assert record["elapsed_s"] >= 0

    def test_loader_receives_root_and_relative_path(self):
        received = {}

        def recording_loader(processed_root, processed_filepath):
            received["root"] = processed_root
            received["path"] = processed_filepath
            return fake_loader(processed_root, processed_filepath)

        self.process(load_audio=recording_loader)

        assert received == {
            "root": PROCESSED_ROOT,
            "path": "audio/sample_1.wav",
        }

    def test_extract_receives_audio_sample_rate_row_and_config(self):
        received = {}

        def recording_extract(audio, sample_rate, row, config):
            received["length"] = len(audio)
            received["sample_rate"] = sample_rate
            received["sample_id"] = row["sample_id"]
            received["config"] = config
            return {"value_a": 1.0, "value_b": 2.0}

        self.process(
            feature_module=make_feature_module(extract=recording_extract),
            config={"feature": {"setting": 1}},
        )

        assert received == {
            "length": SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE,
            "sample_id": "sample_1",
            "config": {"feature": {"setting": 1}},
        }

    def test_loader_error_becomes_error_row(self):
        def failing_loader(processed_root, processed_filepath):
            raise FileNotFoundError("Processed audio not found")

        record = self.process(load_audio=failing_loader)

        assert record["status"] == "error"
        assert record["error_type"] == "FileNotFoundError"
        assert "Processed audio not found" in record["error_message"]
        assert record["processed_duration_s"] is None

    def test_extract_error_becomes_error_row_with_duration(self):
        def failing_extract(audio, sample_rate, row, config):
            raise ValueError("Insufficient speech")

        record = self.process(
            feature_module=make_feature_module(extract=failing_extract),
        )

        assert record["status"] == "error"
        assert record["error_type"] == "ValueError"
        assert record["processed_duration_s"] == pytest.approx(1.0)

    def test_custom_exclusion_type_is_recorded(self):
        class NoUsableSpeech(ValueError):
            pass

        def excluded_extract(audio, sample_rate, row, config):
            raise NoUsableSpeech("Audio contains only digital silence.")

        record = self.process(
            feature_module=make_feature_module(extract=excluded_extract),
        )

        assert record["error_type"] == "error"
        assert record["error_type"] == "NoUsableSpeech"

    @pytest.mark.parametrize(
        "bad_value",
        [
            np.nan,
            np.inf,
            None,
            "not a number",
        ],
    )
    def test_non_finite_feature_value_becomes_error_row(self, bad_value):
        def bad_extract(audio, sample_rate, row, config):
            return {"value_a": 1.0, "value_b": bad_value}

        record = self.process(
            feature_module=make_feature_module(extract=bad_extract),
        )

        assert record["status"] == "error"
        assert record["error_type"] == "ValueError"
        assert "Non-finite feature values in: ['value_b']" in record["error_message"]

    def test_numeric_values_are_converted_to_float(self):
        def numpy_extract(audio, sample_rate, row, config):
            return {"value_a": np.float32(1.5), "value_b": 3}

        record = self.process(
            feature_module=make_feature_module(extract=numpy_extract),
        )

        assert type(record["value_a"]) is float
        assert type(record["value_b"]) is float
        assert record["value_b"] == 3.0

    def test_wrong_columns_raise_schema_error(self):
        def wrong_extract(audio, sample_rate, row, config):
            return {"value_a": 1.0}

        with pytest.raises(FeatureSchemaError, match="wrong columns"):
            self.process(
                feature_module=make_feature_module(extract=wrong_extract),
            )


class TestRunFeatureManifest:

    def test_every_sample_is_processed_and_saved(self, tmp_path):
        results_path = tmp_path / "features.csv"

        results = run_pipeline(make_manifest(), results_path)

        assert results["sample_id"].tolist() == ["sample_1", "sample_2", "sample_3"]
        assert results["status"].tolist() == ["ok", "ok", "ok"]
        assert results["manifest_index"].tolist() == [0, 1, 2]
        assert results.columns.tolist() == results_columns()

        saved = pd.read_csv(results_path)

        assert len(saved) == 3
        assert saved.columns.tolist() == results_columns()

    def test_successful_samples_are_not_repeated(self, tmp_path):
        results_path = tmp_path / "features.csv"
        run_pipeline(make_manifest(), results_path)

        calls = []

        def counting_loader(processed_root, processed_filepath):
            calls.append(processed_filepath)
            return fake_loader(processed_root, processed_filepath)

        results = run_pipeline(
            make_manifest(),
            results_path,
            load_audio=counting_loader,
        )

        assert calls == []
        assert len(results) == 3

    def test_new_manifest_samples_are_processed_on_resume(self, tmp_path):
        results_path = tmp_path / "features.csv"
        run_pipeline(make_manifest(["sample_1"]), results_path)

        calls = []

        def counting_loader(processed_root, processed_filepath):
            calls.append(processed_filepath)
            return fake_loader(processed_root, processed_filepath)

        results = run_pipeline(
            make_manifest(["sample_1", "sample_2"]),
            results_path,
            load_audio=counting_loader,
        )

        assert calls == ["audio/sample_2.wav"]
        assert results["sample_id"].tolist() == ["sample_1", "sample_2"]

    def test_previous_errors_are_skipped_by_default(self, tmp_path):
        results_path = tmp_path / "features.csv"

        def failing_loader(processed_root, processed_filepath):
            if processed_filepath.endswith("sample_2.wav"):
                raise OSError("unreadable")
            return fake_loader(processed_root, processed_filepath)

        run_pipeline(make_manifest(), results_path, load_audio=failing_loader)

        calls = []

        def counting_loader(processed_root, processed_filepath):
            calls.append(processed_filepath)
            return fake_loader(processed_root, processed_filepath)

        results = run_pipeline(
            make_manifest(),
            results_path,
            load_audio=counting_loader,
        )

        assert calls == []
        assert results.set_index("sample_id").loc["sample_2", "status"] == "error"

    def test_retry_errors_reprocesses_only_failed_samples(self, tmp_path):
        results_path = tmp_path / "features.csv"

        def failing_loader(processed_root, processed_filepath):
            if processed_filepath.endswith("sample_2.wav"):
                raise OSError("unreadable")
            return fake_loader(processed_root, processed_filepath)

        run_pipeline(make_manifest(), results_path, load_audio=failing_loader)

        calls = []

        def counting_loader(processed_root, processed_filepath):
            calls.append(processed_filepath)
            return fake_loader(processed_root, processed_filepath)

        results = run_pipeline(
            make_manifest(),
            results_path,
            load_audio=counting_loader,
            retry_errors=True,
        )

        assert calls == ["audio/sample_2.wav"]
        assert results["status"].tolist() == ["ok", "ok", "ok"]
        assert results["sample_id"].tolist() == ["sample_1", "sample_2", "sample_3"]
        assert results["error_type"].isna().all()

    def test_sample_errors_do_not_stop_the_run(self, tmp_path):
        def failing_loader(processed_root, processed_filepath):
            if processed_filepath.endswith("sample_1.wav"):
                raise OSError("unreadable")
            return fake_loader(processed_root, processed_filepath)

        results = run_pipeline(
            make_manifest(),
            tmp_path / "features.csv",
            load_audio=failing_loader,
        )

        assert results["status"].tolist() == ["error", "ok", "ok"]
        assert results.loc[0, ["value_a", "value_b"]].isna().all()

    def test_fail_fast_stops_after_saving_the_error_row(self, tmp_path):
        results_path = tmp_path / "features.csv"

        def failing_loader(processed_root, processed_filepath):
            if processed_filepath.endswith("sample_2.wav"):
                raise OSError("unreadable")
            return fake_loader(processed_root, processed_filepath)

        with pytest.raises(
            RuntimeError,
            match="Feature extraction failed for sample sample_2: unreadable",
        ):
            run_pipeline(
                make_manifest(),
                results_path,
                load_audio=failing_loader,
                fail_fast=True,
            )

        saved = pd.read_csv(results_path)

        assert saved["sample_id"].tolist() == ["sample_1", "sample_2"]
        assert saved["status"].tolist() == ["ok", "error"]

    def test_checkpoint_is_saved_after_every_sample(self, tmp_path):
        results_path = tmp_path / "features.csv"

        def interrupting_loader(processed_root, processed_filepath):
            if processed_filepath.endswith("sample_3.wav"):
                raise KeyboardInterrupt
            return fake_loader(processed_root, processed_filepath)

        with pytest.raises(KeyboardInterrupt):
            run_pipeline(
                make_manifest(),
                results_path,
                load_audio=interrupting_loader,
            )

        assert pd.read_csv(results_path)["sample_id"].tolist() == [
            "sample_1",
            "sample_2",
        ]

    def test_schema_error_stops_the_run(self, tmp_path):
        results_path = tmp_path / "features.csv"

        def wrong_extract(audio, sample_rate, row, config):
            return {"value_a": 1.0}

        with pytest.raises(FeatureSchemaError):
            run_pipeline(
                make_manifest(),
                results_path,
                feature_module=make_feature_module(extract=wrong_extract),
            )

        # A code bug is not recorded as a per-sample error row.
        assert not results_path.exists()

    def test_invalid_feature_module_fails_before_processing(self, tmp_path):
        with pytest.raises(TypeError, match="missing required attributes"):
            run_pipeline(
                make_manifest(),
                tmp_path / "features.csv",
                feature_module=SimpleNamespace(FEATURE_NAME="broken"),
            )

    def test_numeric_sample_ids_are_handled_as_strings(self, tmp_path):
        results_path = tmp_path / "features.csv"
        manifest = make_manifest(["001", "002"])

        run_pipeline(manifest, results_path)
        results = run_pipeline(manifest, results_path)

        assert results["sample_id"].tolist() == ["001", "002"]

    def test_manifest_is_not_modified(self, tmp_path):
        manifest = make_manifest()
        original = manifest.copy()

        run_pipeline(manifest, tmp_path / "features.csv")

        pd.testing.assert_frame_equal(manifest, original)

    def test_summary_is_printed(self, tmp_path, capsys):
        run_pipeline(make_manifest(), tmp_path / "features.csv")

        output = capsys.readouterr().out

        assert "Feature: fake_feature (version 1)" in output
        assert "Manifest samples: 3" in output
        assert "Successful: 3" in output
        assert "Errors: 0" in output


class TestRunFeatureManifestValidation:

    @pytest.mark.parametrize(
        "column",
        REQUIRED_MANIFEST_COLUMNS,
    )
    def test_missing_required_manifest_column_raises_error(self, tmp_path, column):
        manifest = make_manifest().drop(columns=column)

        with pytest.raises(ValueError, match="missing required columns"):
            run_pipeline(manifest, tmp_path / "features.csv")

    def test_blank_required_value_raises_error(self, tmp_path):
        manifest = make_manifest()
        manifest.loc[1, "processed_filepath"] = ""

        with pytest.raises(ValueError, match="processed_filepath"):
            run_pipeline(manifest, tmp_path / "features.csv")

    def test_duplicate_sample_ids_raise_error(self, tmp_path):
        manifest = make_manifest(["sample_1", "sample_1"])

        with pytest.raises(ValueError, match="duplicate sample_id"):
            run_pipeline(manifest, tmp_path / "features.csv")

    def test_existing_results_from_another_manifest_raise_error(self, tmp_path):
        results_path = tmp_path / "features.csv"
        run_pipeline(make_manifest(["other_sample"]), results_path)

        with pytest.raises(
            ValueError,
            match=r"not present in this manifest: \['other_sample'\]",
        ):
            run_pipeline(make_manifest(), results_path)

    def test_invalid_existing_status_raises_error(self, tmp_path):
        results_path = tmp_path / "features.csv"
        run_pipeline(make_manifest(["sample_1"]), results_path)

        saved = pd.read_csv(results_path)
        saved.loc[0, "status"] = "pending"
        saved.to_csv(results_path, index=False)

        with pytest.raises(
            ValueError,
            match=r"invalid status values: \['pending'\]",
        ):
            run_pipeline(make_manifest(["sample_1"]), results_path)

    def test_existing_results_missing_feature_column_raise_error(self, tmp_path):
        results_path = tmp_path / "features.csv"
        run_pipeline(make_manifest(["sample_1"]), results_path)

        pd.read_csv(results_path).drop(columns="value_b").to_csv(
            results_path,
            index=False,
        )

        with pytest.raises(ValueError, match="missing columns"):
            run_pipeline(make_manifest(["sample_1"]), results_path)


# run using:
# python -m pytest tests/test_extraction_pipeline.py -v
