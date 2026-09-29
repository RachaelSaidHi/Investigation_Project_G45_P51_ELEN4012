"""
Tests for the transcription pipeline functionality
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.asr import transcription_pipeline 


# Helpers

def make_transcription_result(
    text="decoded transcript",
    duration_s=10.0,
    preprocess_s=0.10,
    inference_s=0.50,
    decode_s=0.10,
):
    elapsed_s = preprocess_s + inference_s + decode_s

    return SimpleNamespace(
        text=text,
        duration_s=duration_s,
        preprocess_s=preprocess_s,
        inference_s=inference_s,
        decode_s=decode_s,
        elapsed_s=elapsed_s,
        real_time_factor=elapsed_s / duration_s,
    )


class DummyTranscriber:
    """
    Small stand-in for WhisperTranscriber.

    It records calls to transcribe() so that resume/retry behaviour can be
    verified without loading Whisper.
    """

    def __init__(self, result=None, error=None, config=None):
        self.result = result or make_transcription_result()
        self.error = error
        self.calls = []
        self.config = config or {
            "model_name": "openai/whisper-tiny",
            "decode_mode": "sequential_long_form",
            "device": "cpu",
            "hardware": "CPU",
            "precision": "torch.float32",
        }

    def describe(self):
        return dict(self.config)

    def transcribe(self, filepath):
        self.calls.append(filepath)

        if self.error is not None:
            raise self.error

        return self.result


@pytest.fixture
def transcriber_config():
    return {
        "model_name": "openai/whisper-tiny",
        "decode_mode": "sequential_long_form",
        "device": "cpu",
        "hardware": "CPU",
        "precision": "torch.float32",
    }


@pytest.fixture
def make_transcriber(transcriber_config):
    def _make(result=None, error=None, config=None):
        return DummyTranscriber(
            result=result,
            error=error,
            config=config or transcriber_config,
        )

    return _make


@pytest.fixture
def make_audio_file(tmp_path):
    """
    Create a fake file that passes validate_audio_path().
    """

    def _make(filename="sample.wav"):
        data_root = tmp_path / "raw"
        path = data_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dummy audio")
        return data_root, path

    return _make


def make_existing_record(
    sample_id="sample_1",
    manifest_index=0,
    status="ok",
    transcription_raw="decoded transcript",
):
    """
    Create a complete checkpoint row with all RESULT_COLUMNS present.
    """

    record = {column: None for column in transcription_pipeline.RESULT_COLUMNS}

    record.update(
        {
            "sample_id": sample_id,
            "dataset": "saa",
            "filepath": "sample.wav",
            "manifest_index": manifest_index,
            "status": status,
            "model_name": "openai/whisper-tiny",
            "decode_mode": "sequential_long_form",
            "device": "cpu",
            "hardware": "CPU",
            "precision": "torch.float32",
            "transcription_raw": transcription_raw,
            "completed_at_utc": "2026-01-01T00:00:00+00:00",
        }
    )

    if status == "error":
        record["error_type"] = "RuntimeError"
        record["error_message"] = "previous failure"
        record["transcription_raw"] = None

    return record


class TestValidateManifest:

    def test_valid_manifest_is_accepted(self):
        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1",
                "filepath": "sample.wav",
            }
        ])

        transcription_pipeline.validate_manifest(manifest)

    def test_missing_required_column_fails(self):
        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1",
            }
        ])

        with pytest.raises(ValueError, match="missing required columns"):
            transcription_pipeline.validate_manifest(manifest)

    def test_missing_sample_id_fails(self):
        manifest = pd.DataFrame([
            {
                "sample_id": None,
                "filepath": "sample.wav",
            }
        ])

        with pytest.raises(ValueError, match="missing sample_id"):
            transcription_pipeline.validate_manifest(manifest)

    def test_missing_filepath_fails(self):
        manifest = pd.DataFrame([
            {"sample_id": "sample_1", 
             "filepath": None
            }
        ])

        with pytest.raises(ValueError, match="missing filepath"):
            transcription_pipeline.validate_manifest(manifest)

    def test_duplicate_sample_ids_fail(self):
        manifest = pd.DataFrame([
            {"sample_id": "sample_1", "filepath": "a.wav"},
            {"sample_id": "sample_1", "filepath": "b.wav"},
        ])

        with pytest.raises(ValueError, match="duplicate sample_id"):
            transcription_pipeline.validate_manifest(manifest)


class TestValidateAudioPath:

    def test_existing_file_is_returned(self, make_audio_file):
        data_root, path = make_audio_file("sample.wav")
        row = pd.Series({"filepath": "sample.wav"})

        result = transcription_pipeline.validate_audio_path(row, data_root)

        assert result == path

    def test_missing_file_fails(self, tmp_path):
        data_root = tmp_path / "raw"
        data_root.mkdir()

        row = pd.Series({"filepath": "missing.wav"})

        with pytest.raises(FileNotFoundError, match="does not exist"):
            transcription_pipeline.validate_audio_path(row, data_root)

    def test_directory_path_fails(self, tmp_path):
        data_root = tmp_path / "raw"
        data_root.mkdir()

        directory = data_root / "not_audio"
        directory.mkdir()

        row = pd.Series({"filepath": "not_audio"})

        with pytest.raises(ValueError, match="not a file"):
            transcription_pipeline.validate_audio_path(row, data_root)


class TestGetManifestDuration:

    def test_missing_duration_column_returns_none(self):
        row = pd.Series({"sample_id": "1"})

        assert transcription_pipeline.get_manifest_duration(row) is None

    def test_nan_duration_returns_none(self):
        row = pd.Series({"duration_s": float("nan")})

        assert transcription_pipeline.get_manifest_duration(row) is None

    def test_duration_is_returned_as_float(self):
        row = pd.Series({"duration_s": 12.5})

        result = transcription_pipeline.get_manifest_duration(row)

        assert result == 12.5
        assert isinstance(result, float)


class TestBuildSuccessRecord:

    def test_success_record_contains_transcription_and_configuration(self, transcriber_config):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "filepath": "sample.wav",
                "duration_s": 10.0,
            }
        )

        transcription = make_transcription_result(
            text="hello world",
            duration_s=10.0,
        )

        record = transcription_pipeline.build_success_record(
            row=row,
            manifest_index=3,
            transcription=transcription,
            config=transcriber_config,
            duration_tolerance_s=0.5,
        )

        assert record["sample_id"] == "sample_1"
        assert record["dataset"] == "saa"
        assert record["manifest_index"] == 3
        assert record["status"] == "ok"
        assert record["error_type"] == None
        assert record["error_message"] == None
        assert record["transcription_raw"] == "hello world"
        assert record["model_name"] == "openai/whisper-tiny"
        assert record["decode_mode"] == "sequential_long_form"
        assert record["processed_duration_s"] == pytest.approx(10.0)
        assert record["duration_difference_s"] == pytest.approx(0.0)
        assert record["duration_mismatch"] is False

    def test_exact_duration_tolerance_is_not_a_mismatch(self, transcriber_config):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "filepath": "sample.wav",
                "duration_s": 10.0,
            }
        )

        transcription = make_transcription_result(duration_s=10.5)

        record = transcription_pipeline.build_success_record(
            row=row,
            manifest_index=0,
            transcription=transcription,
            config=transcriber_config,
            duration_tolerance_s=0.5,
        )

        assert record["duration_difference_s"] == pytest.approx(0.5)
        assert record["duration_mismatch"] is False

    def test_duration_above_tolerance_is_a_mismatch(self, transcriber_config):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "filepath": "sample.wav",
                "duration_s": 10.0,
            }
        )

        transcription = make_transcription_result(duration_s=10.6)

        record = transcription_pipeline.build_success_record(
            row=row,
            manifest_index=0,
            transcription=transcription,
            config=transcriber_config,
            duration_tolerance_s=0.5,
        )

        assert record["duration_difference_s"] == pytest.approx(0.6)
        assert record["duration_mismatch"] is True

    def test_missing_manifest_duration_leaves_duration_diagnostics_empty(self, transcriber_config):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "filepath": "sample.wav",
            }
        )

        record = transcription_pipeline.build_success_record(
            row=row,
            manifest_index=0,
            transcription=make_transcription_result(),
            config=transcriber_config,
            duration_tolerance_s=0.5,
        )

        assert record["manifest_duration_s"] is None
        assert record["duration_difference_s"] is None
        assert record["duration_mismatch"] is None


class TestBuildErrorRecord:

    def test_error_record_stores_exception_details(self, transcriber_config):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "filepath": "sample.wav",
                "duration_s": 10.0,
            }
        )

        record = transcription_pipeline.build_error_record(
            row=row,
            manifest_index=2,
            config=transcriber_config,
            error=ValueError("broken sample"),
        )

        assert record["sample_id"] == "sample_1"
        assert record["dataset"] == "saa"
        assert record["manifest_index"] == 2
        assert record["status"] == "error"
        assert record["error_type"] == "ValueError"
        assert record["error_message"] == "broken sample"
        assert record["model_name"] == "openai/whisper-tiny"
        assert record["decode_mode"] == "sequential_long_form"
        assert record["transcription_raw"] is None
        assert record["manifest_duration_s"] == pytest.approx(10.0)
        assert record["processed_duration_s"] is None
        assert record["duration_difference_s"] == None
        assert record["duration_mismatch"] is None


class TestLoadExistingResults:

    def test_missing_checkpoint_returns_empty_result_table(self, tmp_path):
        path = tmp_path / "transcriptions.csv"

        results = transcription_pipeline.load_existing_results(path)

        assert results.empty
        assert list(results.columns) == transcription_pipeline.RESULT_COLUMNS

    def test_checkpoint_missing_required_columns_fails(self, tmp_path):
        path = tmp_path / "transcriptions.csv"
        pd.DataFrame([{"sample_id": "sample_1"}]).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing columns"):
            transcription_pipeline.load_existing_results(path)

    def test_duplicate_sample_ids_in_checkpoint_fail(self, tmp_path):
        path = tmp_path / "transcriptions.csv"

        results = pd.DataFrame(
            [
                make_existing_record("sample_1", 0),
                make_existing_record("sample_1", 1),
            ],
            columns=transcription_pipeline.RESULT_COLUMNS,
        )
        results.to_csv(path, index=False)

        with pytest.raises(ValueError, match="duplicate sample IDs"):
            transcription_pipeline.load_existing_results(path)

    def test_existing_checkpoint_is_loaded(self, tmp_path):
        path = tmp_path / "transcriptions.csv"

        existing = pd.DataFrame(
            [
                make_existing_record(
                    "sample_1",
                    0,
                    status="ok",
                    transcription_raw="hello world",
                ),
                make_existing_record(
                    "sample_2",
                    1,
                    status="error",
                ),
            ],
            columns=transcription_pipeline.RESULT_COLUMNS,
        )

        existing.to_csv(path, index=False)

        results = transcription_pipeline.load_existing_results(path)

        assert len(results) == 2

        assert list(results.columns) == transcription_pipeline.RESULT_COLUMNS

        assert results.iloc[0]["sample_id"] == "sample_1"
        assert results.iloc[0]["status"] == "ok"
        assert results.iloc[0]["transcription_raw"] == "hello world"

        assert results.iloc[1]["sample_id"] == "sample_2"
        assert results.iloc[1]["status"] == "error"


class TestUpsertResult:

    def test_new_sample_is_inserted(self):
        results = pd.DataFrame(columns=transcription_pipeline.RESULT_COLUMNS)
        record = make_existing_record("sample_1", 0)

        updated = transcription_pipeline.upsert_result(results, record)

        assert len(updated) == 1
        assert updated.iloc[0]["sample_id"] == "sample_1"

    def test_existing_sample_is_replaced(self):
        results = pd.DataFrame(
            [make_existing_record("sample_1", 0, status="error")],
            columns=transcription_pipeline.RESULT_COLUMNS,
        )

        replacement = make_existing_record(
            "sample_1",
            0,
            status="ok",
            transcription_raw="retry worked",
        )

        updated = transcription_pipeline.upsert_result(results, replacement)

        assert len(updated) == 1
        assert updated.iloc[0]["status"] == "ok"
        assert updated.iloc[0]["transcription_raw"] == "retry worked"

    def test_results_are_sorted_by_manifest_index(self):
        results = pd.DataFrame(
            [make_existing_record("sample_2", 1)],
            columns=transcription_pipeline.RESULT_COLUMNS,
        )

        updated = transcription_pipeline.upsert_result(
            results,
            make_existing_record("sample_1", 0),
        )

        assert updated["sample_id"].tolist() == ["sample_1", "sample_2"]


class TestSaveResultsAtomic:

    def test_results_are_saved_and_temp_file_removed(self, tmp_path):
        path = tmp_path / "results" / "transcriptions.csv"
        results = pd.DataFrame(
            [make_existing_record("sample_1", 0)],
            columns=transcription_pipeline.RESULT_COLUMNS,
        )

        transcription_pipeline.save_results_atomic(results, path)

        assert path.exists()
        assert not path.with_name(path.name + ".tmp").exists()

        loaded = pd.read_csv(path)
        assert loaded.iloc[0]["sample_id"] == "sample_1"