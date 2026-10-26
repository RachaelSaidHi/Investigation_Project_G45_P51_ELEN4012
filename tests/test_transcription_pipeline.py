"""
Tests for the transcription pipeline functionality
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.asr import transcription_pipeline

import numpy as np
from unittest.mock import Mock


# Helpers

def make_transcription_result(
    text="decoded transcript",
    duration_s=10.0,
    feature_extraction_s=0.10,
    inference_s=0.50,
    decode_s=0.10,
):
    elapsed_s = feature_extraction_s + inference_s + decode_s

    return SimpleNamespace(
        text=text,
        duration_s=duration_s,
        feature_extraction_s=feature_extraction_s,
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

    def transcribe(self, audio):
        self.calls.append(audio)

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


# @pytest.fixture
# def make_audio_file(tmp_path):
#     """
#     Create a fake file that passes validate_audio_path().
#     """

#     def _make(filename="sample.wav"):
#         data_root = tmp_path / "raw"
#         path = data_root / filename
#         path.parent.mkdir(parents=True, exist_ok=True)
#         path.write_bytes(b"dummy audio")
#         return data_root, path

#     return _make


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
            "filepath": "raw/sample.mp3",
            "processed_filepath": "saa/sample.wav",
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


@pytest.fixture
def processed_audio():
    return np.zeros(
        10 * 16000,
        dtype=np.float32,
    )


@pytest.fixture
def mock_audio_loader(
    monkeypatch,
    processed_audio,
):
    loader = Mock(
        return_value=processed_audio
    )

    monkeypatch.setattr(
        transcription_pipeline,
        "load_processed_audio",
        loader,
    )

    return loader


class TestValidateManifest:

    def test_valid_manifest_is_accepted(self):
        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1",
                "processed_filepath": "sample.wav",
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
                "processed_filepath": "sample.wav",
            }
        ])

        with pytest.raises(ValueError, match="missing sample_id"):
            transcription_pipeline.validate_manifest(manifest)

    def test_missing_processed_filepath_fails(self):
        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1",
                "processed_filepath": None,
            }
        ])

        with pytest.raises(
            ValueError, 
            match="missing processed_filepath"
        ):
            transcription_pipeline.validate_manifest(manifest)

    def test_duplicate_sample_ids_fail(self):
        manifest = pd.DataFrame([
            {"sample_id": "sample_1", "processed_filepath": "a.wav"},
            {"sample_id": "sample_1", "processed_filepath": "b.wav"},
        ])

        with pytest.raises(ValueError, match="duplicate sample_id"):
            transcription_pipeline.validate_manifest(manifest)

    @pytest.mark.parametrize(
        "invalid_sample_id",
        [
            "",
            " ",
            "   ",
        ],
    )
    def test_empty_sample_id_fails(
        self,
        invalid_sample_id,
    ):
        manifest = pd.DataFrame([
        {
            "sample_id": invalid_sample_id,
            "processed_filepath": "sample.wav",
        }
        ])

        with pytest.raises(ValueError, match="empty sample_id"):
            transcription_pipeline.validate_manifest(manifest)

    @pytest.mark.parametrize(
        "processed_filepath",
        [
            "", 
            " ", 
            "   "],
    )
    def test_empty_processed_filepath_fails(
        self,
        processed_filepath,
    ):
        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1",
                "processed_filepath": processed_filepath,
            }
        ])

        with pytest.raises(
            ValueError,
            match="empty processed_filepath",
        ):
            transcription_pipeline.validate_manifest(
                manifest
            )


# class TestValidateAudioPath:

#     def test_existing_file_is_returned(self, make_audio_file):
#         data_root, path = make_audio_file("sample.wav")
#         row = pd.Series({"filepath": "sample.wav"})

#         result = transcription_pipeline.validate_audio_path(row, data_root)

#         assert result == path

#     def test_missing_file_fails(self, tmp_path):
#         data_root = tmp_path / "raw"
#         data_root.mkdir()

#         row = pd.Series({"filepath": "missing.wav"})

#         with pytest.raises(FileNotFoundError, match="does not exist"):
#             transcription_pipeline.validate_audio_path(row, data_root)

#     def test_directory_path_fails(self, tmp_path):
#         data_root = tmp_path / "raw"
#         data_root.mkdir()

#         directory = data_root / "not_audio"
#         directory.mkdir()

#         row = pd.Series({"filepath": "not_audio"})

#         with pytest.raises(ValueError, match="not a file"):
#             transcription_pipeline.validate_audio_path(row, data_root)


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
                "filepath": "raw/sample.mp3",
                "processed_filepath": "saa/sample.wav",
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

        assert (record["processed_filepath"] == "saa/sample.wav")

    def test_exact_duration_tolerance_is_not_a_mismatch(self, transcriber_config):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "filepath": "raw/sample.mp3",
                "processed_filepath": "saa/sample.wav",
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
                "filepath": "raw/sample.mp3",
                "processed_filepath": "saa/sample.wav",
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
                "filepath": "raw/sample.mp3",
                "processed_filepath": "saa/sample.wav"
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
                "filepath": "raw/sample.mp3",
                "processed_filepath": "saa/sample.wav",
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

        assert (record["processed_filepath"] == "saa/sample.wav")


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


class TestRunTranscriptionManifest:

    def test_successful_sample_is_transcribed_and_checkpointed(
        self,
        tmp_path,
        make_transcriber,
        mock_audio_loader,
        processed_audio,
    ):
        processed_root = tmp_path / "processed"

        results_path = tmp_path / "results" / "transcriptions.csv"

        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "saa",
                    "filepath": "raw/sample.mp3",
                    "processed_filepath": "saa/sample.wav",
                    "duration_s": 10.0,
                }
            ]
        )

        transcriber = make_transcriber(
            result=make_transcription_result(
                text="hello world",
                duration_s=10.0,
            )
        )

        results = (
            transcription_pipeline.run_transcription_manifest(
                manifest=manifest,
                transcriber=transcriber,
                processed_root=processed_root,
                results_path=results_path,
            )
        )

        mock_audio_loader.assert_called_once_with(
            processed_root=processed_root,
            processed_filepath="saa/sample.wav",
        )


        assert len(transcriber.calls) == 1
        assert transcriber.calls[0] is processed_audio
        assert len(results) == 1
        assert results.iloc[0]["status"] == "ok"
        assert results.iloc[0]["transcription_raw"] == "hello world"
        assert results_path.exists()

    def test_successful_sample_is_skipped_when_resuming(
        self,
        tmp_path,
        mock_audio_loader,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"
        results_path = tmp_path / "transcriptions.csv"

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        first_transcriber = make_transcriber()

        transcription_pipeline.run_transcription_manifest(
            manifest,
            first_transcriber,
            processed_root,
            results_path,
        )

        assert len(first_transcriber.calls) == 1

        mock_audio_loader.reset_mock()
    
        second_transcriber = make_transcriber()

        results = transcription_pipeline.run_transcription_manifest(
            manifest,
            second_transcriber,
            processed_root,
            results_path,
        )

        mock_audio_loader.assert_not_called()

        assert second_transcriber.calls == []
        assert results.iloc[0]["status"] == "ok"

    def test_transcription_error_is_saved(self, tmp_path, mock_audio_loader, make_transcriber):
        processed_root = tmp_path / "processed"
        results_path = tmp_path / "transcriptions.csv"

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        failing_transcriber = make_transcriber(
            error=ValueError("bad audio")
        )

        results = transcription_pipeline.run_transcription_manifest(
            manifest,
            failing_transcriber,
            processed_root,
            results_path,
        )

        assert results.iloc[0]["status"] == "error"
        assert results.iloc[0]["error_type"] == "ValueError"
        assert results.iloc[0]["error_message"] == "bad audio"

        saved = pd.read_csv(results_path)
        assert saved.iloc[0]["status"] == "error"

    def test_missing_audio_becomes_error_record(
        self,
        tmp_path,
        monkeypatch,
        make_transcriber,   
    ):
        processed_root = tmp_path / "processed"

        loader = Mock(
            side_effect=FileNotFoundError(
                "Processed audio file not found"
            )
        )

        monkeypatch.setattr(
            transcription_pipeline,
            "load_processed_audio",
            loader,
        )

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        results_path = tmp_path / "transcriptions.csv"
        transcriber = make_transcriber()

        results = transcription_pipeline.run_transcription_manifest(
            manifest,
            transcriber,
            processed_root,
            results_path,
        )

        assert transcriber.calls == []
        assert results.iloc[0]["status"] == "error"
        assert results.iloc[0]["error_type"] == "FileNotFoundError"

    def test_previous_error_is_not_retried_by_default(
        self,
        tmp_path,
        mock_audio_loader,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"
        results_path = tmp_path / "transcriptions.csv"

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        failing_transcriber = make_transcriber(
            error=RuntimeError("first failure")
        )

        transcription_pipeline.run_transcription_manifest(
            manifest,
            failing_transcriber,
            processed_root,
            results_path,
        )

        working_transcriber = make_transcriber()
        mock_audio_loader.reset_mock()

        results = transcription_pipeline.run_transcription_manifest(
            manifest,
            working_transcriber,
            processed_root,
            results_path,
            retry_errors=False,
        )

        assert len(failing_transcriber.calls) == 1
        assert working_transcriber.calls == []
        assert results.iloc[0]["status"] == "error"
        assert results.iloc[0]["error_type"] == "RuntimeError"
        assert results.iloc[0]["error_message"] == "first failure"
        
        assert mock_audio_loader.call_count == 0
        assert working_transcriber.calls == []

    def test_previous_error_is_retried_when_requested(
        self,
        tmp_path,
        mock_audio_loader,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"
        results_path = tmp_path / "transcriptions.csv"

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        failing_transcriber = make_transcriber(
            error=RuntimeError("first failure")
        )
        
        transcription_pipeline.run_transcription_manifest(
            manifest,
            failing_transcriber,
            processed_root,
            results_path,
        )

        retry_transcriber = make_transcriber(
            result=make_transcription_result(text="retry worked")
        )

        results = transcription_pipeline.run_transcription_manifest(
            manifest,
            retry_transcriber,
            processed_root,
            results_path,
            retry_errors=True,
        )

        assert len(failing_transcriber.calls) == 1
        assert len(retry_transcriber.calls) == 1
        assert results.iloc[0]["status"] == "ok"
        assert results.iloc[0]["error_type"] == None
        assert results.iloc[0]["error_message"] == None
        assert results.iloc[0]["transcription_raw"] == "retry worked"

    def test_fail_fast_saves_error_before_raising(
        self,
        tmp_path,
        mock_audio_loader,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"
        results_path = tmp_path / "transcriptions.csv"

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        failing_transcriber = make_transcriber(
            error=ValueError("broken sample")
        )

        with pytest.raises(RuntimeError, match="Transcription failed for sample"):
            transcription_pipeline.run_transcription_manifest(
                manifest,
                failing_transcriber,
                processed_root,
                results_path,
                fail_fast=True,
            )

        assert results_path.exists()

        saved = pd.read_csv(results_path)
        assert saved.iloc[0]["status"] == "error"
        assert saved.iloc[0]["error_type"] == "ValueError"

    def test_existing_result_not_in_manifest_fails(
        self,
        tmp_path,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"
        results_path = tmp_path / "transcriptions.csv"

        existing = pd.DataFrame(
            [make_existing_record("unknown_sample", 0)],
            columns=transcription_pipeline.RESULT_COLUMNS,
        )
        existing.to_csv(results_path, index=False)

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1", 
                "processed_filepath": "sample.wav"
            }
        ])

        with pytest.raises(ValueError, match="not present in this manifest"):
            transcription_pipeline.run_transcription_manifest(
                manifest,
                make_transcriber(),
                processed_root,
                results_path,
            )

    def test_manifest_order_is_preserved_in_results(
        self,
        tmp_path,
        mock_audio_loader,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"

        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "sample_a",
                    "processed_filepath": "a.wav",
                },
                {
                    "sample_id": "sample_b",
                    "processed_filepath": "b.wav",
                },
            ]
        )

        results = transcription_pipeline.run_transcription_manifest(
            manifest,
            make_transcriber(),
            processed_root,
            tmp_path / "transcriptions.csv",
        )

        assert results["sample_id"].tolist() == [
            "sample_a",
            "sample_b",
        ]

        assert results["manifest_index"].tolist() == [0, 1]
        assert mock_audio_loader.call_count == 2

    def test_without_fail_fast_continues_after_error(
        self,
        tmp_path,
        monkeypatch,
        make_transcriber,
    ):
        processed_root = tmp_path / "processed"

        good_audio = np.ones(
            16000,
            dtype=np.float32,
        )

        manifest = pd.DataFrame([
            {
                "sample_id": "sample_1",
                "processed_filepath": "bad.wav",
                "dataset": "saa",
            },
            {
                "sample_id": "sample_2",
                "processed_filepath": "good.wav",
                "dataset": "saa",
            },
        ])

        def fake_load_processed_audio(
            processed_root,
            processed_filepath,
        ):
            if processed_filepath == "bad.wav":
                raise RuntimeError("fake failure")

            return good_audio

        monkeypatch.setattr(
            transcription_pipeline,
            "load_processed_audio",
            fake_load_processed_audio,
        )

        transcriber = make_transcriber(
            result=make_transcription_result(
                text="successful transcript"
            )
        )

        results_path = (tmp_path / "transcriptions.csv")

        results = (
            transcription_pipeline.run_transcription_manifest(
                manifest=manifest,
                transcriber=transcriber,
                processed_root=processed_root,
                results_path=results_path,
                fail_fast=False,
            )
        )

        # First sample fails while loading, so only the second
        # sample reaches Whisper.
        assert len(transcriber.calls) == 1
        assert transcriber.calls[0] is good_audio

        results = results.set_index("sample_id")

        assert results.loc["sample_1", "status"] == "error"
        assert results.loc["sample_1", "error_type"] == "RuntimeError"
        assert results.loc["sample_1", "error_message"] == "fake failure"
        assert results.loc["sample_2", "status"] == "ok"
        assert results.loc["sample_2", "transcription_raw"] == "successful transcript"

    