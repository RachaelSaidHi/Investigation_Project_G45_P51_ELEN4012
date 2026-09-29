"""
Tests for the scoring pipeline functionality

Normalization and WER are mocked where appropriate so these tests focus on
pipeline behaviour rather than re-testing normalize.py and metrics.py.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import src.asr.scoring_pipeline as scoring_pipeline


def make_manifest(rows=None):
    if rows is None:
        rows = [
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "reference_transcript": "hello world",
            }
        ]

    return pd.DataFrame(rows)


def make_transcriptions(rows=None):
    if rows is None:
        rows = [
            {
                "sample_id": "sample_1",
                "status": "ok",
                "transcription_raw": "hello world",
                "model_name": "openai/whisper-tiny",
                "decode_mode": "sequential_long_form",
            }
        ]

    return pd.DataFrame(rows)


def make_wer_result(
    wer=0.25,
    substitutions=1,
    deletions=0,
    insertions=0,
    hits=3,
    n_reference_words=4,
):
    return SimpleNamespace(
        wer=wer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        hits=hits,
        n_reference_words=n_reference_words,
    )


class TestValidateManifest:

    def test_valid_manifest_is_accepted(self):
        scoring_pipeline.validate_manifest(make_manifest())

    def test_missing_required_column_fails(self):
        manifest = pd.DataFrame(
            [{"sample_id": "sample_1", "dataset": "saa"}]
        )

        with pytest.raises(ValueError, match="missing required columns"):
            scoring_pipeline.validate_manifest(manifest)

    def test_missing_sample_id_fails(self):
        manifest = make_manifest()
        manifest.loc[0, "sample_id"] = None

        with pytest.raises(ValueError, match="missing sample_id"):
            scoring_pipeline.validate_manifest(manifest)

    def test_duplicate_sample_ids_fail(self):
        manifest = make_manifest(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "saa",
                    "reference_transcript": "one",
                },
                {
                    "sample_id": "sample_1",
                    "dataset": "saa",
                    "reference_transcript": "two",
                },
            ]
        )

        with pytest.raises(ValueError, match="duplicate sample_id"):
            scoring_pipeline.validate_manifest(manifest)


class TestValidateTranscriptions:

    def test_valid_transcriptions_are_accepted(self):
        scoring_pipeline.validate_transcriptions(make_transcriptions())

    def test_missing_required_column_fails(self):
        transcriptions = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "status": "ok",
                }
            ]
        )

        with pytest.raises(ValueError, match="missing required columns"):
            scoring_pipeline.validate_transcriptions(transcriptions)

    def test_missing_sample_id_fails(self):
        transcriptions = make_transcriptions()
        transcriptions.loc[0, "sample_id"] = None

        with pytest.raises(ValueError, match="missing sample_id"):
            scoring_pipeline.validate_transcriptions(transcriptions)

    def test_duplicate_sample_ids_fail(self):
        transcriptions = make_transcriptions(
            [
                {
                    "sample_id": "sample_1",
                    "status": "ok",
                    "transcription_raw": "one",
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                },
                {
                    "sample_id": "sample_1",
                    "status": "ok",
                    "transcription_raw": "two",
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                },
            ]
        )

        with pytest.raises(ValueError, match="duplicate sample_id"):
            scoring_pipeline.validate_transcriptions(transcriptions)


class TestValidateSampleIds:

    def test_missing_transcription_rows_are_allowed(self):
        manifest = make_manifest(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "saa",
                    "reference_transcript": "one",
                },
                {
                    "sample_id": "sample_2",
                    "dataset": "saa",
                    "reference_transcript": "two",
                },
            ]
        )

        transcriptions = make_transcriptions(
            [
                {
                    "sample_id": "sample_1",
                    "status": "ok",
                    "transcription_raw": "one",
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                }
            ]
        )

        scoring_pipeline.validate_sample_ids(manifest, transcriptions)

    def test_unknown_transcription_sample_id_fails(self):
        manifest = make_manifest()

        transcriptions = make_transcriptions(
            [
                {
                    "sample_id": "unknown_sample",
                    "status": "ok",
                    "transcription_raw": "hello",
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                }
            ]
        )

        with pytest.raises(ValueError, match="not present in the manifest"):
            scoring_pipeline.validate_sample_ids(manifest, transcriptions)


class TestEmptyMetrics:

    def test_all_metric_fields_are_none(self):
        metrics = scoring_pipeline.empty_metrics()

        assert metrics == {
            "wer": None,
            "substitutions": None,
            "deletions": None,
            "insertions": None,
            "hits": None,
            "n_reference_words": None,
        }


class TestBuildPendingRecord:

    def test_pending_record_has_expected_statuses(self):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
            }
        )

        record = scoring_pipeline.build_pending_record(row)

        assert record["sample_id"] == "sample_1"
        assert record["transcription_status"] == "pending"
        assert record["scoring_status"] == "pending_transcription"
        assert record["model_name"] is None
        assert record["wer"] is None


class TestBuildErrorRecord:

    def test_transcription_error_record(self):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "model_name": "openai/whisper-tiny",
                "decode_mode": "sequential_long_form",
                "status": "error",
            }
        )

        record = scoring_pipeline.build_error_record(row)

        assert record["transcription_status"] == "error"
        assert record["scoring_status"] == "transcription_error"
        assert record["scoring_error_type"] is None
        assert record["wer"] is None

    def test_scoring_error_record_stores_exception(self):
        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "saa",
                "model_name": "openai/whisper-tiny",
                "decode_mode": "sequential_long_form",
                "status": "ok",
            }
        )

        record = scoring_pipeline.build_error_record(
            row,
            ValueError("normalization failed"),
        )

        assert record["transcription_status"] == "ok"
        assert record["scoring_status"] == "scoring_error"
        assert record["scoring_error_type"] == "ValueError"
        assert record["scoring_error_message"] == "normalization failed"
        assert record["wer"] is None

