"""
Tests for the scoring pipeline functionality

Normalization and WER are mocked where appropriate so these tests focus on
pipeline behaviour rather than re-testing normalize.py and metrics.py.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import sys

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
        manifest = make_manifest()
        manifest.loc[0, "sample_id"] = invalid_sample_id

        with pytest.raises(ValueError, match="empty sample_id"):
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
        transcriptions = make_transcriptions()

        transcriptions.loc[
            0,
            "sample_id",
        ] = invalid_sample_id

        with pytest.raises(
            ValueError,
            match="empty sample_id",
        ):
            scoring_pipeline.validate_transcriptions(
                transcriptions
            )


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
        assert record["dataset"] == "saa"
        assert record["transcription_status"] == "pending"
        assert record["scoring_status"] == "pending_transcription"
        assert record["reference_norm"] is None
        assert record["hypothesis_norm"] is None
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
        assert record["reference_norm"] is None
        assert record["hypothesis_norm"] is None
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


class TestScoreSample:

    def test_reference_and_hypothesis_are_normalized_and_scored(self, monkeypatch):
        normalization_calls = []

        def fake_normalize(text, dataset, drop_punctuation_words=True):
            normalization_calls.append((text, dataset))

            if text == "THIS IS REFERENCE RAW":
                return "this is reference normalized", 1

            return "this is hypothesis normalized", 2

        metric_calls = []

        def fake_compute_wer(reference, hypothesis):
            metric_calls.append((reference, hypothesis))
            return make_wer_result()

        monkeypatch.setattr(
            scoring_pipeline,
            "normalize_text",
            fake_normalize,
        )
        monkeypatch.setattr(
            scoring_pipeline,
            "compute_wer",
            fake_compute_wer,
        )

        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "afrispeech",
                "reference_transcript": "THIS IS REFERENCE RAW",
                "transcription_raw": "THIS IS HYPOTHESIS RAW",
                "model_name": "openai/whisper-tiny",
                "decode_mode": "sequential_long_form",
                "status": "ok",
            }
        )

        result = scoring_pipeline.score_sample(row)

        assert normalization_calls == [
            ("THIS IS REFERENCE RAW", "afrispeech"),
            ("THIS IS HYPOTHESIS RAW", "afrispeech"),
        ]
        assert metric_calls == [
            ("this is reference normalized", "this is hypothesis normalized")
        ]

        assert result["reference_norm"] == "this is reference normalized"
        assert result["hypothesis_norm"] == "this is hypothesis normalized"
        assert result["n_punctuation_reference"] == 1
        assert result["n_punctuation_hypothesis"] == 2
        assert result["scoring_status"] == "ok"
        assert result["wer"] == pytest.approx(0.25)
        assert result["substitutions"] == 1
        assert result["n_reference_words"] == 4

    def test_empty_normalized_reference_does_not_compute_wer(self, monkeypatch):
        def fake_normalize(text, dataset, drop_punctuation_words=True):
            if text == "REFERENCE RAW":
                return "", 0

            return "some hypothesis", 0

        def should_not_be_called(*args, **kwargs):
            pytest.fail("compute_wer should not be called for an empty reference")

        monkeypatch.setattr(
            scoring_pipeline,
            "normalize_text",
            fake_normalize,
        )
        monkeypatch.setattr(
            scoring_pipeline,
            "compute_wer",
            should_not_be_called,
        )

        row = pd.Series(
            {
                "sample_id": "sample_1",
                "dataset": "afrispeech",
                "reference_transcript": "REFERENCE RAW",
                "transcription_raw": "HYPOTHESIS RAW",
                "model_name": "openai/whisper-tiny",
                "decode_mode": "sequential_long_form",
                "status": "ok",
            }
        )

        result = scoring_pipeline.score_sample(row)

        assert result["scoring_status"] == "empty_reference"
        assert result["wer"] is None
        assert result["substitutions"] is None
        assert result["deletions"] is None
        assert result["insertions"] is None
        assert result["hits"] is None
        assert result["n_reference_words"] is None

    
    def test_score_sample_forwards_drop_punctuation_flag(self, monkeypatch):
        calls = []

        def fake_normalize(text, dataset, drop_punctuation_words=True):
            calls.append(drop_punctuation_words)
            return "hello world", 0

        monkeypatch.setattr(scoring_pipeline, "normalize_text", fake_normalize)

        row = {
            "sample_id": "s1", "dataset": "afrispeech", "model_name": "m", "decode_mode": "d",
            "status": "ok", "reference_transcript": "Hello world", "transcription_raw": "hello world",
        }
        scoring_pipeline.score_sample(row, drop_punctuation_words=False)

        assert calls == [False, False]   # reference and hypothesis


class TestRunScoring:

    def scoring_pipeline_error_and_pending_samples_are_routed_correctly(self, monkeypatch):
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
                {
                    "sample_id": "sample_3",
                    "dataset": "saa",
                    "reference_transcript": "three",
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
                },
                {
                    "sample_id": "sample_2",
                    "status": "error",
                    "transcription_raw": None,
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                },
            ]
        )

        def fake_score_sample(row):
            return {
                "sample_id": str(row["sample_id"]),
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "decode_mode": row["decode_mode"],
                "transcription_status": row["status"],
                "scoring_status": "ok",
                "scoring_error_type": None,
                "scoring_error_message": None,
                "reference_norm": "one",
                "hypothesis_norm": "one",
                "n_punctuation_reference": 0,
                "n_punctuation_hypothesis": 0,
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 1,
                "n_reference_words": 1,
                "scored_at_utc": "2026-01-01T00:00:00+00:00",
            }

        monkeypatch.setattr(
            scoring_pipeline,
            "score_sample",
            fake_score_sample,
        )

        results = scoring_pipeline.run_scoring(manifest, transcriptions)
        results = results.set_index("sample_id")

        assert results.loc["sample_1", "scoring_status"] == "ok"
        assert (results.loc["sample_2", "scoring_status"] == "transcription_error")
        assert (results.loc["sample_3", "scoring_status"]== "pending_transcription")

    def test_empty_reference_is_handled_in_run_scoring(
        self,
        monkeypatch,
        capsys,
    ):
        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "afrispeech",
                    "reference_transcript": "REFERENCE RAW",
                }
            ]
        )

        transcriptions = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "status": "ok",
                    "transcription_raw": "HYPOTHESIS RAW",
                    "model_name": "openai/whisper-tiny",
                    "decode_mode": "sequential_long_form",
                }
            ]
        )

        def fake_normalize(text, dataset, drop_punctuation_words=True):
            if text == "REFERENCE RAW":
                return "", 0

            return "hypothesis normalized", 0

        def should_not_be_called(*args, **kwargs):
            pytest.fail(
                "compute_wer should not be called for an empty reference"
            )

        monkeypatch.setattr(
            scoring_pipeline,
            "normalize_text",
            fake_normalize,
        )

        monkeypatch.setattr(
            scoring_pipeline,
            "compute_wer",
            should_not_be_called,
        )

        results = scoring_pipeline.run_scoring(
            manifest,
            transcriptions,
        )

        row = results.iloc[0]

        assert row["scoring_status"] == "empty_reference"
        assert pd.isna(row["wer"])
        assert pd.isna(row["substitutions"])
        assert pd.isna(row["deletions"])
        assert pd.isna(row["insertions"])

        output = capsys.readouterr().out
        assert "EMPTY REFERENCE | WER undefined." in output

    def test_manifest_metadata_is_source_of_truth(self, monkeypatch):
        manifest = make_manifest(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "afrispeech",
                    "reference_transcript": "hello",
                    "filepath": "correct/path.wav",
                }
            ]
        )

        transcriptions = make_transcriptions(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "WRONG_DATASET",
                    "filepath": "wrong/path.wav",
                    "status": "ok",
                    "transcription_raw": "hello",
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                }
            ]
        )

        datasets_seen = []

        def fake_normalize(text, dataset, drop_punctuation_words=True):
            datasets_seen.append(dataset)
            return text, 0

        monkeypatch.setattr(
            scoring_pipeline,
            "normalize_text",
            fake_normalize,
        )
        monkeypatch.setattr(
            scoring_pipeline,
            "compute_wer",
            lambda reference, hypothesis: make_wer_result(
                wer=0.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                hits=1,
                n_reference_words=1,
            ),
        )

        results = scoring_pipeline.run_scoring(manifest, transcriptions)

        assert results.iloc[0]["dataset"] == "afrispeech"
        assert datasets_seen == ["afrispeech", "afrispeech"]

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
            scoring_pipeline.run_scoring(manifest, transcriptions)

    def test_scoring_exception_is_recorded_when_not_fail_fast(self, monkeypatch):
        manifest = make_manifest()
        transcriptions = make_transcriptions()

        def fail(_row, drop_punctuation_words=True):
            raise ValueError("fake scoring problem")

        monkeypatch.setattr(scoring_pipeline, "score_sample", fail)

        results = scoring_pipeline.run_scoring(
            manifest,
            transcriptions,
            fail_fast=False,
        )

        row = results.iloc[0]

        assert row["scoring_status"] == "scoring_error"
        assert row["scoring_error_type"] == "ValueError"
        assert row["scoring_error_message"] == "fake scoring problem"
        assert pd.isna(row["wer"])

    def test_fail_fast_raises_on_scoring_exception(self, monkeypatch):
        manifest = make_manifest()
        transcriptions = make_transcriptions()

        def fail(_row):
            raise ValueError("fake scoring problem")

        monkeypatch.setattr(scoring_pipeline, "score_sample", fail)

        with pytest.raises(RuntimeError, match="Scoring failed for sample sample_1"):
            scoring_pipeline.run_scoring(
                manifest,
                transcriptions,
                fail_fast=True,
            )

    def test_result_columns_and_manifest_order_are_preserved(self, monkeypatch):
        manifest = make_manifest(
            [
                {
                    "sample_id": "sample_b",
                    "dataset": "saa",
                    "reference_transcript": "b",
                },
                {
                    "sample_id": "sample_a",
                    "dataset": "saa",
                    "reference_transcript": "a",
                },
            ]
        )

        transcriptions = make_transcriptions(
            [
                {
                    "sample_id": "sample_a",
                    "status": "error",
                    "transcription_raw": None,
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                },
                {
                    "sample_id": "sample_b",
                    "status": "error",
                    "transcription_raw": None,
                    "model_name": "whisper",
                    "decode_mode": "sequential_long_form",
                },
            ]
        )

        results = scoring_pipeline.run_scoring(manifest, transcriptions)

        assert list(results.columns) == scoring_pipeline.RESULT_COLUMNS
        assert results["sample_id"].tolist() == ["sample_b", "sample_a"]
