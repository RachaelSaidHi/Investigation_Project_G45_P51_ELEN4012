"""
Tests for the script to run the transcription pipeline

The scoring pipeline itself is mocked in main() tests because its behaviour is
covered separately in test_scoring_pipeline.py.
"""

import json
import sys
from unittest.mock import Mock

import pandas as pd
import pytest

import scripts.asr.run_scoring as run_scoring


class TestParseArgs:

    def test_required_argument_and_defaults(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir", "results/asr/runs/test_run"
            ],
        )

        args = run_scoring.parse_args()

        assert args.run_dir.name == "test_run"
        assert args.output_name == "scored.csv"
        assert args.fail_fast is False

    def test_optional_arguments_are_parsed(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir", "results/asr/runs/test_run",
                "--output-name", "custom_scores.csv",
                "--fail-fast",
            ],
        )

        args = run_scoring.parse_args()

        assert args.output_name == "custom_scores.csv"
        assert args.fail_fast is True


class TestBuildScoringConfig:

    def test_config_contains_paths_normalizer_and_versions(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest_snapshot.csv"
        transcriptions_path = tmp_path / "transcriptions.csv"

        manifest_path.write_text("sample_id\n", encoding="utf-8")
        transcriptions_path.write_text("sample_id\n", encoding="utf-8")

        monkeypatch.setattr(
            run_scoring,
            "describe_normalizer",
            lambda: {"name": "test-normalizer"},
        )
        monkeypatch.setattr(
            run_scoring,
            "version",
            lambda package: f"{package}-test-version",
        )

        config = run_scoring.build_scoring_config(
            manifest_path,
            transcriptions_path,
        )

        assert config["manifest_path"] == str(manifest_path.resolve())
        assert config["transcriptions_path"] == str(transcriptions_path.resolve())
        assert config["normalization"] == {"name": "test-normalizer"}
        assert config["software"]["jiwer"] == "jiwer-test-version"
        assert config["software"]["transformers"] == "transformers-test-version"
        assert "python" in config["software"]
        assert "pandas" in config["software"]
        assert "scored_at_utc" in config


class TestMain:

    def test_missing_run_directory_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir", str(tmp_path / "missing_run")
            ],
        )

        with pytest.raises(FileNotFoundError, match="Run directory does not exist"):
            run_scoring.main()

    def test_run_path_that_is_not_directory_fails(self, tmp_path, monkeypatch):
        run_path = tmp_path / "not_a_directory"
        run_path.write_text("file", encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir", str(run_path),
            ],
        )

        with pytest.raises(ValueError, match="Run path is not a directory"):
            run_scoring.main()

    def test_missing_manifest_snapshot_fails(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        pd.DataFrame([{"sample_id": "sample_1"}]).to_csv(
            run_dir / "transcriptions.csv",
            index=False,
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir", str(run_dir),
            ],
        )

        with pytest.raises(FileNotFoundError, match="Manifest snapshot does not exist"):
            run_scoring.main()

    def test_missing_transcriptions_file_fails(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        pd.DataFrame([{"sample_id": "sample_1"}]).to_csv(
            run_dir / "manifest_snapshot.csv",
            index=False,
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir",
                str(run_dir),
            ],
        )

        with pytest.raises(FileNotFoundError, match="Transcription results do not exist"):
            run_scoring.main()

    def test_main_loads_inputs_scores_and_saves_outputs(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        manifest = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "dataset": "saa",
                    "reference_transcript": "hello world",
                }
            ]
        )
        transcriptions = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "status": "ok",
                    "transcription_raw": "hello world",
                    "model_name": "openai/whisper-tiny",
                    "decode_mode": "sequential_long_form",
                }
            ]
        )

        manifest.to_csv(run_dir / "manifest_snapshot.csv", index=False)
        transcriptions.to_csv(run_dir / "transcriptions.csv", index=False)

        expected_results = pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "scoring_status": "ok",
                    "wer": 0.0,
                }
            ]
        )

        scoring_mock = Mock(return_value=expected_results)
        config_mock = Mock(
            return_value={
                "test": "config",
                "scored_at_utc": "2026-01-01T00:00:00+00:00",
            }
        )

        monkeypatch.setattr(run_scoring, "run_scoring", scoring_mock)
        monkeypatch.setattr(run_scoring, "build_scoring_config", config_mock)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir", str(run_dir),
                "--output-name", "custom_scores.csv",
                "--fail-fast",
            ],
        )

        run_scoring.main()

        scoring_mock.assert_called_once()
        kwargs = scoring_mock.call_args.kwargs

        assert kwargs["fail_fast"] is True
        assert kwargs["manifest"]["sample_id"].tolist() == ["sample_1"]
        assert kwargs["transcriptions"]["sample_id"].tolist() == ["sample_1"]

        output_path = run_dir / "custom_scores.csv"
        config_path = run_dir / "scoring_config.json"

        assert output_path.exists()
        assert config_path.exists()

        saved = pd.read_csv(output_path)
        assert saved.iloc[0]["sample_id"] == "sample_1"
        assert saved.iloc[0]["scoring_status"] == "ok"
        assert saved.iloc[0]["wer"] == pytest.approx(0.0)

        with open(config_path, encoding="utf-8") as file:
            saved_config = json.load(file)

        assert saved_config["test"] == "config"

        config_mock.assert_called_once_with(
            run_dir / "manifest_snapshot.csv",
            run_dir / "transcriptions.csv",
        )

    def test_default_output_name_is_scored_csv(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        pd.DataFrame([{"sample_id": "sample_1"}]).to_csv(
            run_dir / "manifest_snapshot.csv",
            index=False,
        )
        pd.DataFrame([{"sample_id": "sample_1"}]).to_csv(
            run_dir / "transcriptions.csv",
            index=False,
        )

        monkeypatch.setattr(
            run_scoring,
            "run_scoring",
            Mock(return_value=pd.DataFrame([{"sample_id": "sample_1"}])),
        )
        monkeypatch.setattr(
            run_scoring,
            "build_scoring_config",
            Mock(return_value={"test": "config"}),
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_scoring.py",
                "--run-dir",
                str(run_dir),
            ],
        )

        run_scoring.main()

        assert (run_dir / "scored.csv").exists()
        assert (run_dir / "scoring_config.json").exists()
