"""
Tests for the script to run the transcription pipeline

The transcription pipeline itself is mocked in main() tests because its behaviour is
covered separately in test_scoring_pipeline.py.
"""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pathlib import Path

import scripts.asr.run_transcription as run_transcription


def make_run_config():
    return {
        "manifest_path": "/tmp/manifest.csv",
        "manifest_sha256": "abc123",
        "processed_root": "/tmp/data/processed",
        "whisper": {
            "model_name": "openai/whisper-tiny",
            "decode_mode": "sequential_long_form",
            "device": "cpu",
            "hardware": "CPU",
            "precision": "torch.float32",
        },
        "transcription": {
            "duration_tolerance_s": 0.5,
        },
        "software": {
            "python": "3.test",
            "torch": "2.test",
            "transformers": "5.test",
        },
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }


class DummyTranscriber:
    def __init__(self):
        self.description = {
            "model_name": "openai/whisper-tiny",
            "decode_mode": "sequential_long_form",
            "device": "cpu",
            "hardware": "CPU",
            "precision": "torch.float32",
        }

    def describe(self):
        return dict(self.description)


class TestParseArgs:

    def test_required_arguments_and_defaults(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_transcription.py",
                "--manifest", "manifest.csv",
                "--run-name", "test_run",
                "--model", "openai/whisper-tiny"
            ],
        )

        args = run_transcription.parse_args()

        assert args.manifest.name == "manifest.csv"
        assert args.run_name == "test_run"
        assert args.model == "openai/whisper-tiny"
        assert args.decode_mode == "sequential_long_form"
        assert args.device is None
        assert args.fp32 is False
        assert args.fail_fast is False
        assert args.retry_errors is False
        assert args.duration_tolerance == pytest.approx(0.5)

        assert (
            args.processed_root == run_transcription.DEFAULT_PROCESSED_ROOT
        )

    def test_optional_flags_are_parsed(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_transcription.py",
                "--manifest", "manifest.csv",
                "--run-name", "test_run",
                "--model", "openai/whisper-tiny",
                "--decode-mode", "short_form",
                "--device", "cpu",
                "--fp32",
                "--fail-fast",
                "--retry-errors",
                "--duration-tolerance", "0.25",
                "--processed-root", "custom/processed"
            ],
        )

        args = run_transcription.parse_args()

        assert args.decode_mode == "short_form"
        assert args.device == "cpu"
        assert args.fp32 is True
        assert args.fail_fast is True
        assert args.retry_errors is True
        assert args.duration_tolerance == pytest.approx(0.25)
        assert args.processed_root == Path("custom/processed")


class TestFileSha256:

    def test_hash_matches_hashlib(self, tmp_path):
        path = tmp_path / "manifest.csv"
        contents = b"sample_id,processed_filepath\n1,a.wav\n"
        path.write_bytes(contents)

        expected = hashlib.sha256(contents).hexdigest()

        assert run_transcription.file_sha256(path) == expected

    def test_hash_changes_when_file_changes(self, tmp_path):
        path = tmp_path / "manifest.csv"

        contents = b"sample_id,processed_filepath\n1,a.wav\n"
        path.write_bytes(contents)
        first_hash = run_transcription.file_sha256(path)

        second_contents = b"sample_id,processed_filepath\n2,b.wav\n"
        path.write_bytes(second_contents)
        second_hash = run_transcription.file_sha256(path)

        assert first_hash != second_hash

    def test_hash_same_when_file_same(self, tmp_path):
        path = tmp_path / "manifest.csv"
        
        contents = b"sample_id,processed_filepath\n1,a.wav\n"
        path.write_bytes(contents)
        first_hash = run_transcription.file_sha256(path)

        second_hash = run_transcription.file_sha256(path)

        assert first_hash == second_hash


class TestBuildRunConfig:

    def test_config_contains_manifest_transcriber_and_tolerance(self, tmp_path):
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("sample_id,processed_filepath\n", encoding="utf-8")

        processed_root = tmp_path / "data" / "processed"

        args = SimpleNamespace(
            manifest=manifest,
            processed_root=processed_root,
            duration_tolerance=0.5,
        )

        transcriber = DummyTranscriber()

        config = run_transcription.build_run_config(
            args=args,
            transcriber=transcriber,
            manifest_hash="abc123",
        )

        assert config["manifest_path"] == str(manifest.resolve())
        assert config["manifest_sha256"] == "abc123"
        assert config["processed_root"] == str(processed_root.resolve())
        assert config["whisper"] == transcriber.describe()
        assert config["transcription"]["duration_tolerance_s"] == pytest.approx(0.5)
        assert "python" in config["software"]
        assert "torch" in config["software"]
        assert "transformers" in config["software"]
        assert "created_at_utc" in config


class TestValidateResumeConfig:

    def test_identical_configs_are_accepted(self):
        previous = make_run_config()
        current = copy.deepcopy(previous)

        run_transcription.validate_resume_config(previous, current)

    @pytest.mark.parametrize(
        "path,new_value,expected_name",
        [
            (("manifest_sha256",), "different", "manifest"),
            (("processed_root",), "/other/processed", "processed_root"),
            (("whisper", "model_name"), "openai/whisper-base", "model_name"),
            (("whisper", "decode_mode"), "short_form", "decode_mode"),
            (("whisper", "precision"), "torch.float16", "precision"),
            (("whisper", "device"), "cuda", "device"),
            (("whisper", "hardware"), "RTX 4060", "hardware"),
            (("software", "torch"), "different", "torch"),
            (("software", "transformers"), "different", "transformers"),
            (("transcription", "duration_tolerance_s"), 1.0, "duration_tolerance_s"),
        ],
    )
    def test_changed_experimental_setting_fails(
        self,
        path,
        new_value,
        expected_name,
    ):
        previous = make_run_config()
        current = copy.deepcopy(previous)

        target = current
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = new_value

        with pytest.raises(ValueError, match=expected_name):
            run_transcription.validate_resume_config(previous, current)

    def test_multiple_mismatches_are_all_reported(self):
        previous = make_run_config()
        current = copy.deepcopy(previous)
        
        current["whisper"]["model_name"] = "openai/whisper-large"
        current["software"]["torch"] = "99.test"

        with pytest.raises(ValueError) as excinfo:
            run_transcription.validate_resume_config(previous, current)
            
        error_msg = str(excinfo.value)
        assert "model_name:" in error_msg
        assert "torch:" in error_msg
        assert "Cannot resume this run because its configuration has changed" in error_msg

    def test_created_at_time_is_not_part_of_resume_comparison(self):
        previous = make_run_config()
        current = copy.deepcopy(previous)
        current["created_at_utc"] = "2099-01-01T00:00:00+00:00"

        run_transcription.validate_resume_config(previous, current)

class TestPrepareRunDirectory:

    def test_new_run_creates_config_and_manifest_snapshot(self, tmp_path):
        run_dir = tmp_path / "runs" / "test_run"
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text(
            "sample_id,processed_filepath\nsample_1,a.wav\n",
            encoding="utf-8",
        )

        config = make_run_config()

        snapshot = run_transcription.prepare_run_directory(
            run_dir=run_dir,
            manifest_path=manifest_path,
            run_config=config,
        )

        assert run_dir.exists()
        assert snapshot == run_dir / "manifest_snapshot.csv"
        assert snapshot.read_text(encoding="utf-8") == manifest_path.read_text(encoding="utf-8")

        with open(run_dir / "run_config.json", encoding="utf-8") as file:
            saved_config = json.load(file)

        assert saved_config == config

    def test_existing_valid_run_is_resumed_without_replacing_snapshot(self, tmp_path):
        run_dir = tmp_path / "run"
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("original manifest", encoding="utf-8")

        config = make_run_config()

        snapshot = run_transcription.prepare_run_directory(
            run_dir,
            manifest_path,
            config,
        )

        manifest_path.write_text("changed source manifest", encoding="utf-8")

        resumed_snapshot = run_transcription.prepare_run_directory(
            run_dir,
            manifest_path,
            copy.deepcopy(config),
        )

        assert resumed_snapshot == snapshot
        assert snapshot.read_text(encoding="utf-8") == "original manifest"

    def test_existing_run_without_config_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "manifest_snapshot.csv").write_text(
            "sample_id,processed_filepath\n",
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="run_config.json"):
            run_transcription.prepare_run_directory(
                run_dir,
                tmp_path / "manifest.csv",
                make_run_config(),
            )

    def test_existing_run_without_manifest_snapshot_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with open(run_dir / "run_config.json", "w", encoding="utf-8") as file:
            json.dump(make_run_config(), file)

        with pytest.raises(FileNotFoundError, match="manifest_snapshot.csv"):
            run_transcription.prepare_run_directory(
                run_dir,
                tmp_path / "manifest.csv",
                make_run_config(),
            )

    def test_existing_run_with_changed_config_fails(self, tmp_path):
        run_dir = tmp_path / "run"
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("sample_id,processed_filepath\n", encoding="utf-8")

        previous = make_run_config()
        run_transcription.prepare_run_directory(run_dir, manifest_path, previous)

        current = copy.deepcopy(previous)
        current["whisper"]["model_name"] = "openai/whisper-base"

        with pytest.raises(ValueError, match="model_name"):
            run_transcription.prepare_run_directory(
                run_dir,
                manifest_path,
                current,
            )


class TestMain:

    def test_missing_manifest_fails_before_loading_whisper(self, tmp_path, monkeypatch):
        transcriber_loader = Mock()
        monkeypatch.setattr(run_transcription, "WhisperTranscriber", transcriber_loader)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_transcription.py",
                "--manifest", str(tmp_path / "missing.csv"),
                "--run-name", "test_run",
                "--model", "openai/whisper-tiny"
            ],
        )

        with pytest.raises(FileNotFoundError, match="Manifest does not exist"):
            run_transcription.main()

        transcriber_loader.assert_not_called()

    def test_manifest_directory_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_transcription.py",
                "--manifest", str(tmp_path),
                "--run-name", "test_run",
                "--model", "openai/whisper-tiny"
            ],
        )

        with pytest.raises(ValueError, match="Manifest path is not a file"):
            run_transcription.main()

    def test_main_creates_run_and_calls_pipeline_with_snapshot(
        self,
        tmp_path,
        monkeypatch,
    ):
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "processed_filepath": "sample.wav",
                }
            ]
        ).to_csv(manifest_path, index=False)

        processed_root = tmp_path / "data" / "processed"
        results_root = tmp_path / "results" / "asr" / "runs"

        dummy_transcriber = DummyTranscriber()
        transcriber_loader = Mock(return_value=dummy_transcriber)
        pipeline_mock = Mock(return_value=pd.DataFrame())

        monkeypatch.setattr(
            run_transcription,
            "WhisperTranscriber",
            transcriber_loader,
        )
        monkeypatch.setattr(
            run_transcription,
            "run_transcription_manifest",
            pipeline_mock,
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_transcription.py",
                "--manifest", str(manifest_path),
                "--run-name", "test_run",
                "--model", "openai/whisper-tiny",
                "--device", "cpu",
                "--processed-root", str(processed_root),
                "--results-root", str(results_root),
                "--retry-errors",
                "--fail-fast",
                "--duration-tolerance", "0.25"
            ],
        )

        run_transcription.main()

        transcriber_loader.assert_called_once_with(
            model_name="openai/whisper-tiny",
            device="cpu",
            use_fp16=True,
            decode_mode="sequential_long_form",
        )

        run_dir = results_root / "test_run"
        snapshot = run_dir / "manifest_snapshot.csv"

        assert (run_dir / "run_config.json").exists()
        assert snapshot.exists()

        pipeline_mock.assert_called_once()
        kwargs = pipeline_mock.call_args.kwargs

        assert kwargs["transcriber"] is dummy_transcriber
        assert kwargs["processed_root"] == processed_root
        assert kwargs["results_path"] == run_dir / "transcriptions.csv"
        assert kwargs["retry_errors"] is True
        assert kwargs["fail_fast"] is True
        assert kwargs["duration_tolerance_s"] == pytest.approx(0.25)

        assert kwargs["manifest"]["sample_id"].tolist() == ["sample_1"]
        assert kwargs["manifest"]["processed_filepath"].tolist() == [
        "sample.wav"
    ]

    def test_fp32_flag_disables_fp16(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(
            [
                {
                    "sample_id": "sample_1",
                    "processed_filepath": "sample.wav",
                }
            ]
        ).to_csv(manifest_path, index=False)
        
        dummy_transcriber = DummyTranscriber()
        transcriber_loader = Mock(return_value=dummy_transcriber)

        monkeypatch.setattr(
            run_transcription, 
            "WhisperTranscriber", 
            transcriber_loader
        )

        monkeypatch.setattr(
            run_transcription,
            "run_transcription_manifest",
            Mock(return_value=pd.DataFrame()),
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_transcription.py",
                "--manifest", str(manifest_path),
                "--run-name", "fp32_run",
                "--model", "openai/whisper-tiny",
                "--fp32",
                "--results-root",
                str(tmp_path / "runs"),
            ],
        )

        run_transcription.main()

        assert transcriber_loader.call_args.kwargs["use_fp16"] is False

