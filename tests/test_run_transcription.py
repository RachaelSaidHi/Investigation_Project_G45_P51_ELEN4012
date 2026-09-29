"""
Tests for the script to run the transcription pipeline
"""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import scripts.asr.run_transcription as run_transcription


def make_run_config():
    return {
        "manifest_path": "/tmp/manifest.csv",
        "manifest_sha256": "abc123",
        "data_root": "/tmp/data/raw",
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
                "--duration-tolerance", "0.25"
            ],
        )

        args = run_transcription.parse_args()

        assert args.decode_mode == "short_form"
        assert args.device == "cpu"
        assert args.fp32 is True
        assert args.fail_fast is True
        assert args.retry_errors is True
        assert args.duration_tolerance == pytest.approx(0.25)


class TestFileSha256:

    def test_hash_matches_hashlib(self, tmp_path):
        path = tmp_path / "manifest.csv"
        contents = b"sample_id,filepath\n1,a.wav\n"
        path.write_bytes(contents)

        expected = hashlib.sha256(contents).hexdigest()

        assert run_transcription.file_sha256(path) == expected

    def test_hash_changes_when_file_changes(self, tmp_path):
        path = tmp_path / "manifest.csv"

        contents = b"sample_id,filepath\n1,a.wav\n"
        path.write_bytes(contents)
        first_hash = run_transcription.file_sha256(path)

        contents = b"2,b.wav\n"
        path.write_bytes(contents)
        second_hash = run_transcription.file_sha256(path)

        assert first_hash != second_hash

    def test_hash_same_when_file_same(self, tmp_path):
        path = tmp_path / "manifest.csv"
        
        contents = b"sample_id,filepath\n1,a.wav\n"
        path.write_bytes(contents)
        first_hash = run_transcription.file_sha256(path)

        second_hash = run_transcription.file_sha256(path)

        assert first_hash == second_hash


class TestBuildRunConfig:

    def test_config_contains_manifest_transcriber_and_tolerance(self, tmp_path):
        manifest = tmp_path / "manifest.csv"
        manifest.write_text("sample_id,filepath\n", encoding="utf-8")

        data_root = tmp_path / "data" / "raw"

        args = SimpleNamespace(
            manifest=manifest,
            data_root=data_root,
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
        assert config["data_root"] == str(data_root.resolve())
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
            (("data_root",), "/other/data", "data_root"),
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

    def test_created_at_time_is_not_part_of_resume_comparison(self):
        previous = make_run_config()
        current = copy.deepcopy(previous)
        current["created_at_utc"] = "2099-01-01T00:00:00+00:00"

        run_transcription.validate_resume_config(previous, current)
