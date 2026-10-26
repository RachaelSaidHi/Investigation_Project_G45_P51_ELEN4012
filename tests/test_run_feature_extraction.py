"""
Tests for the script that runs DSP feature extraction over a manifest.

The extraction pipeline itself is covered in test_extraction_pipeline.py.
These tests cover feature selection, config loading, run configuration,
resume checks and the multi-feature command-line behaviour. End-to-end runs
use the lightweight rms_test feature on synthetic processed audio.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.dsp.run_feature_extraction as run_feature_extraction
from scripts.dsp.run_feature_extraction import FEATURE_MODULES

FEATURE_MODULES = FEATURE_MODULES + ["rms_test"]

from src.common.run_management import (
    RUN_CONFIG_NAME, 
    MANIFEST_SNAPSHOT_NAME
)

from src.preprocessing.audio import (
    TARGET_SAMPLE_RATE, 
    save_processed_audio
)


PROJECT_FEATURES_DIR = Path("src/dsp/features")
PROJECT_CONFIG_DIR = Path("configs/dsp")

TEST_SPEECH_MASK_CONFIG = {
    "frame_length_s": 0.025,
    "hop_length_s": 0.010,
    "reference_percentile": 95,
    "threshold_db": 30,
    "max_gap_s": 0.10,
    "min_speech_run_s": 0.05,
    "min_total_speech_s": 1.0,
}


def make_tone(duration_s=1.0, amplitude=0.1):
    """
    Create a short synthetic 220 Hz tone.
    """

    time = np.arange(int(TARGET_SAMPLE_RATE * duration_s)) / TARGET_SAMPLE_RATE

    return (amplitude * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def write_json(path, data):
    """
    Write a JSON config file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)


def make_project(tmp_path, sample_ids=("sample_1", "sample_2")):
    """
    Create a manifest, processed audio and config folder for one run.

    Returns the paths needed by the command-line arguments.
    """

    processed_root = tmp_path / "processed"
    config_dir = tmp_path / "configs"
    results_root = tmp_path / "results"

    rows = []

    for sample_id in sample_ids:
        relative_path = f"audio/{sample_id}.wav"

        save_processed_audio(
            make_tone(),
            processed_root / relative_path,
        )

        rows.append(
            {
                "sample_id": sample_id,
                "dataset": "saa",
                "processed_filepath": relative_path,
                "sex": "female",
                "accent_group": "south_african_english",
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    write_json(config_dir / "rms_test.json", {})
    write_json(config_dir / "speech_mask.json", TEST_SPEECH_MASK_CONFIG)

    return SimpleNamespace(
        manifest=manifest_path,
        processed_root=processed_root,
        config_dir=config_dir,
        results_root=results_root,
    )


def make_args(project, run_name="test_run", retry_errors=False, fail_fast=False):
    """
    Return parsed-argument-like settings for run_one_feature().
    """

    return SimpleNamespace(
        manifest=project.manifest,
        run_name=run_name,
        processed_root=project.processed_root,
        results_root=project.results_root,
        config_dir=project.config_dir,
        retry_errors=retry_errors,
        fail_fast=fail_fast,
    )


def set_argv(monkeypatch, project, *features, extra=()):
    """
    Set sys.argv for main() using the given project paths.
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_feature_extraction.py",
            "--manifest", str(project.manifest),
            "--run-name", "test_run",
            "--processed-root", str(project.processed_root),
            "--results-root", str(project.results_root),
            "--config-dir", str(project.config_dir),
            "--feature", *features,
            *extra,
        ],
    )


class TestResolveFeatureNames:

    def test_single_feature(self):
        assert run_feature_extraction.resolve_feature_names(["mfcc"]) == ["mfcc"]

    def test_order_is_kept_and_duplicates_removed(self):
        result = run_feature_extraction.resolve_feature_names(
            ["mfcc", "pitch", "mfcc", "speech_rate_reference"]
        )

        assert result == ["mfcc", "pitch", "speech_rate_reference"]

    def test_all_expands_to_feature_modules(self):
        result = run_feature_extraction.resolve_feature_names(["all"])

        assert result == run_feature_extraction.FEATURE_MODULES
        assert result is not run_feature_extraction.FEATURE_MODULES

    def test_all_with_other_features_raises_error(self):
        with pytest.raises(ValueError, match="either --feature all or a list"):
            run_feature_extraction.resolve_feature_names(["all", "mfcc"])


class TestFeatureModuleList:

    def test_feature_modules_are_unique(self):
        modules = run_feature_extraction.FEATURE_MODULES

        assert len(modules) == len(set(modules))

    def test_every_listed_feature_has_module_and_config(self):
        for feature_name in run_feature_extraction.FEATURE_MODULES:
            assert (PROJECT_FEATURES_DIR / f"{feature_name}.py").is_file()
            assert (PROJECT_CONFIG_DIR / f"{feature_name}.json").is_file()


class TestLoadFeatureModule:

    def test_project_features_load(self):
        for feature_name in run_feature_extraction.FEATURE_MODULES:
                assert (PROJECT_FEATURES_DIR / f"{feature_name}.py").is_file()
                assert (PROJECT_CONFIG_DIR / f"{feature_name}.json").is_file()
        module = run_feature_extraction.load_feature_module(feature_name)

        assert module.FEATURE_NAME == feature_name

    def test_unknown_feature_raises_clear_error(self):
        with pytest.raises(
            ValueError,
            match="Unknown feature 'not_a_feature'",
        ):
            run_feature_extraction.load_feature_module("not_a_feature")

    def test_missing_dependency_inside_feature_is_not_hidden(self, monkeypatch):
        def failing_import(module_path):
            raise ModuleNotFoundError(
                "No module named 'parselmouth'",
                name="parselmouth",
            )

        monkeypatch.setattr(
            run_feature_extraction.importlib,
            "import_module",
            failing_import,
        )

        with pytest.raises(ModuleNotFoundError, match="parselmouth"):
            run_feature_extraction.load_feature_module("speech_rate_acoustic")

    def test_import_path_uses_features_package(self, monkeypatch):
        received = {}

        def recording_import(module_path):
            received["path"] = module_path
            return SimpleNamespace(
                FEATURE_NAME="pitch",
                FEATURE_VERSION="1",
                feature_columns=lambda config: [],
                extract=lambda *args: {},
            )

        monkeypatch.setattr(
            run_feature_extraction.importlib,
            "import_module",
            recording_import,
        )

        run_feature_extraction.load_feature_module("pitch")

        assert received["path"] == "src.dsp.features.pitch"

    def test_incomplete_module_raises_error(self, monkeypatch):
        monkeypatch.setattr(
            run_feature_extraction.importlib,
            "import_module",
            lambda module_path: SimpleNamespace(FEATURE_NAME="pitch"),
        )

        with pytest.raises(TypeError, match="missing required attributes"):
            run_feature_extraction.load_feature_module("pitch")

    def test_mismatched_feature_name_raises_error(self, monkeypatch):
        monkeypatch.setattr(
            run_feature_extraction.importlib,
            "import_module",
            lambda module_path: SimpleNamespace(
                FEATURE_NAME="f0",
                FEATURE_VERSION="1",
                feature_columns=lambda config: [],
                extract=lambda *args: {},
            ),
        )

        with pytest.raises(ValueError, match="FEATURE_NAME = 'f0'"):
            run_feature_extraction.load_feature_module("pitch")


class TestConfigFiles:

    def test_read_json_loads_file(self, tmp_path):
        path = tmp_path / "config.json"
        write_json(path, {"n_mfcc": 13})

        assert run_feature_extraction.read_json(path) == {"n_mfcc": 13}

    def test_read_json_missing_file_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            run_feature_extraction.read_json(tmp_path / "missing.json")

    def test_load_config_combines_feature_and_speech_mask(self, tmp_path):
        write_json(tmp_path / "mfcc.json", {"n_mfcc": 13})
        write_json(tmp_path / "speech_mask.json", TEST_SPEECH_MASK_CONFIG)

        config = run_feature_extraction.load_config(tmp_path, "mfcc")

        assert config == {
            "feature": {"n_mfcc": 13},
            "speech_mask": TEST_SPEECH_MASK_CONFIG,
        }

    def test_load_config_missing_feature_config_raises_error(self, tmp_path):
        write_json(tmp_path / "speech_mask.json", TEST_SPEECH_MASK_CONFIG)

        with pytest.raises(FileNotFoundError, match="pitch.json"):
            run_feature_extraction.load_config(tmp_path, "pitch")

    def test_load_config_missing_speech_mask_raises_error(self, tmp_path):
        write_json(tmp_path / "mfcc.json", {})

        with pytest.raises(FileNotFoundError, match="speech_mask.json"):
            run_feature_extraction.load_config(tmp_path, "mfcc")

    def test_project_configs_load_for_implemented_features(self):
        for feature_name in FEATURE_MODULES:
            config = run_feature_extraction.load_config(
                PROJECT_CONFIG_DIR,
                feature_name,
            )

            assert set(config) == {"feature", "speech_mask"}


class TestRunConfig:

    def test_package_version_of_installed_package(self):
        assert run_feature_extraction.package_version("pandas") == pd.__version__

    def test_package_version_of_missing_package(self):
        result = run_feature_extraction.package_version(
            "definitely-not-an-installed-package-xyz"
        )

        assert result == "not installed"

    def test_build_run_config_contents(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "manifest.csv"
        manifest_path.write_text("sample_id\nsample_1\n", encoding="utf-8")

        monkeypatch.setattr(
            run_feature_extraction,
            "package_version",
            lambda package: f"{package}-test-version",
        )

        module = SimpleNamespace(FEATURE_NAME="mfcc", FEATURE_VERSION="3")
        config = {"feature": {"n_mfcc": 13}, "speech_mask": {}}

        run_config = run_feature_extraction.build_run_config(
            manifest_path=manifest_path,
            processed_root=tmp_path / "processed",
            feature_module=module,
            config=config,
        )

        assert run_config["manifest_sha256"] == run_feature_extraction.file_sha256(manifest_path)
        assert run_config["feature"] == {"name": "mfcc", "version": "3"}
        assert run_config["config"] == config
        assert "python" in run_config["software"]

        for package in run_feature_extraction.RECORDED_PACKAGES:
            assert run_config["software"][package] == f"{package}-test-version"

        assert run_config["paths"] == {
            "manifest": str(manifest_path.resolve()),
            "processed_root": str((tmp_path / "processed").resolve()),
        }
        assert "created_at_utc" in run_config

    def test_machine_specific_keys_are_ignored_on_resume(self):
        assert "paths" in run_feature_extraction.IGNORED_CONFIG_KEYS
        assert "created_at_utc" in run_feature_extraction.IGNORED_CONFIG_KEYS


class TestParseArgs:

    def test_required_arguments_and_defaults(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_feature_extraction.py",
                "--manifest", "data/manifests/dev_manifest.csv",
                "--run-name", "dev_v1",
                "--feature", "mfcc",
            ],
        )

        args = run_feature_extraction.parse_args()

        assert args.manifest == Path("data/manifests/dev_manifest.csv")
        assert args.run_name == "dev_v1"
        assert args.feature == ["mfcc"]
        assert args.processed_root == run_feature_extraction.DEFAULT_PROCESSED_ROOT
        assert args.results_root == run_feature_extraction.DEFAULT_RESULTS_ROOT
        assert args.config_dir == run_feature_extraction.DEFAULT_CONFIG_DIR
        assert args.retry_errors is False
        assert args.fail_fast is False

    def test_several_features_and_flags(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_feature_extraction.py",
                "--manifest", "m.csv",
                "--run-name", "r",
                "--feature", "mfcc", "pitch",
                "--retry-errors",
                "--fail-fast",
            ],
        )

        args = run_feature_extraction.parse_args()

        assert args.manifest == Path("m.csv")
        assert args.run_name == "r"
        assert args.processed_root == run_feature_extraction.DEFAULT_PROCESSED_ROOT
        assert args.results_root == run_feature_extraction.DEFAULT_RESULTS_ROOT
        assert args.config_dir == run_feature_extraction.DEFAULT_CONFIG_DIR
        assert args.feature == ["mfcc", "pitch"]
        assert args.retry_errors is True
        assert args.fail_fast is True

    @pytest.mark.parametrize(
        "missing",
        [
            "--manifest",
            "--run-name",
            "--feature",
        ],
    )
    def test_missing_required_argument_exits(self, monkeypatch, missing):
        arguments = {
            "--manifest": ["m.csv"],
            "--run-name": ["r"],
            "--feature": ["mfcc"],
        }

        argv = ["run_feature_extraction.py"]

        for flag, values in arguments.items():
            if flag != missing:
                argv.extend([flag, *values])

        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            run_feature_extraction.parse_args()


class TestRunOneFeature:

    def test_creates_run_folder_and_extracts_features(self, tmp_path):
        project = make_project(tmp_path)

        results = run_feature_extraction.run_one_feature(
            "rms_test",
            make_args(project),
        )

        run_dir = project.results_root / "test_run" / "rms_test"

        assert (run_dir / RUN_CONFIG_NAME).exists()
        assert (run_dir / MANIFEST_SNAPSHOT_NAME).exists()
        assert (run_dir / run_feature_extraction.RESULTS_FILE_NAME).exists()

        assert results["status"].tolist() == ["ok", "ok"]
        assert results["rms_value"].tolist() == pytest.approx(
            [0.1 / np.sqrt(2)] * 2,
            rel=1e-3,
        )

    def test_rerun_resumes_without_reprocessing(self, tmp_path, capsys):
        project = make_project(tmp_path)

        run_feature_extraction.run_one_feature("rms_test", make_args(project))
        capsys.readouterr()

        results = run_feature_extraction.run_one_feature(
            "rms_test",
            make_args(project),
        )

        output = capsys.readouterr().out

        assert "Resuming run" in output
        assert "Samples to process: 0" in output
        assert len(results) == 2

    def test_resume_uses_snapshot_not_rebuilt_manifest(self, tmp_path):
        project = make_project(tmp_path)

        run_feature_extraction.run_one_feature("rms_test", make_args(project))

        snapshot = (
            project.results_root / "test_run" / "rms_test" / MANIFEST_SNAPSHOT_NAME
        )
        snapshot_before = snapshot.read_bytes()

        # Changing the manifest changes its hash, so resuming is refused
        # rather than silently processing a different sample set.
        manifest = pd.read_csv(project.manifest)
        manifest.iloc[:1].to_csv(project.manifest, index=False)

        with pytest.raises(ValueError, match="manifest_sha256"):
            run_feature_extraction.run_one_feature("rms_test", make_args(project))

        assert snapshot.read_bytes() == snapshot_before

    def test_changed_feature_config_refuses_to_resume(self, tmp_path):
        project = make_project(tmp_path)

        run_feature_extraction.run_one_feature("rms_test", make_args(project))

        write_json(project.config_dir / "rms_test.json", {"new_setting": 1})

        with pytest.raises(ValueError, match="config.feature.new_setting"):
            run_feature_extraction.run_one_feature("rms_test", make_args(project))

    def test_different_run_name_creates_separate_run(self, tmp_path):
        project = make_project(tmp_path)

        run_feature_extraction.run_one_feature("rms_test", make_args(project))
        write_json(project.config_dir / "rms_test.json", {"new_setting": 1})

        run_feature_extraction.run_one_feature(
            "rms_test",
            make_args(project, run_name="second_run"),
        )

        assert (project.results_root / "second_run" / "rms_test").is_dir()

    def test_numeric_looking_sample_ids_stay_strings(self, tmp_path):
        project = make_project(tmp_path, sample_ids=("001", "002"))

        results = run_feature_extraction.run_one_feature(
            "rms_test",
            make_args(project),
        )

        assert results["sample_id"].tolist() == ["001", "002"]

    def test_missing_processed_audio_is_an_error_row(self, tmp_path):
        project = make_project(tmp_path)
        (project.processed_root / "audio" / "sample_2.wav").unlink()

        results = run_feature_extraction.run_one_feature(
            "rms_test",
            make_args(project),
        )

        assert results["status"].tolist() == ["ok", "error"]
        assert results.loc[1, "error_type"] == "FileNotFoundError"


class TestMain:

    def test_missing_manifest_raises_error(self, tmp_path, monkeypatch):
        project = make_project(tmp_path)
        project.manifest = tmp_path / "missing.csv"

        set_argv(monkeypatch, project, "rms_test")

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            run_feature_extraction.main()

    def test_single_feature_runs(self, tmp_path, monkeypatch):
        project = make_project(tmp_path)
        set_argv(monkeypatch, project, "rms_test")

        run_feature_extraction.main()

        results_path = (
            project.results_root
            / "test_run"
            / "rms_test"
            / run_feature_extraction.RESULTS_FILE_NAME
        )

        assert len(pd.read_csv(results_path)) == 2

    def test_single_feature_error_is_raised_normally(self, tmp_path, monkeypatch):
        project = make_project(tmp_path)
        set_argv(monkeypatch, project, "not_a_feature")

        with pytest.raises(ValueError, match="Unknown feature"):
            run_feature_extraction.main()

    def test_failing_feature_does_not_stop_other_features(self, tmp_path, monkeypatch, capsys):
        project = make_project(tmp_path)
        set_argv(monkeypatch, project, "not_a_feature", "rms_test")

        with pytest.raises(SystemExit) as exit_info:
            run_feature_extraction.main()

        output = capsys.readouterr().out

        assert "1 feature(s) failed: ['not_a_feature']" in str(exit_info.value.code)
        assert "not_a_feature: FAILED (ValueError" in output
        assert "rms_test: 2 ok, 0 sample errors" in output
        assert (project.results_root / "test_run" / "rms_test").is_dir()

    def test_all_features_succeeding_does_not_exit(self, tmp_path, monkeypatch, capsys):
        project = make_project(tmp_path)

        monkeypatch.setattr(
            run_feature_extraction,
            "run_one_feature",
            lambda feature_name, args: pd.DataFrame({"status": ["ok", "error"]}),
        )

        set_argv(monkeypatch, project, "mfcc", "rms_test")

        run_feature_extraction.main()

        output = capsys.readouterr().out

        assert "mfcc: 1 ok, 1 sample errors" in output
        assert "rms_test: 1 ok, 1 sample errors" in output

    def test_fail_fast_stops_at_first_failing_feature(self, tmp_path, monkeypatch, capsys):
        project = make_project(tmp_path)
        calls = []

        def fake_run_one_feature(feature_name, args):
            calls.append(feature_name)
            raise RuntimeError("feature crashed")

        monkeypatch.setattr(
            run_feature_extraction,
            "run_one_feature",
            fake_run_one_feature,
        )

        set_argv(
            monkeypatch,
            project,
            "mfcc",
            "rms_test",
            extra=["--fail-fast"],
        )

        with pytest.raises(SystemExit):
            run_feature_extraction.main()

        output = capsys.readouterr().out

        assert calls == ["mfcc"]
        assert "rms_test: not run" in output

    def test_all_runs_every_listed_feature(self, tmp_path, monkeypatch):
        project = make_project(tmp_path)
        calls = []

        def fake_run_one_feature(feature_name, args):
            calls.append(feature_name)
            return pd.DataFrame({"status": ["ok"]})

        monkeypatch.setattr(
            run_feature_extraction,
            "run_one_feature",
            fake_run_one_feature,
        )

        set_argv(monkeypatch, project, "all")

        run_feature_extraction.main()

        assert calls == run_feature_extraction.FEATURE_MODULES


# run using:
# python -m pytest tests/test_run_feature_extraction.py -v
