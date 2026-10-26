"""
Tests for the shared run-management helpers (run folders and checkpoints).
"""

import hashlib
import json

import pandas as pd
import pytest

from src.common.run_management import (
    DEFAULT_IGNORED_CONFIG_KEYS,
    MANIFEST_SNAPSHOT_NAME,
    RUN_CONFIG_NAME,
    _flatten_config,
    _is_ignored,
    file_sha256,
    load_existing_results,
    prepare_run_directory,
    save_results_atomic,
    upsert_result,
    validate_resume_config,
)


COLUMNS = [
    "sample_id",
    "manifest_index",
    "status",
    "value",
]

UNORDERED_COLUMNS = [
    "sample_id",
    "status",
    "value",
]


def make_results(rows=None, columns=COLUMNS):
    """
    Return a results table with the given rows.
    """

    return pd.DataFrame(
        rows or [],
        columns=columns,
    )


def make_record(sample_id, manifest_index, status="ok", value=1.0):
    """
    Return one result record.
    """

    return {
        "sample_id": sample_id,
        "manifest_index": manifest_index,
        "status": status,
        "value": value,
    }


def make_run_config(**overrides):
    """
    Return a nested run configuration.
    """

    config = {
        "manifest_sha256": "abc123",
        "feature": {
            "name": "mfcc",
            "version": "1",
        },
        "config": {
            "feature": {
                "n_mfcc": 13,
            },
        },
        "created_at_utc": "2026-01-01T00:00:00+00:00",
    }

    config.update(overrides)

    return config


def write_manifest(path):
    """
    Write a small manifest CSV and return its path.
    """

    pd.DataFrame(
        {
            "sample_id": ["sample_1", "sample_2"],
            "dataset": ["saa", "afrispeech"],
        }
    ).to_csv(
        path,
        index=False,
    )

    return path


class TestLoadExistingResults:

    def test_missing_file_returns_empty_table(self, tmp_path):
        results = load_existing_results(
            tmp_path / "missing.csv",
            COLUMNS,
        )

        assert results.empty
        assert results.columns.tolist() == COLUMNS

    def test_existing_file_is_loaded(self, tmp_path):
        path = tmp_path / "results.csv"

        make_results(
            [
                make_record("sample_1", 0),
                make_record("sample_2", 1, value=2.5),
            ]
        ).to_csv(path, index=False)

        results = load_existing_results(path, COLUMNS)

        assert results["sample_id"].tolist() == ["sample_1", "sample_2"]
        assert results["value"].tolist() == [1.0, 2.5]

    def test_sample_ids_keep_leading_zeros(self, tmp_path):
        path = tmp_path / "results.csv"

        make_results(
            [
                make_record("007", 0),
                make_record("010", 1),
            ]
        ).to_csv(path, index=False)

        results = load_existing_results(path, COLUMNS)

        assert results["sample_id"].tolist() == ["007", "010"]

    def test_columns_are_returned_in_requested_order(self, tmp_path):
        path = tmp_path / "results.csv"

        make_results(
            [make_record("sample_1", 0)]
        )[["value", "status", "manifest_index", "sample_id"]].to_csv(
            path,
            index=False,
        )

        results = load_existing_results(path, COLUMNS)

        assert results.columns.tolist() == COLUMNS

    def test_extra_columns_are_dropped(self, tmp_path):
        path = tmp_path / "results.csv"

        table = make_results([make_record("sample_1", 0)])
        table["old_column"] = "x"
        table.to_csv(path, index=False)

        results = load_existing_results(path, COLUMNS)

        assert "old_column" not in results.columns

    def test_missing_columns_raise_error(self, tmp_path):
        path = tmp_path / "results.csv"

        make_results(
            [make_record("sample_1", 0)]
        ).drop(columns="value").to_csv(path, index=False)

        with pytest.raises(
            ValueError,
            match=r"missing columns: \['value'\]",
        ):
            load_existing_results(path, COLUMNS)

    def test_duplicate_sample_ids_raise_error(self, tmp_path):
        path = tmp_path / "results.csv"

        make_results(
            [
                make_record("sample_1", 0),
                make_record("sample_1", 0),
            ]
        ).to_csv(path, index=False)

        with pytest.raises(
            ValueError,
            match="Existing results contains duplicate sample_id",
        ):
            load_existing_results(path, COLUMNS)

    def test_blank_sample_id_raises_error(self, tmp_path):
        path = tmp_path / "results.csv"

        make_results(
            [
                make_record("sample_1", 0),
                make_record("", 1),
            ]
        ).to_csv(path, index=False)

        with pytest.raises(
            ValueError,
            match="missing or empty sample_id",
        ):
            load_existing_results(path, COLUMNS)


class TestUpsertResult:

    def test_first_record_is_inserted_into_empty_table(self):
        results = upsert_result(
            make_results(),
            make_record("sample_1", 0),
            COLUMNS,
        )

        assert len(results) == 1
        assert results.columns.tolist() == COLUMNS
        assert results.iloc[0]["sample_id"] == "sample_1"

    def test_new_sample_is_appended(self):
        results = make_results([make_record("sample_1", 0)])

        results = upsert_result(
            results,
            make_record("sample_2", 1),
            COLUMNS,
        )

        assert results["sample_id"].tolist() == ["sample_1", "sample_2"]

    def test_existing_sample_is_replaced(self):
        results = make_results(
            [
                make_record("sample_1", 0, status="error", value=None),
                make_record("sample_2", 1),
            ]
        )

        results = upsert_result(
            results,
            make_record("sample_1", 0, status="ok", value=9.0),
            COLUMNS,
        )

        assert len(results) == 2
        assert results["sample_id"].tolist().count("sample_1") == 1

        row = results.set_index("sample_id").loc["sample_1"]

        assert row["status"] == "ok"
        assert row["value"] == 9.0

    def test_rows_are_sorted_by_manifest_index(self):
        results = make_results(
            [
                make_record("sample_a", 0),
                make_record("sample_c", 2),
            ]
        )

        results = upsert_result(
            results,
            make_record("sample_b", 1),
            COLUMNS,
        )

        assert results["sample_id"].tolist() == [
            "sample_a",
            "sample_b",
            "sample_c",
        ]
        assert results.index.tolist() == [0, 1, 2]

    def test_without_order_column_new_rows_go_last(self):
        results = make_results(
            [
                {"sample_id": "sample_b", "status": "ok", "value": 1.0},
            ],
            columns=UNORDERED_COLUMNS,
        )

        results = upsert_result(
            results,
            {"sample_id": "sample_a", "status": "ok", "value": 2.0},
            UNORDERED_COLUMNS,
        )

        assert results["sample_id"].tolist() == ["sample_b", "sample_a"]

    def test_missing_record_fields_become_empty(self):
        results = upsert_result(
            make_results(),
            {"sample_id": "sample_1", "manifest_index": 0, "status": "error"},
            COLUMNS,
        )

        assert pd.isna(results.iloc[0]["value"])

    def test_extra_record_fields_are_ignored(self):
        record = make_record("sample_1", 0)
        record["unexpected"] = "ignored"

        results = upsert_result(
            make_results(),
            record,
            COLUMNS,
        )

        assert results.columns.tolist() == COLUMNS

    def test_numeric_sample_id_matches_string_id(self):
        results = make_results([make_record("7", 0, value=1.0)])

        results = upsert_result(
            results,
            make_record(7, 0, value=2.0),
            COLUMNS,
        )

        assert len(results) == 1
        assert results.iloc[0]["value"] == 2.0

    def test_original_table_is_not_modified(self):
        results = make_results([make_record("sample_1", 0)])
        original = results.copy()

        upsert_result(
            results,
            make_record("sample_2", 1),
            COLUMNS,
        )

        pd.testing.assert_frame_equal(results, original)


class TestSaveResultsAtomic:

    def test_results_are_written(self, tmp_path):
        path = tmp_path / "results.csv"
        results = make_results([make_record("sample_1", 0)])

        save_results_atomic(results, path, COLUMNS)

        saved = pd.read_csv(path)

        assert saved.columns.tolist() == COLUMNS
        assert saved["sample_id"].tolist() == ["sample_1"]

    def test_parent_directories_are_created(self, tmp_path):
        path = tmp_path / "run" / "feature" / "results.csv"

        save_results_atomic(make_results(), path, COLUMNS)

        assert path.exists()

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        path = tmp_path / "results.csv"

        save_results_atomic(make_results(), path, COLUMNS)

        assert not (tmp_path / "results.csv.tmp").exists()
        assert [p.name for p in tmp_path.iterdir()] == ["results.csv"]

    def test_existing_checkpoint_is_replaced(self, tmp_path):
        path = tmp_path / "results.csv"

        save_results_atomic(
            make_results([make_record("sample_1", 0)]),
            path,
            COLUMNS,
        )

        save_results_atomic(
            make_results(
                [
                    make_record("sample_1", 0),
                    make_record("sample_2", 1),
                ]
            ),
            path,
            COLUMNS,
        )

        assert len(pd.read_csv(path)) == 2

    def test_only_listed_columns_are_written_in_order(self, tmp_path):
        path = tmp_path / "results.csv"
        results = make_results([make_record("sample_1", 0)])
        results["scratch"] = "not saved"

        save_results_atomic(
            results,
            path,
            ["value", "sample_id"],
        )

        assert pd.read_csv(path).columns.tolist() == ["value", "sample_id"]

    def test_failed_write_keeps_previous_checkpoint(self, tmp_path, monkeypatch):
        path = tmp_path / "results.csv"

        save_results_atomic(
            make_results([make_record("sample_1", 0)]),
            path,
            COLUMNS,
        )

        # function to simulate an os.replace() failure
        def fail_replace(source, destination):
            raise OSError("simulated crash")

        monkeypatch.setattr(
            "src.common.run_management.os.replace",
            fail_replace,
        )

        with pytest.raises(OSError, match="simulated crash"):
            save_results_atomic(
                make_results(
                    [
                        make_record("sample_1", 0),
                        make_record("sample_2", 1),
                    ]
                ),
                path,
                COLUMNS,
            )

        assert pd.read_csv(path)["sample_id"].tolist() == ["sample_1"]

    def test_saved_results_round_trip_through_loader(self, tmp_path):
        path = tmp_path / "results.csv"
        results = make_results(
            [
                make_record("001", 0, value=0.5),
                make_record("002", 1, status="error", value=None),
            ]
        )

        save_results_atomic(results, path, COLUMNS)
        loaded = load_existing_results(path, COLUMNS)

        assert loaded["sample_id"].tolist() == ["001", "002"]
        assert loaded["status"].tolist() == ["ok", "error"]


class TestFileSha256:

    def test_hash_matches_hashlib(self, tmp_path):
        path = tmp_path / "manifest.csv"
        content = b"sample_id,dataset\nsample_1,saa\n"
        path.write_bytes(content)

        assert file_sha256(path) == hashlib.sha256(content).hexdigest()

    def test_empty_file_has_known_hash(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_bytes(b"")

        assert file_sha256(path) == hashlib.sha256(b"").hexdigest()

    def test_large_file_is_hashed_in_chunks(self, tmp_path):
        path = tmp_path / "large.bin"
        content = b"x" * (3 * 1024 * 1024 + 17)
        path.write_bytes(content)

        assert file_sha256(path) == hashlib.sha256(content).hexdigest()

    def test_changed_file_changes_hash(self, tmp_path):
        path = tmp_path / "manifest.csv"

        path.write_text("sample_id\na\n", encoding="utf-8")
        first = file_sha256(path)

        path.write_text("sample_id\nb\n", encoding="utf-8")
        second = file_sha256(path)

        assert first != second


class TestFlattenConfig:

    def test_nested_keys_are_joined_with_dots(self):
        flat = _flatten_config(
            {
                "feature": {
                    "name": "mfcc",
                    "settings": {"n_mfcc": 13},
                },
                "seed": 0,
            }
        )

        assert flat == {
            "feature.name": "mfcc",
            "feature.settings.n_mfcc": 13,
            "seed": 0,
        }

    def test_lists_are_kept_whole(self):
        flat = _flatten_config({"thresholds": [25, 30]})

        assert flat == {"thresholds": [25, 30]}

    def test_empty_config_flattens_to_empty_dict(self):
        assert _flatten_config({}) == {}


class TestIsIgnored:

    def test_exact_key_is_ignored(self):
        assert _is_ignored("created_at_utc", ("created_at_utc",))

    def test_nested_key_under_ignored_key_is_ignored(self):
        assert _is_ignored("paths.manifest", ("paths",))
        assert _is_ignored("paths.output", ("paths",))

    def test_key_sharing_only_a_prefix_is_not_ignored(self):
        # "paths_extra" is not nested under "paths".
        assert not _is_ignored("paths_extra", ("paths",))

    def test_no_ignored_keys(self):
        assert not _is_ignored("feature.name", ())

    def test_key_does_not_include_ignored(self):
        assert not _is_ignored("feature.name", ("created_at_utc"),)


class TestValidateResumeConfig:

    def test_identical_configs_pass(self):
        validate_resume_config(
            make_run_config(),
            make_run_config(),
        )

    def test_created_at_is_ignored_by_default(self):
        assert "created_at_utc" in DEFAULT_IGNORED_CONFIG_KEYS

        validate_resume_config(
            make_run_config(created_at_utc="2026-01-01"),
            make_run_config(created_at_utc="2026-02-02"),
        )

    def test_changed_value_raises_error(self):
        current = make_run_config(manifest_sha256="different")

        with pytest.raises(
            ValueError,
            match="manifest_sha256: previous='abc123', current='different'",
        ):
            validate_resume_config(
                make_run_config(),
                current,
            )

    def test_changed_nested_value_is_reported_with_dotted_key(self):
        current = make_run_config(
            config={"feature": {"n_mfcc": 20}},
        )

        with pytest.raises(
            ValueError,
            match="config.feature.n_mfcc: previous=13, current=20",
        ):
            validate_resume_config(
                make_run_config(),
                current,
            )

    def test_key_only_in_current_config_raises_error(self):
        current = make_run_config(new_setting=True)

        with pytest.raises(
            ValueError,
            match="new_setting: previous='<missing>', current=True",
        ):
            validate_resume_config(
                make_run_config(),
                current,
            )

    def test_key_only_in_previous_config_raises_error(self):
        previous = make_run_config(old_setting=1)

        with pytest.raises(
            ValueError,
            match="old_setting: previous=1, current='<missing>'",
        ):
            validate_resume_config(
                previous,
                make_run_config(),
            )

    def test_every_difference_is_listed(self):
        current = make_run_config(
            manifest_sha256="different",
            feature={"name": "pitch", "version": "2"},
        )

        with pytest.raises(ValueError) as error:
            validate_resume_config(
                make_run_config(),
                current,
            )

        message = str(error.value)

        assert "manifest_sha256" in message
        assert "feature.name" in message
        assert "feature.version" in message
        assert "--run-name" in message

    def test_custom_ignored_key_ignores_nested_values(self):
        previous = make_run_config(paths={"manifest": "/machine_a/m.csv"})
        current = make_run_config(paths={"manifest": "/machine_b/m.csv"})

        validate_resume_config(
            previous,
            current,
            ignored_keys=("created_at_utc", "paths"),
        )

    def test_custom_ignored_keys_replace_defaults(self):
        previous = make_run_config(created_at_utc="a")
        current = make_run_config(created_at_utc="b")

        with pytest.raises(ValueError, match="created_at_utc"):
            validate_resume_config(
                previous,
                current,
                ignored_keys=("paths",),
            )

    def test_list_values_are_compared_whole(self):
        with pytest.raises(ValueError, match="thresholds"):
            validate_resume_config(
                {"thresholds": [25, 30, 5]},
                {"thresholds": [30, 25, 5]},
            )


class TestPrepareRunDirectory:

    def test_new_run_creates_directory_config_and_snapshot(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "runs" / "dev_v1" / "mfcc"
        run_config = make_run_config()

        assert not run_dir.exists()

        snapshot_path = prepare_run_directory(
            run_dir,
            manifest_path,
            run_config,
        )

        assert run_dir.is_dir()
        assert snapshot_path == run_dir / MANIFEST_SNAPSHOT_NAME

        with open(run_dir / RUN_CONFIG_NAME, encoding="utf-8") as file:
            assert json.load(file) == run_config

        assert snapshot_path.read_bytes() == manifest_path.read_bytes()

    def test_new_run_prints_created_message(self, tmp_path, capsys):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"

        prepare_run_directory(run_dir, manifest_path, make_run_config())

        assert "Created run" in capsys.readouterr().out

    def test_matching_existing_run_is_resumed(self, tmp_path, capsys):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"

        prepare_run_directory(run_dir, manifest_path, make_run_config())

        snapshot_path = prepare_run_directory(
            run_dir,
            manifest_path,
            make_run_config(created_at_utc="later"),
        )

        assert snapshot_path == run_dir / MANIFEST_SNAPSHOT_NAME
        assert "Resuming run" in capsys.readouterr().out

    def test_resume_does_not_overwrite_snapshot_or_config(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"
        original_config = make_run_config()

        prepare_run_directory(run_dir, manifest_path, original_config)
        original_snapshot = (run_dir / MANIFEST_SNAPSHOT_NAME).read_bytes()

        # The source manifest is rebuilt after the run started.
        manifest_path.write_text("sample_id\nchanged\n", encoding="utf-8")

        prepare_run_directory(
            run_dir,
            manifest_path,
            make_run_config(created_at_utc="later"),
        )

        assert (run_dir / MANIFEST_SNAPSHOT_NAME).read_bytes() == original_snapshot

        with open(run_dir / RUN_CONFIG_NAME, encoding="utf-8") as file:
            assert json.load(file) == original_config

    def test_changed_config_refuses_to_resume(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"

        prepare_run_directory(run_dir, manifest_path, make_run_config())

        with pytest.raises(
            ValueError,
            match="configuration has changed",
        ):
            prepare_run_directory(
                run_dir,
                manifest_path,
                make_run_config(feature={"name": "mfcc", "version": "2"}),
            )

    def test_custom_ignored_keys_are_passed_through(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"

        prepare_run_directory(
            run_dir,
            manifest_path,
            make_run_config(paths={"manifest": "/a"}),
            ignored_keys=("created_at_utc", "paths"),
        )

        prepare_run_directory(
            run_dir,
            manifest_path,
            make_run_config(paths={"manifest": "/b"}),
            ignored_keys=("created_at_utc", "paths"),
        )

    def test_existing_directory_without_config_raises_error(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / MANIFEST_SNAPSHOT_NAME).write_text("sample_id\n", encoding="utf-8")

        with pytest.raises(
            FileNotFoundError,
            match=f"no {RUN_CONFIG_NAME}",
        ):
            prepare_run_directory(run_dir, manifest_path, make_run_config())

    def test_existing_directory_without_snapshot_raises_error(self, tmp_path):
        manifest_path = write_manifest(tmp_path / "manifest.csv")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with open(run_dir / RUN_CONFIG_NAME, "w", encoding="utf-8") as file:
            json.dump(make_run_config(), file)

        with pytest.raises(
            FileNotFoundError,
            match=f"no {MANIFEST_SNAPSHOT_NAME}",
        ):
            prepare_run_directory(run_dir, manifest_path, make_run_config())

    def test_missing_source_manifest_raises_error_for_new_run(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare_run_directory(
                tmp_path / "run",
                tmp_path / "missing_manifest.csv",
                make_run_config(),
            )


# run using:
# python -m pytest tests/test_run_management.py -v
