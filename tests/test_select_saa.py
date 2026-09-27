"""Tests for SAA-specific study sample selection."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.select_saa import (
    ACCENT_GROUPS,
    MAX_AGE,
    MIN_AGE,
    _filter_adults,
    _select_accent_groups,
    select_saa_study,
)
from src.data.schema import MANIFEST_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "full_manifest.csv"


def make_manifest_row(**overrides):
    """Return a valid synthetic manifest row with optional overrides."""

    defaults = {
        "sample_id": "saa_test",
        "dataset": "saa",
        "source_split": "",
        "source_sample_id": "test",
        "filepath": "saa/recordings/test.mp3",
        "processed_filepath": "",
        "speaker_id": "saa_spk_test",
        "accent_raw": "english",
        "accent_group": "",
        "country": "usa",
        "native_language": "english",
        "birthplace": "",
        "sex": "female",
        "age": 30,
        "age_group": "",
        "age_onset": 0,
        "duration_s": 20.0,
        "original_sample_rate": 44100,
        "domain": "read_fixed",
        "reference_transcript": "please call stella",
    }

    defaults.update(overrides)
    return defaults


def make_manifest(rows):
    """Build a manifest DataFrame from synthetic rows."""

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


@pytest.fixture
def candidate_manifest():
    """Create all seven valid accent groups plus deliberately excluded rows."""

    rows = [
        make_manifest_row(
            sample_id="saa_american",
            speaker_id="saa_spk_1",
            native_language="english",
            country="usa",
        ),
        make_manifest_row(
            sample_id="saa_amharic",
            speaker_id="saa_spk_2",
            native_language="amharic",
            country="ethiopia",
        ),
        make_manifest_row(
            sample_id="saa_hausa",
            speaker_id="saa_spk_3",
            native_language="hausa",
            country="nigeria",
        ),
        make_manifest_row(
            sample_id="saa_kiswahili",
            speaker_id="saa_spk_4",
            native_language="kiswahili",
            country="tanzania",
        ),
        make_manifest_row(
            sample_id="saa_tigrigna",
            speaker_id="saa_spk_5",
            native_language="tigrigna",
            country="eritrea",
        ),
        make_manifest_row(
            sample_id="saa_yoruba",
            speaker_id="saa_spk_6",
            native_language="yoruba",
            country="nigeria",
        ),
        make_manifest_row(
            sample_id="saa_hindi",
            speaker_id="saa_spk_7",
            native_language="hindi",
            country="india",
        ),

        # Native English, but not American.
        make_manifest_row(
            sample_id="saa_english_uk",
            speaker_id="saa_spk_8",
            native_language="english",
            country="uk",
        ),

        # American country, but not native English.
        make_manifest_row(
            sample_id="saa_spanish_usa",
            speaker_id="saa_spk_9",
            native_language="spanish",
            country="usa",
        ),

        # Unselected accent.
        make_manifest_row(
            sample_id="saa_mandarin",
            speaker_id="saa_spk_10",
            native_language="mandarin",
            country="china",
        ),

        # Matching metadata, but from another dataset.
        make_manifest_row(
            sample_id="afrispeech_hausa",
            speaker_id="afrispeech_spk_1",
            dataset="afrispeech",
            native_language="hausa",
            country="nigeria",
        ),
    ]

    return make_manifest(rows)


class TestAccentGroupSelection:

    def test_all_expected_groups_are_selected(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert set(result["accent_group"]) == set(ACCENT_GROUPS.keys())

    def test_exactly_one_synthetic_sample_per_group(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)
        counts = result["accent_group"].value_counts()

        assert len(result) == 7
        assert (counts == 1).all()

    def test_accent_group_labels_are_assigned_correctly(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)
        labels = dict(zip(result["sample_id"], result["accent_group"]))

        expected = {
            "saa_american": "american_english",
            "saa_amharic": "amharic",
            "saa_hausa": "hausa",
            "saa_kiswahili": "kiswahili",
            "saa_tigrigna": "tigrigna",
            "saa_yoruba": "yoruba",
            "saa_hindi": "hindi",
        }

        assert labels == expected

    def test_american_english_requires_usa_country(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "saa_american" in result["sample_id"].values
        assert "saa_english_uk" not in result["sample_id"].values

    def test_american_english_requires_native_english(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "saa_spanish_usa" not in result["sample_id"].values

    def test_unselected_accents_are_excluded(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "saa_mandarin" not in result["sample_id"].values

    def test_non_saa_rows_are_excluded(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "afrispeech_hausa" not in result["sample_id"].values
        assert (result["dataset"] == "saa").all()

    def test_input_manifest_is_not_modified(self, candidate_manifest):
        _select_accent_groups(candidate_manifest)

        assert (candidate_manifest["accent_group"] == "").all()

    def test_missing_required_group_raises_error(self, candidate_manifest):
        manifest = candidate_manifest[
            candidate_manifest["native_language"] != "yoruba"
        ].copy()

        with pytest.raises(
            ValueError,
            match="No SAA samples found for accent group 'yoruba'",
        ):
            _select_accent_groups(manifest)


class TestAdultFilter:

    def test_eligible_adult_is_kept(self):
        result = _filter_adults(make_manifest([make_manifest_row(age=30)]))

        assert len(result) == 1

    def test_minimum_age_boundary_is_kept(self):
        result = _filter_adults(make_manifest([make_manifest_row(age=MIN_AGE)]))

        assert len(result) == 1

    def test_below_minimum_age_is_removed(self):
        result = _filter_adults(make_manifest([make_manifest_row(age=MIN_AGE - 1)]))

        assert len(result) == 0

    def test_maximum_age_boundary_is_kept(self):
        result = _filter_adults(make_manifest([make_manifest_row(age=MAX_AGE)]))

        assert len(result) == 1

    def test_above_maximum_age_is_removed(self):
        result = _filter_adults(make_manifest([make_manifest_row(age=MAX_AGE + 1)]))

        assert len(result) == 0

    def test_unknown_age_is_removed(self):
        result = _filter_adults(make_manifest([make_manifest_row(age=None)]))

        assert len(result) == 0

    def test_non_numeric_age_is_removed(self):
        result = _filter_adults(make_manifest([make_manifest_row(age="unknown")]))

        assert len(result) == 0

    def test_mixed_ages_retains_only_eligible(self):
        manifest = make_manifest([
            make_manifest_row(sample_id="saa_child", speaker_id="saa_spk_1", age=7),
            make_manifest_row(sample_id="saa_adult", speaker_id="saa_spk_2", age=30),
            make_manifest_row(sample_id="saa_elderly", speaker_id="saa_spk_3", age=90),
            make_manifest_row(sample_id="saa_unknown", speaker_id="saa_spk_4", age=None),
        ])

        result = _filter_adults(manifest)
        ages = pd.to_numeric(result["age"])

        assert set(result["sample_id"]) == {"saa_adult"}
        assert (ages >= MIN_AGE).all()
        assert (ages <= MAX_AGE).all()


class TestSelectSAAStudy:

    def test_combines_selection_and_age_filter(self, candidate_manifest):
        """A minor matching an accent group should be selected but then filtered."""

        minor = make_manifest([
            make_manifest_row(
                sample_id="saa_yoruba_minor",
                speaker_id="saa_spk_minor",
                native_language="yoruba",
                country="nigeria",
                age=12,
            ),
        ])

        manifest = pd.concat([candidate_manifest, minor], ignore_index=True)
        result = select_saa_study(manifest)

        assert set(result["accent_group"]) == set(ACCENT_GROUPS.keys())
        assert "saa_yoruba_minor" not in result["sample_id"].values
        assert (pd.to_numeric(result["age"]) >= MIN_AGE).all()


# ======================================================================
# Real-data sanity checks — skips if full manifest is not built
# ======================================================================


@pytest.fixture(scope="module")
def real_saa_study():
    """Select SAA study rows from the installed full manifest."""

    if not FULL_MANIFEST.exists():
        pytest.skip("Full manifest has not been built")

    full = pd.read_csv(FULL_MANIFEST, keep_default_na=False)
    return select_saa_study(full)


class TestRealSAAStudy:

    def test_all_expected_accent_groups_present(self, real_saa_study):
        assert set(real_saa_study["accent_group"]) == set(ACCENT_GROUPS.keys())

    def test_every_accent_group_has_samples(self, real_saa_study):
        counts = real_saa_study["accent_group"].value_counts()
        assert (counts > 0).all()

    def test_no_duplicate_sample_ids(self, real_saa_study):
        assert real_saa_study["sample_id"].is_unique

    def test_no_duplicate_speaker_ids(self, real_saa_study):
        assert real_saa_study["speaker_id"].is_unique

    def test_all_samples_are_saa(self, real_saa_study):
        assert (real_saa_study["dataset"] == "saa").all()

    def test_all_accent_groups_are_assigned(self, real_saa_study):
        assert not real_saa_study["accent_group"].isna().any()
        empty = real_saa_study["accent_group"].astype(str).str.strip().eq("")
        assert not empty.any()

    def test_all_ages_are_within_eligibility_range(self, real_saa_study):
        ages = pd.to_numeric(real_saa_study["age"])
        assert (ages >= MIN_AGE).all()
        assert (ages <= MAX_AGE).all()


# run using:
# python -m pytest tests/test_select_saa.py -v