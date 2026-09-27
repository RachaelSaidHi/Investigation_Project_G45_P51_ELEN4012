"""Tests for AfriSpeech-specific study sample selection."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.select_afrispeech import (
    ADULT_AGE_GROUPS,
    AFRISPEECH_GROUPS,
    _filter_adults,
    _select_accent_groups,
    select_afrispeech_study,
)
from src.data.schema import MANIFEST_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "full_manifest.csv"


def make_manifest_row(**overrides):
    """Return a valid synthetic AfriSpeech manifest row."""

    defaults = {
        "sample_id": "afrispeech_test",
        "dataset": "afrispeech",
        "source_split": "test",
        "source_sample_id": "test",
        "filepath": "afrispeech/audio/test.wav",
        "processed_filepath": "",
        "speaker_id": "afrispeech_spk_test",
        "accent_raw": "hausa",
        "accent_group": "",
        "country": "nigeria",
        "native_language": "",
        "birthplace": "",
        "sex": "female",
        "age": "",
        "age_group": "19-25",
        "age_onset": "",
        "duration_s": 5.0,
        "original_sample_rate": 44100,
        "domain": "general",
        "reference_transcript": "test transcript",
    }

    defaults.update(overrides)
    return defaults


def make_manifest(rows):
    """Build a manifest DataFrame from synthetic rows."""

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


@pytest.fixture
def candidate_manifest():
    """Create valid study rows plus deliberately excluded rows."""

    rows = [
        # Valid selections.
        make_manifest_row(
            sample_id="afrispeech_hausa",
            speaker_id="afrispeech_spk_1",
            accent_raw="hausa",
            source_split="test",
            age_group="19-25",
        ),
        make_manifest_row(
            sample_id="afrispeech_yoruba",
            speaker_id="afrispeech_spk_2",
            accent_raw="yoruba",
            source_split="test",
            age_group="26-40",
        ),

        # Matching accent and split, but under 18.
        make_manifest_row(
            sample_id="afrispeech_hausa_minor",
            speaker_id="afrispeech_spk_3",
            accent_raw="hausa",
            source_split="test",
            age_group="<18yrs",
        ),

        # Matching accent and split, but unknown age.
        make_manifest_row(
            sample_id="afrispeech_yoruba_unknown_age",
            speaker_id="afrispeech_spk_4",
            accent_raw="yoruba",
            source_split="test",
            age_group="",
        ),

        # Selected accent, but wrong split.
        make_manifest_row(
            sample_id="afrispeech_hausa_train",
            speaker_id="afrispeech_spk_5",
            accent_raw="hausa",
            source_split="train",
            age_group="19-25",
        ),

        # Acquired accent, but not selected for study.
        make_manifest_row(
            sample_id="afrispeech_afemai",
            speaker_id="afrispeech_spk_6",
            accent_raw="afemai",
            source_split="test",
            age_group="19-25",
        ),

        # Matching accent label, but from another dataset.
        make_manifest_row(
            sample_id="saa_hausa",
            speaker_id="saa_spk_1",
            dataset="saa",
            accent_raw="hausa",
            source_split="",
            native_language="hausa",
            age=30,
        ),
    ]

    return make_manifest(rows)


class TestAccentGroupSelection:

    def test_all_expected_groups_are_selected(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert set(result["accent_group"]) == set(AFRISPEECH_GROUPS.keys())

    def test_matching_accents_and_splits_are_selected(self, candidate_manifest):
        """Selection happens before age filtering, so minors and
        unknown-age rows should be included at this stage."""

        result = _select_accent_groups(candidate_manifest)

        assert set(result["sample_id"]) == {
            "afrispeech_hausa",
            "afrispeech_yoruba",
            "afrispeech_hausa_minor",
            "afrispeech_yoruba_unknown_age",
        }

    def test_accent_group_labels_are_assigned_correctly(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert (
            result.loc[result["accent_raw"] == "hausa", "accent_group"] == "hausa"
        ).all()

        assert (
            result.loc[result["accent_raw"] == "yoruba", "accent_group"] == "yoruba"
        ).all()

    def test_wrong_splits_are_excluded(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "afrispeech_hausa_train" not in result["sample_id"].values

    def test_unselected_accents_are_excluded(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "afrispeech_afemai" not in result["sample_id"].values

    def test_non_afrispeech_rows_are_excluded(self, candidate_manifest):
        result = _select_accent_groups(candidate_manifest)

        assert "saa_hausa" not in result["sample_id"].values
        assert (result["dataset"] == "afrispeech").all()

    def test_input_manifest_is_not_modified(self, candidate_manifest):
        _select_accent_groups(candidate_manifest)

        assert (candidate_manifest["accent_group"] == "").all()

    def test_missing_required_group_raises_error(self, candidate_manifest):
        manifest = candidate_manifest[
            candidate_manifest["accent_raw"] != "yoruba"
        ].copy()

        with pytest.raises(
            ValueError,
            match="No AfriSpeech samples found for study group 'yoruba'",
        ):
            _select_accent_groups(manifest)


class TestAdultFilter:

    @pytest.mark.parametrize("age_group", sorted(ADULT_AGE_GROUPS))
    def test_known_adult_age_groups_are_kept(self, age_group):
        result = _filter_adults(
            make_manifest([make_manifest_row(age_group=age_group)])
        )

        assert len(result) == 1

    def test_under_18_group_is_removed(self):
        result = _filter_adults(
            make_manifest([make_manifest_row(age_group="<18yrs")])
        )

        assert len(result) == 0

    def test_unknown_age_is_removed(self):
        result = _filter_adults(
            make_manifest([make_manifest_row(age_group="")])
        )

        assert len(result) == 0

    def test_unrecognised_age_group_is_removed(self):
        """The allowlist prevents a new source category from
        entering the study without an explicit decision."""

        result = _filter_adults(
            make_manifest([make_manifest_row(age_group="56-70")])
        )

        assert len(result) == 0

    def test_mixed_ages_retains_only_eligible(self):
        manifest = make_manifest([
            make_manifest_row(
                sample_id="afrispeech_adult",
                speaker_id="afrispeech_spk_1",
                age_group="19-25",
            ),
            make_manifest_row(
                sample_id="afrispeech_minor",
                speaker_id="afrispeech_spk_2",
                age_group="<18yrs",
            ),
            make_manifest_row(
                sample_id="afrispeech_unknown",
                speaker_id="afrispeech_spk_3",
                age_group="",
            ),
        ])

        result = _filter_adults(manifest)

        assert set(result["sample_id"]) == {"afrispeech_adult"}
        assert set(result["age_group"]).issubset(ADULT_AGE_GROUPS)


class TestSelectAfriSpeechStudy:

    def test_combines_selection_and_age_filter(self, candidate_manifest):
        """Minors matching an accent group should be selected
        then filtered, leaving only eligible adults."""

        result = select_afrispeech_study(candidate_manifest)

        assert set(result["sample_id"]) == {
            "afrispeech_hausa",
            "afrispeech_yoruba",
        }

        assert set(result["accent_group"]) == {"hausa", "yoruba"}
        assert set(result["age_group"]).issubset(ADULT_AGE_GROUPS)


# ======================================================================
# Real-data sanity checks — skips if full manifest is not built
# ======================================================================


@pytest.fixture(scope="module")
def real_afrispeech_study():
    """Select AfriSpeech study rows from the installed full manifest."""

    if not FULL_MANIFEST.exists():
        pytest.skip("Full manifest has not been built")

    full = pd.read_csv(FULL_MANIFEST, keep_default_na=False)

    # Only run if AfriSpeech data is actually in the manifest.
    if (full["dataset"] == "afrispeech").sum() == 0:
        pytest.skip("No AfriSpeech data in full manifest")

    return select_afrispeech_study(full)


class TestRealAfriSpeechStudy:

    def test_all_expected_accent_groups_present(self, real_afrispeech_study):
        assert set(real_afrispeech_study["accent_group"]) == set(AFRISPEECH_GROUPS.keys())

    def test_every_accent_group_has_samples(self, real_afrispeech_study):
        counts = real_afrispeech_study["accent_group"].value_counts()
        assert (counts > 0).all()

    def test_all_samples_are_afrispeech(self, real_afrispeech_study):
        assert (real_afrispeech_study["dataset"] == "afrispeech").all()

    def test_no_duplicate_sample_ids(self, real_afrispeech_study):
        assert real_afrispeech_study["sample_id"].is_unique

    def test_all_age_groups_are_eligible(self, real_afrispeech_study):
        assert set(real_afrispeech_study["age_group"]).issubset(ADULT_AGE_GROUPS)

    def test_all_accent_groups_are_assigned(self, real_afrispeech_study):
        assert not real_afrispeech_study["accent_group"].isna().any()
        empty = real_afrispeech_study["accent_group"].astype(str).str.strip().eq("")
        assert not empty.any()


# run using:
# python -m pytest tests/test_select_afrispeech.py -v