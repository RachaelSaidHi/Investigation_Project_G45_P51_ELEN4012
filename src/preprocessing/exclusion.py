"""Quality-control exclusion rules for the study audio.

The rules are applied to measurements produced by quality.py. This module
contains the inclusion policy only; audio measurements are calculated
separately.
"""

import pandas as pd

# QC thresholds
MIN_DURATION_S = 5.0
MAX_DURATION_S = 60.0
MIN_HF_ENERGY_PCT = 0.05

REASON_DURATION_TOO_SHORT = "duration_too_short"
REASON_DURATION_TOO_LONG = "duration_too_long"
REASON_BANDWIDTH_TOO_LOW = "bandwidth_too_low"

REASON_SEPARATOR = ";"

REQUIRED_QUALITY_COLUMNS = [
    "processed_duration_s",
    "hf_energy_pct",
]


def get_exclusion_reasons(
    duration_s: float,
    hf_energy_pct: float,
) -> list[str]:
    """Return all QC exclusion reasons that apply to one recording."""

    reasons = []

    if duration_s < MIN_DURATION_S:
        reasons.append(
            REASON_DURATION_TOO_SHORT
        )

    if duration_s > MAX_DURATION_S:
        reasons.append(
            REASON_DURATION_TOO_LONG
        )

    if hf_energy_pct < MIN_HF_ENERGY_PCT:
        reasons.append(
            REASON_BANDWIDTH_TOO_LOW
        )

    return reasons


def should_exclude(
    duration_s: float,
    hf_energy_pct: float,
) -> bool:
    """Return whether a recording fails any current QC rule."""

    return bool(
        get_exclusion_reasons(
            duration_s,
            hf_energy_pct,
        )
    )


def apply_exclusion_rules(
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """Add exclusion decisions and reasons to a quality table."""

    missing_columns = (
        set(REQUIRED_QUALITY_COLUMNS)
        - set(quality.columns)
    )

    if missing_columns:
        raise ValueError(
            "Quality table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Missing measurements must not silently pass the exclusion rules.
    # Comparisons with NaN return False and could otherwise retain a
    # recording for which QC was not completed successfully.
    for column in REQUIRED_QUALITY_COLUMNS:
        if quality[column].isna().any():
            raise ValueError(
                "Quality table contains missing values in "
                f"column: {column}"
            )

    exclusion_reasons = [
        get_exclusion_reasons(
            duration_s,
            hf_energy_pct,
        )
        for duration_s, hf_energy_pct in zip(
            quality["processed_duration_s"],
            quality["hf_energy_pct"],
        )
    ]

    result = quality.copy()

    result["exclude"] = [
        bool(reasons)
        for reasons in exclusion_reasons
    ]

    result["exclude_reason"] = [
        REASON_SEPARATOR.join(reasons)
        for reasons in exclusion_reasons
    ]

    return result