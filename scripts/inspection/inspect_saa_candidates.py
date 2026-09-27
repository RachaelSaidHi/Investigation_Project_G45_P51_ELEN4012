from pathlib import Path

import pandas as pd

from data.load_saa import load_saa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"


def format_countries(series: pd.Series) -> str:
    counts = series.value_counts()

    return ", ".join(
        f"{country} ({count})"
        for country, count in counts.items()
    )


def main() -> None:
    manifest = load_saa(DATA_ROOT)

    rows = []

    for native_language, group in manifest.groupby("native_language"):
        ages = pd.to_numeric(group["age"], errors="coerce").dropna()

        sex_counts = group["sex"].value_counts()

        rows.append(
            {
                "native_language": native_language,
                "speakers": group["speaker_id"].nunique(),
                "minutes": group["duration_s"].sum() / 60,
                "female": int(sex_counts.get("female", 0)),
                "male": int(sex_counts.get("male", 0)),
                "median_age": ages.median() if not ages.empty else None,
                "min_age": ages.min() if not ages.empty else None,
                "max_age": ages.max() if not ages.empty else None,
                "countries": format_countries(group["country"]),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["speakers", "native_language"],
        ascending=[False, True],
    )

    # Small groups are unlikely to be useful as headline Level-1 groups.
    candidates = summary[summary["speakers"] >= 3].copy()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 80)

    print("\nSAA CANDIDATE GROUPS — >= 3 SPEAKERS\n")

    print(
        candidates.to_string(
            index=False,
            formatters={
                "minutes": lambda x: f"{x:.1f}",
                "median_age": lambda x: f"{x:.1f}",
                "min_age": lambda x: f"{x:.1f}",
                "max_age": lambda x: f"{x:.1f}",
            },
        )
    )

    print("\nTOTAL SAA DATA")
    print(f"Recordings: {len(manifest)}")
    print(f"Hours:      {manifest['duration_s'].sum() / 3600:.2f}")


if __name__ == "__main__":
    main()