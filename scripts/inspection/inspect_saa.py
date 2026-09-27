from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

metadata_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "saa"
    / "speakers_all.csv"
)

recordings_dir = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "saa"
    / "recordings"
    / "recordings"
)

df = pd.read_csv(metadata_path)

# Only rows for which the dataset says audio exists
usable = df[df["file_missing?"] == False].copy()

print("Usable recordings:")
print(len(usable))

print("\nCountries with the most speakers:")
print(usable["country"].value_counts().head(20))

print("\nSpeakers from countries currently relevant to us:")
countries = [
    "usa",
    "nigeria",
    "kenya",
    "south africa",
    "india",
]

print("\nAll countries by number of usable speakers:")
country_counts = (
    usable["country"]
    .str.lower()
    .value_counts()
)

print(country_counts.to_string())

print("\nMost common native languages:")
print(
    usable["native_language"]
    .str.lower()
    .value_counts()
    .head(30)
    .to_string()
)

relevant = usable[
    usable["country"].str.lower().isin(countries)
]

print(
    relevant[
        [
            "speakerid",
            "filename",
            "country",
            "native_language",
            "sex",
            "age",
            "age_onset",
        ]
    ].to_string(index=False)
)

print("\nCounts by relevant country:")
print(
    relevant["country"]
    .str.lower()
    .value_counts()
)

print("\nNumber of MP3 files in recordings/:")
print(len(list(recordings_dir.glob("*.mp3"))))

audio_files = list(recordings_dir.glob("*.mp3"))

audio_names = {
    path.stem.lower()
    for path in audio_files
}

expected_names = {
    str(name).strip().lower()
    for name in usable["filename"]
}

missing_from_disk = sorted(expected_names - audio_names)
extra_on_disk = sorted(audio_names - expected_names)

print("\nMetadata rows marked usable:")
print(len(expected_names))

print("\nMP3 files actually present:")
print(len(audio_names))

print("\nMarked usable but missing from disk:")
print(missing_from_disk)

print("\nMP3 files with no matching usable metadata row:")
print(extra_on_disk)
