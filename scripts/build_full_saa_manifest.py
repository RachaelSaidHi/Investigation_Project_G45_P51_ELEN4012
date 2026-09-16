from pathlib import Path

import pandas as pd

from data.load_saa import load_saa
from src.data.validation import validate_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
OUTPUT_PATH = MANIFEST_DIR / "full_manifest.csv"

def main() -> None:
    # Each loader returns the same schema.
    manifests = [
        load_saa(DATA_ROOT),

        # TODO:
        # load_afrispeech(DATA_ROOT),
        # load_edacc(DATA_ROOT),
    ]

    full_manifest = pd.concat(manifests, ignore_index=True,)

    validate_manifest(full_manifest, DATA_ROOT)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True,)

    full_manifest.to_csv(OUTPUT_PATH, index=False,)

    print(f"Manifest written to: {OUTPUT_PATH}")
    print(f"Samples: {len(full_manifest)}")
    print()
    print("Samples by dataset:")
    print(full_manifest["dataset"].value_counts())


if __name__ == "__main__":
    main()