"""Check whether WADA-SNR saturation is caused by digital silence.

For each dev recording, reports the share of exact-zero samples in the raw
and processed audio, the longest run of zeros, and the WADA-SNR estimate
with zero samples removed.

Run from project root:
    python -m scripts.inspection.check_wada_saturation
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.preprocessing.quality import estimate_wada_snr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DEV_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "dev_manifest.csv"
QC_TABLE = PROCESSED_ROOT / "audio_qc_dev_manifest.csv"


def longest_zero_run(audio: np.ndarray) -> int:
    """Return the length of the longest run of exact-zero samples."""

    is_zero = np.concatenate([[0], (audio == 0).astype(int), [0]])
    edges = np.diff(is_zero)

    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    if len(starts) == 0:
        return 0

    return int((ends - starts).max())


def main() -> None:
    manifest = pd.read_csv(DEV_MANIFEST, keep_default_na=False)
    quality = pd.read_csv(QC_TABLE, keep_default_na=False)

    rows = quality.merge(
        manifest[["sample_id", "filepath"]],
        on="sample_id",
    )

    results = []

    for row in rows.itertuples(index=False):
        raw, raw_sample_rate = librosa.load(RAW_ROOT / row.filepath, sr=None, mono=True)
        processed, processed_sample_rate = sf.read(PROCESSED_ROOT / row.processed_filepath, dtype="float32")

        nonzero = processed[processed != 0]
        wada_nonzero_db, _ = estimate_wada_snr(nonzero)

        results.append({
            "sample_id": row.sample_id,
            "wada_snr_db": float(row.wada_snr_db),
            "raw_zero_pct": 100 * np.mean(raw == 0),
            "processed_zero_pct": 100 * np.mean(processed == 0),
            "longest_zero_run_s": longest_zero_run(processed) / processed_sample_rate,
            "wada_without_zeros_db": wada_nonzero_db,
        })

    pd.set_option("display.width", 200)
    print(pd.DataFrame(results).round(2).to_string(index=False))


if __name__ == "__main__":
    main()