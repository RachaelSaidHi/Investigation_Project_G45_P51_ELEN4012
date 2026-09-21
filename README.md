# Investigation_Project_G45_P51_ELEN4012


## 1. Clone and install

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## 2. Set Up Whisper
1. Create a Hugging Face account
2. Run:
```
hf auth login
```
3. Install PyTorch
* if you have a NVIDIA GPU:
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```
* otherwise CPU only:
```
python -m pip install torch torchvision
```


## 3. Acquire the Speech Accent Archive


1. Download the dataset from Kaggle as zip:
https://www.kaggle.com/datasets/rtatman/speech-accent-archive   
**The dataset requires approximately 920 MB of disk space after extraction**

2. In the project repository, create the following directory structure if it does not already exist:

```
data/
└── raw/
    └── saa/
```

3. Extract the contents of the downloaded ZIP file into:
```   
data/raw/saa
```

4. After extraction, the folder structure is:

```
data/
└── raw/
    └── saa/
        ├── speakers_all.csv
        ├── reading-passage.txt
        └── recordings/
            └── recordings/
                ├── afrikaans1.mp3
                ├── afrikaans2.mp3
                ├── ...
                └── yoruba5.mp3
```
The nested `recordings/recordings/` comes from the original Kaggle structure   
The `data/` directory is in `.gitignore` and is not committed.

## 4. Build the manifests

Run these three scripts in order from the project root:

```bash
# Step 1: Build the full SAA manifest (all 2,138 recordings)
python -m scripts.build_full_saa_manifest

# Step 2: Filter to the study accent groups and apply adult filter
python -m scripts.build_study_manifest

# Step 3: Build the small dev subset for quick iteration
python -m scripts.build_dev_manifest
```

This produces three files in `data/manifests/`:

| File | Contents | Use |
|---|---|---|
| `full_manifest.csv` | Every SAA recording in standardised format | Not used directly — input to the next step |
| `study_manifest.csv` | ~409 recordings from the 7 selected accent groups | The actual experimental dataset |
| `dev_manifest.csv` | ~14 recordings, 1 female + 1 male per group | Quick pipeline development and testing |

## 5. Verify

```bash
python -m pytest -v
```

All tests should pass. If SAA is not installed, the SAA loader tests will skip rather than fail.

## 5. What the manifests contain

Both `study_manifest.csv` and `dev_manifest.csv` have the same columns. The important ones for the ASR pipeline are:

- `sample_id` — unique identifier used to key results
- `filepath` — path to the audio file relative to `data/raw/`
- `accent_group` — the experimental accent category
- `reference_transcript` — the correct transcript for WER computation
- `sex` — needed later for sex-matched analysis
- `speaker_id` — one recording per speaker in SAA

To load audio for a given manifest row:

```python
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/raw")
manifest = pd.read_csv("data/manifests/study_manifest.csv")

row = manifest.iloc[0]
audio_path = DATA_ROOT / row["filepath"]
```

## Project Repo Structure
```
project/
│
├── README.md
├── .gitignore
├── requirements.txt
├── src/                        # contains the core code and functions
│   | 
│   ├── asr/
│   │   └── .py files
│   │
│   ├── dsp/
│   │   └── .py files
│   │
│   └── analysis/
│       └── .py files
│
├── scripts/                    # like python notebooks maybe
│   └── python notebooks
│
├── data/
│   ├── README.md
│   ├── raw/                    # GITIGNORED
│   ├── preprocessed/           # GITIGNORED
│   └── manifests/
│       └── .csv
│
├── results/
│   ├── asr/                 
│   ├── dsp/
│   └── analysis/
│
└── tests/
    └── .py files

```
